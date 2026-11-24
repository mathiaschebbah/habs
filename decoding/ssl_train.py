"""GPU Job 2 for Study 08: supervised neural baselines and Multimodal PhysioJEPA suite.

Runs on the HABS H100 NVL backend (~25-30 min). Loads all 22,974 aligned 15-s
multimodal trials (62-ch EEG 128 Hz, 2-ch ECG 128 Hz, 2-ch EDA 16 Hz, 2-ch Resp
16 Hz) plus strictly dev-only unlabeled resting-state windows once into RAM,
then executes the complete frozen Study-08 GPU benchmark on
``STUDY08_SALT = "habs-ssl-2026-11-24-b"``:

1. Multi-seed (seeds 0, 1, 2) 3x3 dev CV + one-time sealed test for:
   - ``EEGNet`` (2.2k params, raw 62-ch EEG)
   - ``EEGConformer`` (237k params, raw 62-ch EEG)
   - ``MultimodalConformer_Scratch`` (0.90M params, matched 4-mod backbone, no SSL)
   - ``PhysioJEPA_Frozen`` (fold-isolated SSL pretraining + frozen linear probes)
   - ``PhysioJEPA_Finetuned`` (fold-isolated SSL pretraining + lightweight fine-tune)
2. Representation dynamics vs participant-identity diagnostics (within-session
   & cross-day subject ID, R^2_subject, pre-vs-video, early-vs-late,
   arrow-of-time, effective rank, temporal-shuffle degradation) comparing
   pre-stimulus RMS normalization vs fixed-scale normalization (Identity
   Shortcut test).
3. Systematic ablations (all fold-isolated on dev 3x3 CV + sealed test):
   - Modality (all 4, EEG-only, Body-only, EEG+ECG, EEG+EDA, EEG+Resp)
   - Temporal window (0-5 s, 0-10 s, -3..10 s, -3..12 s)
   - SSL strategy (cross-modal JEPA, future world-model JEPA, masked-block JEPA,
     raw-signal MAE control, fixed-scale JEPA, JEPA + 11.1h dev resting-state)
   - Data-size scaling (10%, 25%, 50%, 100% of dev participants, PhysioJEPA vs
     MultimodalConformer_Scratch).

Outputs
-------
- Private ``/out/features/jepa_embeddings.parquet``: frozen PhysioJEPA phase
  embeddings + EEG-only early embeddings for Job-3 fusion & cross-modal probing.
- Private ``/out/features/jepa_preds.parquet``: per-trial predictions across
  all models and seeds for Job-3 exact paired bootstrap gains vs Study 07.
- Public ``/out/public/ssl_jepa_suite.json``: complete aggregate metrics, 95%
  CIs, seed SDs, representation diagnostics, and GPU cost telemetry.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from decoding import common, ssl_eval
from decoding.ssl_data import load_dataset_multimodal, load_dev_resting_windows
from decoding.ssl_models import (
    EEGConformer,
    EEGNet,
    PhysioJEPA,
    PhysioJEPAConfig,
    count_parameters,
    effective_rank,
)
from decoding.stress import label
from habs.temporal.data import physical_id

STUDY08_SALT = "habs-ssl-2026-11-24-b"
SEEDS = (0, 1, 2)
SSL_EPOCHS = 55
SUP_EPOCHS = 24
FT_EPOCHS = 18
BATCH_SIZE = 256


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device_store(
    data: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor | np.ndarray]:
    """Pin float16/bool multimodal arrays once in H100 HBM3 (~6.3 GB / 93 GB)."""
    out: dict[str, torch.Tensor | np.ndarray] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.dtype.kind in ("f", "b", "i", "u"):
            out[k] = torch.from_numpy(v).to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _index_tensor(
    arr: np.ndarray | torch.Tensor,
    idx: np.ndarray | torch.Tensor,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gather rows from either a GPU-resident tensor or a CPU NumPy array."""
    if isinstance(arr, torch.Tensor):
        idx_t = (
            idx
            if isinstance(idx, torch.Tensor)
            else torch.as_tensor(idx, dtype=torch.long, device=arr.device)
        )
        out = arr.index_select(0, idx_t)
        return (
            out.to(device=device, dtype=dtype) if dtype is not None else out.to(device)
        )
    idx_np = idx.cpu().numpy() if isinstance(idx, torch.Tensor) else idx
    t = torch.from_numpy(arr[idx_np])
    return (
        t.to(device=device, dtype=dtype, non_blocking=True)
        if dtype is not None
        else t.to(device, non_blocking=True)
    )


def _slice_batch(
    data: dict[str, np.ndarray | torch.Tensor],
    idx: np.ndarray | torch.Tensor,
    device: torch.device,
    *,
    time_slice: tuple[int, int] = (0, 15),
    use_fixed_eeg: bool = False,
    shuffle_time_rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Build GPU tensor batch from HBM3-resident tensors (or CPU arrays in tests)."""
    t0, t1 = time_slice
    eeg = _index_tensor(data["eeg"], idx, device, dtype=torch.float32)[:, :, t0:t1, :]
    if use_fixed_eeg:
        sc = _index_tensor(data["eeg_scale"], idx, device, dtype=torch.float32)
        eeg = torch.clamp(eeg * sc[:, :, None, None] / 25.0, -10.0, 10.0)
    ecg = _index_tensor(data["ecg"], idx, device, dtype=torch.float32)[:, :, t0:t1, :]
    eda = _index_tensor(data["eda"], idx, device, dtype=torch.float32)[:, :, t0:t1, :]
    resp = _index_tensor(data["resp"], idx, device, dtype=torch.float32)[:, :, t0:t1, :]
    valid = _index_tensor(data["valid"], idx, device, dtype=torch.bool)[:, :, t0:t1]
    if shuffle_time_rng is not None:
        t_len = t1 - t0
        perm = torch.from_numpy(shuffle_time_rng.permutation(t_len)).to(device)
        eeg = eeg[:, :, perm, :]
        ecg = ecg[:, :, perm, :]
        eda = eda[:, :, perm, :]
        resp = resp[:, :, perm, :]
        valid = valid[:, :, perm]
    return {"eeg": eeg, "ecg": ecg, "eda": eda, "resp": resp, "valid": valid}


def _targets_for_subset(
    frame: pd.DataFrame, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary stress plus person-centered standardized valence & arousal."""
    sub = frame.iloc[idx]
    y_s = label(sub).to_numpy(np.float32)
    ppl = sub["participant_id"]
    yv = (
        sub["rating_valence"] - sub["rating_valence"].groupby(ppl).transform("mean")
    ).to_numpy(np.float32)
    ya = (
        sub["rating_arousal"] - sub["rating_arousal"].groupby(ppl).transform("mean")
    ).to_numpy(np.float32)
    yv = yv / max(float(np.std(yv)), 1e-4)
    ya = ya / max(float(np.std(ya)), 1e-4)
    return y_s, yv, ya


def _internal_val_split(
    frame: pd.DataFrame, train_idx: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Person-disjoint 85/15 split strictly inside ``train_idx`` for early stopping."""
    people = np.sort(frame.iloc[train_idx]["participant_id"].unique())
    rng = np.random.default_rng(seed + 1009)
    perm = rng.permutation(people)
    n_val = max(2, int(round(0.15 * len(people))))
    val_p = set(perm[:n_val].tolist())
    pids = frame.iloc[train_idx]["participant_id"].to_numpy()
    is_val = np.array([p in val_p for p in pids], dtype=bool)
    return train_idx[~is_val], train_idx[is_val]


def train_supervised_eeg_model(
    model_kind: str,
    data: dict[str, np.ndarray],
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    device: torch.device,
    *,
    seed: int = 0,
    epochs: int = SUP_EPOCHS,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train EEGNet or EEGConformer on 62-ch 10-s [0, 10) EEG and predict."""
    _set_seed(seed)
    tr_sub, val_sub = _internal_val_split(frame, train_idx, seed)
    ys_tr, yv_tr, ya_tr = _targets_for_subset(frame, tr_sub)
    ys_va, yv_va, ya_va = _targets_for_subset(frame, val_sub)

    model: nn.Module = EEGNet() if model_kind == "eegnet" else EEGConformer()
    model = model.to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_rate = float(np.clip(ys_tr.mean(), 0.05, 0.95))
    pos_w = torch.tensor([(1.0 - pos_rate) / pos_rate], device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    def _get_eeg10(indices: np.ndarray) -> torch.Tensor:
        # Bins 3..13 correspond to [0, 10) s -> (B, 62, 1280) directly in HBM3
        x = _index_tensor(data["eeg"], indices, device, dtype=torch.float32)[
            :, :, 3:13, :
        ]
        return x.reshape(x.shape[0], x.shape[1], -1)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(tr_sub))
        for s in range(0, len(order), BATCH_SIZE):
            bi = order[s : s + BATCH_SIZE]
            x = _get_eeg10(tr_sub[bi])
            ts = torch.from_numpy(ys_tr[bi]).to(device)
            tv = torch.from_numpy(yv_tr[bi]).to(device)
            ta = torch.from_numpy(ya_tr[bi]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = model(x)
                loss = (
                    F.binary_cross_entropy_with_logits(
                        out["stress"].float(), ts, pos_weight=pos_w
                    )
                    + 0.5 * F.smooth_l1_loss(out["valence"].float(), tv)
                    + 0.5 * F.smooth_l1_loss(out["arousal"].float(), ta)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        # Internal validation loss
        model.eval()
        v_losses = []
        with torch.inference_mode():
            for s in range(0, len(val_sub), BATCH_SIZE):
                sl = slice(s, min(len(val_sub), s + BATCH_SIZE))
                x = _get_eeg10(val_sub[sl])
                ts = torch.from_numpy(ys_va[sl]).to(device)
                tv = torch.from_numpy(yv_va[sl]).to(device)
                ta = torch.from_numpy(ya_va[sl]).to(device)
                out = model(x)
                vl = (
                    F.binary_cross_entropy_with_logits(
                        out["stress"].float(), ts, pos_weight=pos_w
                    )
                    + 0.5 * F.smooth_l1_loss(out["valence"].float(), tv)
                    + 0.5 * F.smooth_l1_loss(out["arousal"].float(), ta)
                )
                v_losses.append(float(vl.item()))
        v_mean = float(np.mean(v_losses)) if v_losses else float("inf")
        if v_mean < best_loss:
            best_loss = v_mean
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    n_ev = len(eval_idx)
    p_s = np.zeros(n_ev, dtype=np.float32)
    p_v = np.zeros(n_ev, dtype=np.float32)
    p_a = np.zeros(n_ev, dtype=np.float32)
    embeds = np.zeros((n_ev, model.embed_dim), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n_ev, BATCH_SIZE):
            e = min(n_ev, s + BATCH_SIZE)
            x = _get_eeg10(eval_idx[s:e])
            out = model(x)
            p_s[s:e] = out["stress"].float().cpu().numpy()
            p_v[s:e] = out["valence"].float().cpu().numpy()
            p_a[s:e] = out["arousal"].float().cpu().numpy()
            embeds[s:e] = out["embed"].float().cpu().numpy()

    p_va = p_a / max(float(np.std(p_a)), 1e-6) - p_v / max(float(np.std(p_v)), 1e-6)
    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    del model
    return (
        {
            "stress": p_s,
            "stress_va": p_va.astype(np.float32),
            "valence": p_v,
            "arousal": p_a,
            "embed": embeds,
        },
        {
            "n_params": n_params,
            "train_seconds": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
        },
    )


def pretrain_physio_jepa(
    data: dict[str, np.ndarray],
    train_idx: np.ndarray,
    device: torch.device,
    *,
    strategy: str = "jepa_crossmodal",
    active_modalities: tuple[int, ...] = (0, 1, 2, 3),
    time_slice: tuple[int, int] = (0, 15),
    use_fixed_eeg: bool = False,
    resting_data: dict[str, np.ndarray] | None = None,
    resting_idx: np.ndarray | None = None,
    seed: int = 0,
    epochs: int = SSL_EPOCHS,
) -> tuple[PhysioJEPA, dict]:
    """Pretrain Multimodal PhysioJEPA strictly without labels on ``train_idx``."""
    _set_seed(seed)
    model = PhysioJEPA(PhysioJEPAConfig()).to(device)
    n_params_enc = count_parameters(model.context_encoder)
    n_params_tot = count_parameters(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=0.05)
    n_main = len(train_idx)
    n_rest = (
        len(resting_idx)
        if (resting_data is not None and resting_idx is not None)
        else 0
    )
    n_total = n_main + n_rest
    steps_per_epoch = max(1, math.ceil(n_total / BATCH_SIZE))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, 5 * steps_per_epoch)

    def _lr_mult(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_mult)
    gen = torch.Generator(device="cpu").manual_seed(seed + 77)
    rng = np.random.default_rng(seed + 13)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    step = 0
    first_pred_loss, last_pred_loss = None, None
    last_var_loss, last_erank = None, None

    for ep in range(epochs):
        model.train()
        order = rng.permutation(n_total)
        ep_pred, ep_var = [], []
        last_z = None
        for s in range(0, n_total, BATCH_SIZE):
            bi = order[s : s + BATCH_SIZE]
            main_sel = bi[bi < n_main]
            rest_sel = bi[bi >= n_main] - n_main

            parts = []
            if len(main_sel) > 0:
                parts.append(
                    _slice_batch(
                        data,
                        train_idx[main_sel],
                        device,
                        time_slice=time_slice,
                        use_fixed_eeg=use_fixed_eeg,
                    )
                )
            if (
                len(rest_sel) > 0
                and resting_data is not None
                and resting_idx is not None
            ):
                parts.append(
                    _slice_batch(
                        resting_data,
                        resting_idx[rest_sel],
                        device,
                        time_slice=time_slice,
                        use_fixed_eeg=False,
                    )
                )
            if len(parts) == 1:
                batch = parts[0]
            else:
                batch = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                res = model.ssl_loss(
                    batch,
                    strategy=strategy,
                    active_modalities=active_modalities,
                    generator=gen,
                )
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            model.update_ema(step / max(1, total_steps - 1))
            step += 1
            ep_pred.append(float(res["pred_loss"].item()))
            ep_var.append(float(res["var_loss"].item()))
            last_z = res["z_ctx_flat"]

        mp = float(np.mean(ep_pred))
        if ep == 0:
            first_pred_loss = mp
        last_pred_loss = mp
        last_var_loss = float(np.mean(ep_var))
        if ep == epochs - 1 and last_z is not None:
            last_erank = effective_rank(last_z)

    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    stats = {
        "strategy": strategy,
        "n_pretrain_windows": int(n_total),
        "n_resting_windows": int(n_rest),
        "n_params_encoder": n_params_enc,
        "n_params_total": n_params_tot,
        "ssl_seconds": round(elapsed, 2),
        "peak_vram_gb": round(peak_gb, 2),
        "pred_loss_start": round(first_pred_loss, 4) if first_pred_loss else None,
        "pred_loss_end": round(last_pred_loss, 4) if last_pred_loss else None,
        "var_loss_end": round(last_var_loss, 4) if last_var_loss is not None else None,
        "ctx_effective_rank": round(last_erank, 2) if last_erank else None,
    }
    return model, stats


@torch.inference_mode()
def extract_jepa_features(
    model: PhysioJEPA,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    *,
    active_modalities: tuple[int, ...] = (0, 1, 2, 3),
    time_slice: tuple[int, int] = (0, 15),
    use_fixed_eeg: bool = False,
    use_target_encoder: bool = True,
    shuffle_time_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract phase-structured embeddings from a frozen PhysioJEPA encoder."""
    model.eval()
    n = len(indices)
    d = model.cfg.d_model
    z_ref = np.zeros((n, d), dtype=np.float32)
    z_early = np.zeros((n, d), dtype=np.float32)
    z_late = np.zeros((n, d), dtype=np.float32)
    z_vid = np.zeros((n, d), dtype=np.float32)
    z_eeg_early = np.zeros((n, d), dtype=np.float32)
    embed_delta = np.zeros((n, 3 * d), dtype=np.float32)
    embed_raw = np.zeros((n, 3 * d), dtype=np.float32)
    rng = (
        np.random.default_rng(shuffle_time_seed)
        if shuffle_time_seed is not None
        else None
    )

    for s in range(0, n, BATCH_SIZE):
        e = min(n, s + BATCH_SIZE)
        batch = _slice_batch(
            data,
            indices[s:e],
            device,
            time_slice=time_slice,
            use_fixed_eeg=use_fixed_eeg,
            shuffle_time_rng=rng,
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            rep = model.encode_structured(
                batch,
                active_modalities=active_modalities,
                use_target_encoder=use_target_encoder,
            )
        z_ref[s:e] = rep["z_ref"].float().cpu().numpy()
        z_early[s:e] = rep["z_early"].float().cpu().numpy()
        z_late[s:e] = rep["z_late"].float().cpu().numpy()
        z_vid[s:e] = rep["z_vid"].float().cpu().numpy()
        embed_delta[s:e] = rep["embed_delta"].float().cpu().numpy()
        embed_raw[s:e] = rep["embed_raw"].float().cpu().numpy()
        if 0 in active_modalities:
            m0 = active_modalities.index(0)
            zg = rep["z_grid"][:, m0, :, :].float()  # (B, T, D)
            t_len = zg.shape[1]
            sl_e = slice(3, 8) if t_len >= 13 else slice(0, max(1, t_len // 2))
            z_eeg_early[s:e] = zg[:, sl_e, :].mean(dim=1).cpu().numpy()

    return {
        "z_ref": z_ref,
        "z_early": z_early,
        "z_late": z_late,
        "z_vid": z_vid,
        "z_eeg_early": z_eeg_early,
        "embed_delta": embed_delta,
        "embed_raw": embed_raw,
    }


@torch.inference_mode()
def extract_jepa_window_isolated(
    model: PhysioJEPA,
    data: dict[str, np.ndarray | torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    *,
    active_modalities: tuple[int, ...] = (0, 1, 2, 3),
    use_fixed_eeg: bool = False,
    use_target_encoder: bool = True,
) -> dict[str, np.ndarray]:
    """Encode equal-length windows independently with identical positional embeddings.

    Prevents the Transformer's additive ``time_emb`` from acting as a trivial
    positional-encoding shortcut when probing ``pre_vs_video``, ``early_vs_late``,
    and ``arrow_of_time``.

    - 3-s matched windows (all receive ``time_emb[0:3]``): ``ref [-3, 0)``,
      ``early3 [0, 3)``, ``late3 [7, 10)``.
    - 5-s matched windows (both receive ``time_emb[0:5]``): ``early5 [0, 5)``,
      ``late5 [5, 10)``.
    """
    model.eval()
    enc = model.target_encoder if use_target_encoder else model.context_encoder
    n = len(indices)
    d = model.cfg.d_model
    windows = {
        "ref3": (0, 3),
        "early3": (3, 6),
        "late3": (10, 13),
        "early5": (3, 8),
        "late5": (8, 13),
    }
    out = {k: np.zeros((n, d), dtype=np.float32) for k in windows}
    for key, t_sl in windows.items():
        for s in range(0, n, BATCH_SIZE):
            e = min(n, s + BATCH_SIZE)
            batch = _slice_batch(
                data,
                indices[s:e],
                device,
                time_slice=t_sl,
                use_fixed_eeg=use_fixed_eeg,
            )
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                z, valid, _, _ = enc(batch, active_modalities=active_modalities)
                v = valid.float().unsqueeze(-1)
                pooled = (z.float() * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)
            out[key][s:e] = pooled.cpu().numpy()
    return out


@torch.inference_mode()
def extract_jepa_eeg_only_early(
    model: PhysioJEPA,
    data: dict[str, np.ndarray | torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    *,
    use_fixed_eeg: bool = False,
    use_target_encoder: bool = True,
) -> np.ndarray:
    """Extract leak-free EEG-only early ``[0, 5) s`` delta for brain->body forecasting.

    Runs the frozen target encoder strictly on ``active_modalities=(0,)`` (EEG
    only) and ``time_slice=(0, 8)`` (``[-3, 5) s``) so no peripheral modality and
    no future time step ``t >= 5 s`` can ever leak through self-attention into
    the early EEG embedding used to forecast late ``[5, 10) s`` autonomic state.
    """
    model.eval()
    enc = model.target_encoder if use_target_encoder else model.context_encoder
    n = len(indices)
    d = model.cfg.d_model
    out = np.zeros((n, d), dtype=np.float32)
    for s in range(0, n, BATCH_SIZE):
        e = min(n, s + BATCH_SIZE)
        batch = _slice_batch(
            data,
            indices[s:e],
            device,
            time_slice=(0, 8),
            use_fixed_eeg=use_fixed_eeg,
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z, valid, _, _ = enc(batch, active_modalities=(0,))
            # Shape is (B, 8, D): bins 0:3 are [-3, 0) ref, bins 3:8 are [0, 5) early
            v = valid.float().unsqueeze(-1)
            z_ref = (z[:, :3, :].float() * v[:, :3, :]).sum(dim=1) / v[:, :3, :].sum(
                dim=1
            ).clamp(min=1.0)
            z_early = (z[:, 3:8, :].float() * v[:, 3:8, :]).sum(dim=1) / v[
                :, 3:8, :
            ].sum(dim=1).clamp(min=1.0)
        out[s:e] = (z_early - z_ref).cpu().numpy()
    return out


def train_supervised_multimodal(
    data: dict[str, np.ndarray],
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    device: torch.device,
    *,
    pretrained_jepa: PhysioJEPA | None = None,
    active_modalities: tuple[int, ...] = (0, 1, 2, 3),
    time_slice: tuple[int, int] = (0, 15),
    use_fixed_eeg: bool = False,
    seed: int = 0,
    epochs: int = SUP_EPOCHS,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train matched MultimodalConformer (scratch or fine-tuned from PhysioJEPA)."""
    _set_seed(seed)
    tr_sub, val_sub = _internal_val_split(frame, train_idx, seed)
    ys_tr, yv_tr, ya_tr = _targets_for_subset(frame, tr_sub)
    ys_va, yv_va, ya_va = _targets_for_subset(frame, val_sub)

    if pretrained_jepa is not None:
        model = copy.deepcopy(pretrained_jepa).to(device)
        # Initialize context encoder from smooth EMA target encoder for fine-tuning
        model.context_encoder.load_state_dict(model.target_encoder.state_dict())
        nn.init.normal_(model.sup_head.weight, std=0.02)
        nn.init.zeros_(model.sup_head.bias)
        param_groups = [
            {"params": model.context_encoder.parameters(), "lr": 2e-4},
            {"params": model.sup_head.parameters(), "lr": 1.5e-3},
        ]
        n_ep = FT_EPOCHS
    else:
        model = PhysioJEPA(PhysioJEPAConfig()).to(device)
        param_groups = [
            {"params": model.context_encoder.parameters(), "lr": 8e-4},
            {"params": model.sup_head.parameters(), "lr": 1e-3},
        ]
        n_ep = epochs

    opt = torch.optim.AdamW(param_groups, weight_decay=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    pos_rate = float(np.clip(ys_tr.mean(), 0.05, 0.95))
    pos_w = torch.tensor([(1.0 - pos_rate) / pos_rate], device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 31)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(n_ep):
        model.train()
        order = rng.permutation(len(tr_sub))
        for s in range(0, len(order), BATCH_SIZE):
            bi = order[s : s + BATCH_SIZE]
            batch = _slice_batch(
                data,
                tr_sub[bi],
                device,
                time_slice=time_slice,
                use_fixed_eeg=use_fixed_eeg,
            )
            ts = torch.from_numpy(ys_tr[bi]).to(device)
            tv = torch.from_numpy(yv_tr[bi]).to(device)
            ta = torch.from_numpy(ya_tr[bi]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = model.supervised_forward(
                    batch,
                    active_modalities=active_modalities,
                    use_delta=(time_slice[0] == 0),
                )
                loss = (
                    F.binary_cross_entropy_with_logits(
                        out["stress"].float(), ts, pos_weight=pos_w
                    )
                    + 0.5 * F.smooth_l1_loss(out["valence"].float(), tv)
                    + 0.5 * F.smooth_l1_loss(out["arousal"].float(), ta)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        v_losses = []
        with torch.inference_mode():
            for s in range(0, len(val_sub), BATCH_SIZE):
                sl = slice(s, min(len(val_sub), s + BATCH_SIZE))
                batch = _slice_batch(
                    data,
                    val_sub[sl],
                    device,
                    time_slice=time_slice,
                    use_fixed_eeg=use_fixed_eeg,
                )
                ts = torch.from_numpy(ys_va[sl]).to(device)
                tv = torch.from_numpy(yv_va[sl]).to(device)
                ta = torch.from_numpy(ya_va[sl]).to(device)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    out = model.supervised_forward(
                        batch,
                        active_modalities=active_modalities,
                        use_delta=(time_slice[0] == 0),
                    )
                    vl = (
                        F.binary_cross_entropy_with_logits(
                            out["stress"].float(), ts, pos_weight=pos_w
                        )
                        + 0.5 * F.smooth_l1_loss(out["valence"].float(), tv)
                        + 0.5 * F.smooth_l1_loss(out["arousal"].float(), ta)
                    )
                v_losses.append(float(vl.item()))
        v_mean = float(np.mean(v_losses)) if v_losses else float("inf")
        if v_mean < best_loss:
            best_loss = v_mean
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    n_ev = len(eval_idx)
    p_s = np.zeros(n_ev, dtype=np.float32)
    p_v = np.zeros(n_ev, dtype=np.float32)
    p_a = np.zeros(n_ev, dtype=np.float32)
    embeds = np.zeros((n_ev, model.cfg.d_model * 3), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n_ev, BATCH_SIZE):
            e = min(n_ev, s + BATCH_SIZE)
            batch = _slice_batch(
                data,
                eval_idx[s:e],
                device,
                time_slice=time_slice,
                use_fixed_eeg=use_fixed_eeg,
            )
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = model.supervised_forward(
                    batch,
                    active_modalities=active_modalities,
                    use_delta=(time_slice[0] == 0),
                )
            p_s[s:e] = out["stress"].float().cpu().numpy()
            p_v[s:e] = out["valence"].float().cpu().numpy()
            p_a[s:e] = out["arousal"].float().cpu().numpy()
            embeds[s:e] = out["embed"].float().cpu().numpy()

    p_va = p_a / max(float(np.std(p_a)), 1e-6) - p_v / max(float(np.std(p_v)), 1e-6)
    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    n_params_enc = count_parameters(model.context_encoder)
    del model
    return (
        {
            "stress": p_s,
            "stress_va": p_va.astype(np.float32),
            "valence": p_v,
            "arousal": p_a,
            "embed": embeds,
        },
        {
            "n_params_encoder": n_params_enc,
            "train_seconds": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
        },
    )


def _summarize_predictions(
    frame: pd.DataFrame,
    dev_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    *,
    draws: int = ssl_eval.DRAWS,
) -> dict:
    """Summarize dev and sealed-test stress + valence/arousal metrics."""
    dev_df = frame[frame["split"] == "dev"]
    test_df = frame[frame["split"] != "dev"]
    out = {
        "dev": {
            "stress_direct": ssl_eval.evaluate_stress(
                dev_df, pd.Series(dev_preds["stress"], index=dev_df.index), draws=draws
            ),
            "stress_va": ssl_eval.evaluate_stress(
                dev_df,
                pd.Series(dev_preds["stress_va"], index=dev_df.index),
                draws=draws,
            ),
            "affect": ssl_eval.evaluate_valence_arousal(
                dev_df,
                pd.Series(dev_preds["valence"], index=dev_df.index),
                pd.Series(dev_preds["arousal"], index=dev_df.index),
                draws=draws,
            ),
        }
    }
    for regime in common.REGIMES:
        rm = (test_df["split"] == regime).to_numpy()
        sub = test_df[rm]
        out[regime] = {
            "stress_direct": ssl_eval.evaluate_stress(
                sub,
                pd.Series(test_preds["stress"][rm], index=sub.index),
                draws=draws,
            ),
            "stress_va": ssl_eval.evaluate_stress(
                sub,
                pd.Series(test_preds["stress_va"][rm], index=sub.index),
                draws=draws,
            ),
            "affect": ssl_eval.evaluate_valence_arousal(
                sub,
                pd.Series(test_preds["valence"][rm], index=sub.index),
                pd.Series(test_preds["arousal"][rm], index=sub.index),
                draws=draws,
            ),
        }
    return out


def _aggregate_seeds(seed_results: list[dict], key_path: tuple[str, ...]) -> dict:
    """Compute mean and SD across seeds for a nested scalar metric."""
    vals = []
    for r in seed_results:
        cur = r
        for k in key_path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is not None and np.isfinite(cur):
            vals.append(float(cur))
    if not vals:
        return {"mean": None, "sd": None, "per_seed": []}
    return {
        "mean": round(float(np.mean(vals)), 4),
        "sd": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
        "per_seed": [round(v, 4) for v in vals],
    }


def run_supervised_cv_and_test(
    name: str,
    kind: str,
    data: dict[str, np.ndarray],
    frame: pd.DataFrame,
    device: torch.device,
    *,
    seeds: tuple[int, ...] = SEEDS,
    active_modalities: tuple[int, ...] = (0, 1, 2, 3),
    time_slice: tuple[int, int] = (0, 15),
    sub_dev_people: set[str] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run strict 3x3 dev CV and sealed test across seeds for a supervised model."""
    dev_mask = (frame["split"] == "dev").to_numpy()
    dev_idx = np.where(dev_mask)[0]
    test_idx = np.where(~dev_mask)[0]
    dev_df = frame.iloc[dev_idx]
    pf = dev_df["person_fold"].to_numpy()
    tf = dev_df["theme_fold"].to_numpy()

    seed_metrics = []
    pred_cols: dict[str, np.ndarray] = {}
    total_gpu_s = 0.0
    max_vram = 0.0
    n_params = 0

    for seed in seeds:
        dev_out = {
            k: np.full(len(dev_idx), np.nan, dtype=np.float32)
            for k in ("stress", "stress_va", "valence", "arousal")
        }
        # 3x3 blocked dev cross-fit
        for p in range(3):
            for t in range(3):
                ev_loc = (pf == p) & (tf == t)
                tr_loc = (pf != p) & (tf != t)
                tr_global = dev_idx[tr_loc]
                if sub_dev_people is not None:
                    tr_pids = frame.iloc[tr_global]["participant_id"].to_numpy()
                    tr_global = tr_global[
                        np.array([pid in sub_dev_people for pid in tr_pids], dtype=bool)
                    ]
                ev_global = dev_idx[ev_loc]
                if kind in ("eegnet", "eeg_conformer"):
                    pr, st = train_supervised_eeg_model(
                        kind,
                        data,
                        frame,
                        tr_global,
                        ev_global,
                        device,
                        seed=seed + p * 3 + t,
                    )
                    n_params = st["n_params"]
                else:
                    pr, st = train_supervised_multimodal(
                        data,
                        frame,
                        tr_global,
                        ev_global,
                        device,
                        active_modalities=active_modalities,
                        time_slice=time_slice,
                        seed=seed + p * 3 + t,
                    )
                    n_params = st["n_params_encoder"]
                total_gpu_s += st["train_seconds"]
                max_vram = max(max_vram, st["peak_vram_gb"])
                for k in dev_out:
                    dev_out[k][ev_loc] = pr[k]

        # Full dev -> sealed test
        tr_full = dev_idx
        if sub_dev_people is not None:
            tr_pids = frame.iloc[tr_full]["participant_id"].to_numpy()
            tr_full = tr_full[
                np.array([pid in sub_dev_people for pid in tr_pids], dtype=bool)
            ]
        if kind in ("eegnet", "eeg_conformer"):
            test_pr, st_te = train_supervised_eeg_model(
                kind, data, frame, tr_full, test_idx, device, seed=seed + 100
            )
        else:
            test_pr, st_te = train_supervised_multimodal(
                data,
                frame,
                tr_full,
                test_idx,
                device,
                active_modalities=active_modalities,
                time_slice=time_slice,
                seed=seed + 100,
            )
        total_gpu_s += st_te["train_seconds"]
        max_vram = max(max_vram, st_te["peak_vram_gb"])

        sm = _summarize_predictions(frame, dev_out, test_pr)
        seed_metrics.append(sm)

        # Store full-table predictions for Job-3 paired bootstrap vs Study 07
        for col_k in ("stress", "stress_va", "valence", "arousal"):
            full_arr = np.full(len(frame), np.nan, dtype=np.float32)
            full_arr[dev_idx] = dev_out[col_k]
            full_arr[test_idx] = test_pr[col_k]
            pred_cols[f"{name}_s{seed}_{col_k}"] = full_arr

    # Seed ensemble predictions (mean across seeds)
    ens_dev, ens_test = {}, {}
    for col_k in ("stress", "stress_va", "valence", "arousal"):
        stacked = np.stack([pred_cols[f"{name}_s{s}_{col_k}"] for s in seeds], axis=0)
        mean_arr = np.mean(stacked, axis=0).astype(np.float32)
        pred_cols[f"{name}_ens_{col_k}"] = mean_arr
        ens_dev[col_k] = mean_arr[dev_idx]
        ens_test[col_k] = mean_arr[test_idx]

    summary = {
        "n_params": n_params,
        "seeds": list(seeds),
        "total_gpu_seconds": round(total_gpu_s, 2),
        "peak_vram_gb": round(max_vram, 2),
        "seed_stats": {
            "dev_stress_direct": _aggregate_seeds(
                seed_metrics, ("dev", "stress_direct", "stress", "auc")
            ),
            "dev_stress_va": _aggregate_seeds(
                seed_metrics, ("dev", "stress_va", "stress", "auc")
            ),
            "dev_mean_rho": _aggregate_seeds(
                seed_metrics, ("dev", "affect", "mean_rho")
            ),
            "test_new_both_stress_direct": _aggregate_seeds(
                seed_metrics, ("test_new_both", "stress_direct", "stress", "auc")
            ),
            "test_new_both_stress_va": _aggregate_seeds(
                seed_metrics, ("test_new_both", "stress_va", "stress", "auc")
            ),
            "test_new_person_stress_va": _aggregate_seeds(
                seed_metrics, ("test_new_person", "stress_va", "stress", "auc")
            ),
            "test_new_video_stress_va": _aggregate_seeds(
                seed_metrics, ("test_new_video", "stress_va", "stress", "auc")
            ),
        },
        "seed0": seed_metrics[0],
        "seed_ensemble": _summarize_predictions(frame, ens_dev, ens_test),
    }
    print(
        f"  DONE {name}: dev_va={summary['seed_stats']['dev_stress_va']['mean']} "
        f"(+-{summary['seed_stats']['dev_stress_va']['sd']}), "
        f"new_both={summary['seed_stats']['test_new_both_stress_va']['mean']}, "
        f"gpu_s={summary['total_gpu_seconds']}",
        flush=True,
    )
    return summary, pred_cols


def run_jepa_variant(
    name: str,
    data: dict[str, np.ndarray],
    frame: pd.DataFrame,
    device: torch.device,
    *,
    strategy: str = "jepa_crossmodal",
    active_modalities: tuple[int, ...] = (0, 1, 2, 3),
    time_slice: tuple[int, int] = (0, 15),
    use_fixed_eeg: bool = False,
    resting_data: dict[str, np.ndarray] | None = None,
    seeds: tuple[int, ...] = (0,),
    run_finetune: bool = False,
    run_diagnostics: bool = False,
    sub_dev_people: set[str] | None = None,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Run fold-isolated PhysioJEPA pretraining, linear probing, and optional FT."""
    dev_mask = (frame["split"] == "dev").to_numpy()
    dev_idx = np.where(dev_mask)[0]
    test_idx = np.where(~dev_mask)[0]
    dev_df = frame.iloc[dev_idx]
    pf = dev_df["person_fold"].to_numpy()
    tf = dev_df["theme_fold"].to_numpy()

    seed_frozen_metrics = []
    seed_ft_metrics = []
    ssl_logs = []
    pred_cols: dict[str, np.ndarray] = {}
    exported_embeds: dict[str, np.ndarray] | None = None
    diagnostics: dict = {}
    total_gpu_s = 0.0
    max_vram = 0.0

    for seed in seeds:
        dev_log = np.full(len(dev_idx), np.nan, dtype=np.float32)
        dev_va = np.full(len(dev_idx), np.nan, dtype=np.float32)
        dev_v = np.full(len(dev_idx), np.nan, dtype=np.float32)
        dev_a = np.full(len(dev_idx), np.nan, dtype=np.float32)

        ft_dev_out = {
            k: np.full(len(dev_idx), np.nan, dtype=np.float32)
            for k in ("stress", "stress_va", "valence", "arousal")
        }

        # Strictly fold-isolated SSL pretraining for each dev person fold p in {0, 1, 2}
        for p in range(3):
            pre_tr_local = pf != p
            pre_tr_global = dev_idx[pre_tr_local]
            if sub_dev_people is not None:
                p_arr = frame.iloc[pre_tr_global]["participant_id"].to_numpy()
                pre_tr_global = pre_tr_global[
                    np.array([pid in sub_dev_people for pid in p_arr], dtype=bool)
                ]
            allowed_p = set(frame.iloc[pre_tr_global]["participant_id"].astype(str))
            rest_idx = None
            if resting_data is not None and len(resting_data["pid"]) > 0:
                rest_idx = np.where(np.isin(resting_data["pid"], list(allowed_p)))[0]

            jepa_p, st_ssl = pretrain_physio_jepa(
                data,
                pre_tr_global,
                device,
                strategy=strategy,
                active_modalities=active_modalities,
                time_slice=time_slice,
                use_fixed_eeg=use_fixed_eeg,
                resting_data=resting_data,
                resting_idx=rest_idx,
                seed=seed * 10 + p,
            )
            total_gpu_s += st_ssl["ssl_seconds"]
            max_vram = max(max_vram, st_ssl["peak_vram_gb"])

            # Extract frozen embeddings for dev rows using fold-isolated encoder jepa_p
            rep_dev = extract_jepa_features(
                jepa_p,
                data,
                dev_idx,
                device,
                active_modalities=active_modalities,
                time_slice=time_slice,
                use_fixed_eeg=use_fixed_eeg,
            )
            embed_key = "embed_delta" if time_slice[0] == 0 else "embed_raw"
            x_dev_c = ssl_eval.person_center_matrix(
                rep_dev[embed_key], dev_df["participant_id"].to_numpy()
            )

            # Evaluate the 3 theme folds of held-out person_fold p
            for t in range(3):
                ev_loc = (pf == p) & (tf == t)
                tr_loc = (pf != p) & (tf != t)
                if sub_dev_people is not None:
                    tr_sub_mask = np.array(
                        [
                            pid in sub_dev_people
                            for pid in dev_df["participant_id"].to_numpy()
                        ],
                        dtype=bool,
                    )
                    tr_loc = tr_loc & tr_sub_mask
                sl, sv, pv, pa = ssl_eval._fit_linear_probes(
                    x_dev_c[tr_loc], dev_df.iloc[tr_loc], x_dev_c[ev_loc], max_pca=128
                )
                dev_log[ev_loc] = sl
                dev_va[ev_loc] = sv
                dev_v[ev_loc] = pv
                dev_a[ev_loc] = pa

                if run_finetune:
                    pr_ft, st_ft = train_supervised_multimodal(
                        data,
                        frame,
                        dev_idx[tr_loc],
                        dev_idx[ev_loc],
                        device,
                        pretrained_jepa=jepa_p,
                        active_modalities=active_modalities,
                        time_slice=time_slice,
                        use_fixed_eeg=use_fixed_eeg,
                        seed=seed * 10 + p * 3 + t,
                    )
                    total_gpu_s += st_ft["train_seconds"]
                    max_vram = max(max_vram, st_ft["peak_vram_gb"])
                    for k in ft_dev_out:
                        ft_dev_out[k][ev_loc] = pr_ft[k]

            del jepa_p
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Full dev-only pretraining (ZERO test people, ZERO test themes!) -> sealed test
        pre_full = dev_idx
        if sub_dev_people is not None:
            p_arr = frame.iloc[pre_full]["participant_id"].to_numpy()
            pre_full = pre_full[
                np.array([pid in sub_dev_people for pid in p_arr], dtype=bool)
            ]
        allowed_dev = set(frame.iloc[pre_full]["participant_id"].astype(str))
        rest_dev_idx = None
        if resting_data is not None and len(resting_data["pid"]) > 0:
            rest_dev_idx = np.where(np.isin(resting_data["pid"], list(allowed_dev)))[0]

        jepa_full, st_full = pretrain_physio_jepa(
            data,
            pre_full,
            device,
            strategy=strategy,
            active_modalities=active_modalities,
            time_slice=time_slice,
            use_fixed_eeg=use_fixed_eeg,
            resting_data=resting_data,
            resting_idx=rest_dev_idx,
            seed=seed * 10 + 9,
        )
        total_gpu_s += st_full["ssl_seconds"]
        max_vram = max(max_vram, st_full["peak_vram_gb"])
        ssl_logs.append(st_full)

        all_idx = np.arange(len(frame))
        rep_all = extract_jepa_features(
            jepa_full,
            data,
            all_idx,
            device,
            active_modalities=active_modalities,
            time_slice=time_slice,
            use_fixed_eeg=use_fixed_eeg,
        )
        embed_key = "embed_delta" if time_slice[0] == 0 else "embed_raw"
        x_all_c = ssl_eval.person_center_matrix(
            rep_all[embed_key], frame["participant_id"].to_numpy()
        )
        sl_t, sv_t, pv_t, pa_t = ssl_eval._fit_linear_probes(
            x_all_c[pre_full], frame.iloc[pre_full], x_all_c[test_idx], max_pca=128
        )
        dev_frozen = {
            "stress": dev_log,
            "stress_va": dev_va,
            "valence": dev_v,
            "arousal": dev_a,
        }
        test_frozen = {
            "stress": sl_t.astype(np.float32),
            "stress_va": sv_t.astype(np.float32),
            "valence": pv_t.astype(np.float32),
            "arousal": pa_t.astype(np.float32),
        }
        seed_frozen_metrics.append(
            _summarize_predictions(frame, dev_frozen, test_frozen)
        )

        for col_k in ("stress", "stress_va", "valence", "arousal"):
            arr = np.full(len(frame), np.nan, dtype=np.float32)
            arr[dev_idx] = dev_frozen[col_k]
            arr[test_idx] = test_frozen[col_k]
            pred_cols[f"{name}_frozen_s{seed}_{col_k}"] = arr

        if run_finetune:
            ft_test, st_ft_te = train_supervised_multimodal(
                data,
                frame,
                pre_full,
                test_idx,
                device,
                pretrained_jepa=jepa_full,
                active_modalities=active_modalities,
                time_slice=time_slice,
                use_fixed_eeg=use_fixed_eeg,
                seed=seed * 10 + 99,
            )
            total_gpu_s += st_ft_te["train_seconds"]
            max_vram = max(max_vram, st_ft_te["peak_vram_gb"])
            seed_ft_metrics.append(_summarize_predictions(frame, ft_dev_out, ft_test))
            for col_k in ("stress", "stress_va", "valence", "arousal"):
                arr = np.full(len(frame), np.nan, dtype=np.float32)
                arr[dev_idx] = ft_dev_out[col_k]
                arr[test_idx] = ft_test[col_k]
                pred_cols[f"{name}_ft_s{seed}_{col_k}"] = arr

        # Representation diagnostics & temporal shuffle test on seed 0
        if run_diagnostics and seed == seeds[0]:
            exported_embeds = dict(rep_all)
            if 0 in active_modalities:
                # Strictly EEG-only, strictly t < 5 s (zero peripheral/future leakage!)
                exported_embeds["z_eeg_early"] = extract_jepa_eeg_only_early(
                    jepa_full,
                    data,
                    all_idx,
                    device,
                    use_fixed_eeg=use_fixed_eeg,
                )
            iso = extract_jepa_window_isolated(
                jepa_full,
                data,
                all_idx,
                device,
                active_modalities=active_modalities,
                use_fixed_eeg=use_fixed_eeg,
            )
            unseen_df = frame[frame["split"].isin(["test_new_both", "test_new_person"])]
            unseen_pos = np.where(
                frame["split"].isin(["test_new_both", "test_new_person"]).to_numpy()
            )[0]
            diagnostics = {
                "subject_identity_raw_vid": ssl_eval.probe_subject_identity(
                    unseen_df, rep_all["z_vid"][unseen_pos]
                ),
                "subject_identity_delta": ssl_eval.probe_subject_identity(
                    unseen_df,
                    rep_all["z_vid"][unseen_pos] - rep_all["z_ref"][unseen_pos],
                ),
                "subject_identity_all_crossday_raw": ssl_eval.probe_subject_identity(
                    frame, rep_all["z_vid"]
                ),
                "subject_identity_all_crossday_delta": ssl_eval.probe_subject_identity(
                    frame, rep_all["z_vid"] - rep_all["z_ref"]
                ),
                "temporal_dynamics_single_pass": ssl_eval.probe_temporal_dynamics(
                    frame, rep_all["z_ref"], rep_all["z_early"], rep_all["z_late"]
                ),
                "temporal_dynamics_iso_3s": ssl_eval.probe_temporal_dynamics(
                    frame, iso["ref3"], iso["early3"], iso["late3"]
                ),
                "temporal_dynamics_iso_5s": ssl_eval.probe_temporal_dynamics(
                    frame, iso["ref3"], iso["early5"], iso["late5"]
                ),
            }
            # Temporal-shuffle degradation check on sealed test!
            rep_shuf = extract_jepa_features(
                jepa_full,
                data,
                test_idx,
                device,
                active_modalities=active_modalities,
                time_slice=time_slice,
                use_fixed_eeg=use_fixed_eeg,
                shuffle_time_seed=20261124,
            )
            x_shuf_c = ssl_eval.person_center_matrix(
                rep_shuf["embed_delta"],
                frame.iloc[test_idx]["participant_id"].to_numpy(),
            )
            sl_sh, sv_sh, pv_sh, pa_sh = ssl_eval._fit_linear_probes(
                x_all_c[pre_full], frame.iloc[pre_full], x_shuf_c, max_pca=128
            )
            test_df = frame.iloc[test_idx]
            shuf_out = {}
            for regime in common.REGIMES:
                rm = (test_df["split"] == regime).to_numpy()
                sub = test_df[rm]
                shuf_out[regime] = {
                    "stress_va_shuffled": ssl_eval.evaluate_stress(
                        sub, pd.Series(sv_sh[rm], index=sub.index)
                    )["stress"],
                    "stress_logreg_shuffled": ssl_eval.evaluate_stress(
                        sub, pd.Series(sl_sh[rm], index=sub.index)
                    )["stress"],
                }
            diagnostics["temporal_shuffle_test"] = shuf_out

            # Also compare raw embed (without subtracting z_ref) on sealed test
            x_raw_c = ssl_eval.person_center_matrix(
                rep_all["embed_raw"], frame["participant_id"].to_numpy()
            )
            sl_r, sv_r, _, _ = ssl_eval._fit_linear_probes(
                x_raw_c[pre_full], frame.iloc[pre_full], x_raw_c[test_idx], max_pca=128
            )
            raw_probe = {}
            for regime in common.REGIMES:
                rm = (test_df["split"] == regime).to_numpy()
                sub = test_df[rm]
                raw_probe[regime] = {
                    "stress_va_no_ref_sub": ssl_eval.evaluate_stress(
                        sub, pd.Series(sv_r[rm], index=sub.index)
                    )["stress"],
                    "stress_logreg_no_ref_sub": ssl_eval.evaluate_stress(
                        sub, pd.Series(sl_r[rm], index=sub.index)
                    )["stress"],
                }
            diagnostics["probe_without_ref_subtraction"] = raw_probe

        del jepa_full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build seed ensemble predictions
    def _ens_block(tag: str) -> dict:
        ed, et = {}, {}
        for col_k in ("stress", "stress_va", "valence", "arousal"):
            st = np.stack(
                [pred_cols[f"{name}_{tag}_s{s}_{col_k}"] for s in seeds], axis=0
            )
            m = np.mean(st, axis=0).astype(np.float32)
            pred_cols[f"{name}_{tag}_ens_{col_k}"] = m
            ed[col_k] = m[dev_idx]
            et[col_k] = m[test_idx]
        return _summarize_predictions(frame, ed, et)

    result: dict = {
        "strategy": strategy,
        "modalities": list(active_modalities),
        "time_slice": list(time_slice),
        "use_fixed_eeg": bool(use_fixed_eeg),
        "seeds": list(seeds),
        "total_gpu_seconds": round(total_gpu_s, 2),
        "peak_vram_gb": round(max_vram, 2),
        "ssl_pretraining": ssl_logs,
        "frozen_seed_stats": {
            "dev_stress_direct": _aggregate_seeds(
                seed_frozen_metrics, ("dev", "stress_direct", "stress", "auc")
            ),
            "dev_stress_va": _aggregate_seeds(
                seed_frozen_metrics, ("dev", "stress_va", "stress", "auc")
            ),
            "dev_mean_rho": _aggregate_seeds(
                seed_frozen_metrics, ("dev", "affect", "mean_rho")
            ),
            "test_new_both_stress_va": _aggregate_seeds(
                seed_frozen_metrics, ("test_new_both", "stress_va", "stress", "auc")
            ),
            "test_new_person_stress_va": _aggregate_seeds(
                seed_frozen_metrics, ("test_new_person", "stress_va", "stress", "auc")
            ),
            "test_new_video_stress_va": _aggregate_seeds(
                seed_frozen_metrics, ("test_new_video", "stress_va", "stress", "auc")
            ),
        },
        "frozen_seed0": seed_frozen_metrics[0],
        "frozen_seed_ensemble": _ens_block("frozen"),
    }
    if run_finetune and seed_ft_metrics:
        result["finetuned_seed_stats"] = {
            "dev_stress_direct": _aggregate_seeds(
                seed_ft_metrics, ("dev", "stress_direct", "stress", "auc")
            ),
            "dev_stress_va": _aggregate_seeds(
                seed_ft_metrics, ("dev", "stress_va", "stress", "auc")
            ),
            "dev_mean_rho": _aggregate_seeds(
                seed_ft_metrics, ("dev", "affect", "mean_rho")
            ),
            "test_new_both_stress_va": _aggregate_seeds(
                seed_ft_metrics, ("test_new_both", "stress_va", "stress", "auc")
            ),
            "test_new_person_stress_va": _aggregate_seeds(
                seed_ft_metrics, ("test_new_person", "stress_va", "stress", "auc")
            ),
            "test_new_video_stress_va": _aggregate_seeds(
                seed_ft_metrics, ("test_new_video", "stress_va", "stress", "auc")
            ),
        }
        result["finetuned_seed0"] = seed_ft_metrics[0]
        result["finetuned_seed_ensemble"] = _ens_block("ft")
    if diagnostics:
        result["diagnostics"] = diagnostics

    fd = result["frozen_seed_stats"]["dev_stress_va"]["mean"]
    ft = result["frozen_seed_stats"]["test_new_both_stress_va"]["mean"]
    print(
        f"  DONE {name}: frozen_dev_va={fd} new_both={ft} "
        f"gpu_s={result['total_gpu_seconds']}",
        flush=True,
    )
    return result, pred_cols, exported_embeds


def main() -> None:
    """Execute Study-08 GPU training, ablation, and diagnostic stages."""
    t_all = time.perf_counter()
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out_dir = Path(os.environ["HABS_ANALYSIS_OUT"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg_path = Path(__file__).resolve().parent.parent / "run_config.json"
    stage = (
        json.loads(cfg_path.read_text()).get("stage", "ssl_jepa_suite")
        if cfg_path.is_file()
        else "ssl_jepa_suite"
    )
    run_primary = stage in ("ssl_jepa_primary", "ssl_jepa_suite")
    run_modwin = stage in (
        "ssl_jepa_abl_modwin",
        "ssl_jepa_ablations",
        "ssl_jepa_suite",
    )
    run_stratscale = stage in (
        "ssl_jepa_abl_stratscale",
        "ssl_jepa_ablations",
        "ssl_jepa_suite",
    )
    out_json_name = f"{stage}.json"
    preds_pq_name = (
        "jepa_preds.parquet"
        if stage in ("ssl_jepa_primary", "ssl_jepa_suite")
        else f"{stage}_preds.parquet"
    )

    trials = pd.read_csv(
        out_dir / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    frame = common.assign_splits(trials, STUDY08_SALT)
    dev_people = set(frame[frame["split"] == "dev"]["participant_id"].astype(str))

    workers = min(20, os.cpu_count() or 8)
    t_load0 = time.perf_counter()
    data_cpu = load_dataset_multimodal(
        trials, raw_dir, eeg_norm="pre_rms", workers=workers
    )
    resting_cpu = (
        load_dev_resting_windows(
            raw_dir, dev_people, eeg_norm="pre_rms", workers=workers
        )
        if run_stratscale
        else None
    )
    load_s = round(time.perf_counter() - t_load0, 2)
    mod_valid_frac = {
        nm: round(float(data_cpu["valid"][:, i, :].mean()), 4)
        for i, nm in enumerate(("eeg", "ecg", "eda", "resp"))
    }
    n_rest_win = int(len(resting_cpu["eeg"])) if resting_cpu is not None else 0
    if device.type == "cuda":
        data = _to_device_store(data_cpu, device)
        resting = (
            _to_device_store(resting_cpu, device) if resting_cpu is not None else None
        )
        del data_cpu, resting_cpu
        gc.collect()
        ds_vram = round(torch.cuda.memory_allocated(device) / (1024**3), 2)
    else:
        data = data_cpu
        resting = resting_cpu
        ds_vram = 0.0
    print(
        f"Stage={stage} | Data loaded in {load_s}s: trials={len(frame)}, "
        f"dev_resting_windows={n_rest_win}, dataset_vram_gb={ds_vram}",
        flush=True,
    )

    report: dict = {
        "study": f"08_{stage}",
        "stage": stage,
        "salt": STUDY08_SALT,
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "data_load_seconds": load_s,
        "dataset_vram_gb": ds_vram,
        "dev_resting_windows": n_rest_win,
        "modality_valid_fraction": mod_valid_frac,
        "splits": common.summary(frame),
        "primary_models": {},
        "ablations": {
            "modality": {},
            "window": {},
            "strategy": {},
            "data_scale": {},
        },
    }
    pub = out_dir / "public"
    pub.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_json() -> None:
        report["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
        (pub / out_json_name).write_text(
            json.dumps(report, indent=1, default=lambda x: None)
        )

    all_preds: dict[str, object] = {"trial_id": frame["trial_id"].to_numpy()}

    # ------------------------------------------------------------------
    # 1. Multi-seed Primary Benchmark (3 seeds: 0, 1, 2)
    # ------------------------------------------------------------------
    if run_primary:
        print("=== BLOCK 1: Multi-seed Primary Models ===", flush=True)
        for m_name, m_kind in (
            ("eegnet", "eegnet"),
            ("eeg_conformer", "eeg_conformer"),
            ("multimodal_conformer_scratch", "multimodal_scratch"),
        ):
            res, pcols = run_supervised_cv_and_test(
                m_name, m_kind, data, frame, device, seeds=SEEDS
            )
            report["primary_models"][m_name] = res
            all_preds.update(pcols)
            _checkpoint_json()

        jepa_res, jepa_pcols, jepa_embeds = run_jepa_variant(
            "physio_jepa",
            data,
            frame,
            device,
            strategy="jepa_crossmodal",
            active_modalities=(0, 1, 2, 3),
            time_slice=(0, 15),
            use_fixed_eeg=False,
            seeds=SEEDS,
            run_finetune=True,
            run_diagnostics=True,
        )
        report["primary_models"]["physio_jepa"] = jepa_res
        all_preds.update(jepa_pcols)

        # Export flagship PhysioJEPA embeddings privately for Job-3 fusion/forecast
        if jepa_embeds is not None:
            emb_cols: dict[str, object] = {"trial_id": frame["trial_id"].to_numpy()}
            for key in ("z_ref", "z_early", "z_late", "z_eeg_early"):
                arr = jepa_embeds[key].astype(np.float32)
                for j in range(arr.shape[1]):
                    emb_cols[f"jepa_{key}_{j}"] = arr[:, j]
            pd.DataFrame(emb_cols).to_parquet(
                feat_dir / "jepa_embeddings.parquet", compression="zstd"
            )
        _checkpoint_json()

    if run_modwin:
        # ------------------------------------------------------------------
        # 2. Modality Ablations (seed 0, fold-isolated)
        # ------------------------------------------------------------------
        print("=== BLOCK 2: Modality Ablations ===", flush=True)
        for mod_name, mod_tuple in (
            ("eeg_only", (0,)),
            ("body_only", (1, 2, 3)),
            ("eeg_ecg", (0, 1)),
            ("eeg_eda", (0, 2)),
            ("eeg_resp", (0, 3)),
        ):
            res, pcols, _ = run_jepa_variant(
                f"mod_{mod_name}",
                data,
                frame,
                device,
                strategy="jepa_masked" if len(mod_tuple) == 1 else "jepa_crossmodal",
                active_modalities=mod_tuple,
                seeds=(0,),
            )
            report["ablations"]["modality"][mod_name] = res
            all_preds.update(pcols)
        _checkpoint_json()

        # ------------------------------------------------------------------
        # 3. Temporal Window Ablations (seed 0, fold-isolated)
        # ------------------------------------------------------------------
        print("=== BLOCK 3: Temporal Window Ablations ===", flush=True)
        for win_name, t_sl in (
            ("early_0_5s", (3, 8)),
            ("video_0_10s", (3, 13)),
            ("pre_video_m3_10s", (0, 13)),
        ):
            res, pcols, _ = run_jepa_variant(
                f"win_{win_name}",
                data,
                frame,
                device,
                strategy="jepa_crossmodal",
                time_slice=t_sl,
                seeds=(0,),
            )
            report["ablations"]["window"][win_name] = res
            all_preds.update(pcols)
        _checkpoint_json()

    if run_stratscale:
        # ------------------------------------------------------------------
        # 4. Pretraining Strategy & Identity-Shortcut Ablations (seed 0)
        # ------------------------------------------------------------------
        print("=== BLOCK 4: Pretraining Strategy Ablations ===", flush=True)
        for strat_name, kwargs in (
            ("jepa_future", {"strategy": "jepa_future"}),
            ("jepa_masked", {"strategy": "jepa_masked"}),
            ("mae_recon", {"strategy": "mae_recon"}),
            (
                "jepa_fixed_raw_norm",
                {
                    "strategy": "jepa_crossmodal",
                    "use_fixed_eeg": True,
                    "run_diagnostics": True,
                },
            ),
            (
                "jepa_plus_rest",
                {"strategy": "jepa_crossmodal", "resting_data": resting},
            ),
        ):
            res, pcols, _ = run_jepa_variant(
                f"strat_{strat_name}", data, frame, device, seeds=(0,), **kwargs
            )
            report["ablations"]["strategy"][strat_name] = res
            all_preds.update(pcols)
        _checkpoint_json()

        # ------------------------------------------------------------------
        # 5. Data-Size Scaling Ablation (10%, 25%, 50% of dev participants)
        # ------------------------------------------------------------------
        print("=== BLOCK 5: Data-Size Scaling Ablation ===", flush=True)
        dev_ppl_sorted = sorted(dev_people)
        rng_sc = np.random.default_rng(20261124)
        perm_ppl = rng_sc.permutation(dev_ppl_sorted).tolist()
        for frac in (0.10, 0.25, 0.50):
            n_keep = max(12, int(round(frac * len(perm_ppl))))
            sub_ppl = set(perm_ppl[:n_keep])
            tag = f"pct_{int(round(frac * 100))}"
            res_j, _, _ = run_jepa_variant(
                f"scale_jepa_{tag}",
                data,
                frame,
                device,
                strategy="jepa_crossmodal",
                seeds=(0,),
                sub_dev_people=sub_ppl,
            )
            res_s, _ = run_supervised_cv_and_test(
                f"scale_scratch_{tag}",
                "multimodal_scratch",
                data,
                frame,
                device,
                seeds=(0,),
                sub_dev_people=sub_ppl,
            )
            report["ablations"]["data_scale"][tag] = {
                "n_dev_people": n_keep,
                "physio_jepa_frozen": res_j,
                "multimodal_scratch": res_s,
            }

    # Save private per-trial predictions table for Job-3 paired bootstrap CIs
    pd.DataFrame(all_preds).to_parquet(feat_dir / preds_pq_name, compression="zstd")
    _checkpoint_json()
    print(
        f"COMPLETED Study 08 stage={stage} in {report['total_elapsed_seconds']}s!",
        flush=True,
    )


if __name__ == "__main__":
    main()
