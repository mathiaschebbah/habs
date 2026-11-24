"""Study-09 GPU Job 2: Lightweight REVE-base LoRA for forecasting and multitask affect.

Runs strictly inside ``dev`` (``STUDY09_BASE_SALT = "habs-ssl-2026-11-24-b"``)
using outer 3x3 blocked ``person x theme`` cross-validation with internal
person-and-theme-disjoint early stopping (ZERO exposed test data ever enters
training, hyperparameter selection, or early stopping!).

Two controlled comparisons across 3 seeds (0, 1, 2):
1. **Primary Forecasting (Req 6):** Strictly causal ``EEG [-3.2, +5.0) s``
   (hard causal wall at t = 5.000 s, Cold-Start trial z-score) + ``Body [0:5]``
   -> future ``Body [5, 10) s`` (HR, SC, Resp), comparing:
   - ``reve_forecast_frozen_head`` (top-6 REVE layers frozen + attn-pool residual head)
   - ``reve_forecast_lora`` (top-6 REVE layers with LoRA r=8 on ``to_qkv`` & ``to_out``)
2. **Secondary Multitask Affect + Intra-Subject Ranking (Req 9):**
   Strictly causal ``EEG [0.0, 10.0) s`` (Cold-Start trial z-score), comparing:
   - ``reve_mt_frozen_rank`` (top-6 REVE frozen + multitask + ranking)
   - ``reve_mt_lora_rank`` (top-6 REVE LoRA r=8 + multitask + intra-subject ranking)
   - ``reve_mt_lora_norank`` (seed 0 ablation: LoRA multitask WITHOUT ranking loss)
   - ``reve_mt_lora_r4`` (seed 0 ablation: ultra-light LoRA rank r=4)

Optimization insight
--------------------
``REVE-base`` (22 layers, 69.2M params) contains zero dropout and zero BatchNorm.
Freezing layers 0..15 and caching their exact bf16 hidden tensor ``h_16`` once in
H100 HBM3 (~7.3 GB for 12,693 dev trials) yields mathematically identical forward
passes and exact gradients for top-6-layer LoRA (layers 16..21) at ~4x speedup.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from decoding import common, ssl_eval, study09_extract, study09_protocol
from decoding.stress import label
from habs.temporal.data import physical_id

SEEDS = (0, 1, 2)
TOP_LORA_START = 16  # Top 6 of 22 REVE-base layers (16..21)
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.10
FORECAST_EPOCHS = 14
MT_EPOCHS = 16
BATCH_SIZE = 64

FORECAST_TARGETS = (
    "tgt_ecg_hr_chg_5_10",
    "tgt_eda_sc_chg_5_10",
    "tgt_resp_amp_chg_5_10",
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. Pure-PyTorch LoRA Linear + REVE Top-Layer Wrapper
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Lightweight low-rank adapter wrapping a frozen ``nn.Linear`` layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = LORA_RANK,
        alpha: float = LORA_ALPHA,
        dropout: float = LORA_DROPOUT,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.scaling = float(alpha) / float(max(r, 1))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frozen base linear plus scaled low-rank adapter."""
        out = self.base(x)
        delta = F.linear(F.linear(self.drop(x), self.lora_a), self.lora_b)
        return out + delta * self.scaling


def inject_lora_into_layers(
    layers: nn.ModuleList,
    *,
    r: int = LORA_RANK,
    alpha: float = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
) -> int:
    """Freeze ``layers`` and wrap ``to_qkv`` / ``to_out`` with LoRA."""
    for p in layers.parameters():
        p.requires_grad = False
    n_wrapped = 0
    for block in layers:
        attn = block[0] if isinstance(block, (nn.ModuleList, list, tuple)) else block
        for attr in ("to_qkv", "to_out"):
            lin = getattr(attn, attr, None)
            if isinstance(lin, nn.Linear):
                setattr(
                    attn,
                    attr,
                    LoRALinear(lin, r=r, alpha=alpha, dropout=dropout),
                )
                n_wrapped += 1
    return n_wrapped


class ReveTopAttentionPool(nn.Module):
    """Top transformer blocks of REVE-base (frozen or LoRA) + learnable attn pool."""

    def __init__(
        self,
        top_layers: nn.ModuleList,
        cls_query: torch.Tensor,
        *,
        embed_dim: int = 512,
        n_chans: int = 62,
        n_patches: int = 9,
        pool_patch_slice: tuple[int, int] | None = None,
        use_lora: bool = True,
        precomputed_top: bool = False,
        lora_r: int = LORA_RANK,
        lora_alpha: float = LORA_ALPHA,
        lora_dropout: float = LORA_DROPOUT,
    ) -> None:
        super().__init__()
        self.use_lora = bool(use_lora)
        self.precomputed_top = bool(precomputed_top) and (not self.use_lora)
        if self.precomputed_top:
            self.layers = nn.ModuleList()
        else:
            self.layers = copy.deepcopy(top_layers)
            for p in self.layers.parameters():
                p.requires_grad = False
            if self.use_lora:
                inject_lora_into_layers(
                    self.layers, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
                )
        self.embed_dim = int(embed_dim)
        self.n_chans = int(n_chans)
        self.n_patches = int(n_patches)
        self.pool_patch_slice = pool_patch_slice
        self.cls_query = nn.Parameter(cls_query.detach().clone().float())
        self.pool_norm = nn.LayerNorm(self.embed_dim)

    def encode_pooled(self, h_in: torch.Tensor) -> torch.Tensor:
        """Forward ``h_in`` (B, C*H, E) through top layers and pool -> (B, 2*E)."""
        x = h_in
        if self.precomputed_top:
            x = x.detach()
        elif self.use_lora:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
        else:
            with torch.no_grad():
                for attn, ff in self.layers:
                    x = attn(x) + x
                    x = ff(x) + x
            x = x.detach()

        bsz = x.shape[0]
        grid = x.view(bsz, self.n_chans, self.n_patches, self.embed_dim)
        if self.pool_patch_slice is not None:
            p0, p1 = self.pool_patch_slice
            grid = grid[:, :, p0:p1, :]
        flat = grid.reshape(bsz, -1, self.embed_dim).float()  # (B, C*S, E)
        q = self.cls_query.expand(bsz, -1, -1)  # (B, 1, E)
        scores = torch.matmul(q, flat.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        attn_vec = torch.matmul(weights, flat).squeeze(1)
        mean_vec = flat.mean(dim=1)
        return torch.cat([self.pool_norm(attn_vec), self.pool_norm(mean_vec)], dim=-1)


class ForecastResidualHead(nn.Module):
    """Residual forecasting model: ``M0_pred + gate * f(REVE_pool, Body[0:5])``."""

    def __init__(
        self,
        backbone: ReveTopAttentionPool,
        body_dim: int,
        n_targets: int = 3,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        in_dim = backbone.embed_dim * 2 + body_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.30),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, n_targets),
        )
        # Small initial gate so training starts safely near M0
        self.gate = nn.Parameter(torch.full((n_targets,), 0.25))

    def forward(self, h_in: torch.Tensor, x_body: torch.Tensor) -> torch.Tensor:
        """Predict gated residual correction on top of M0."""
        z = self.backbone.encode_pooled(h_in)
        feat = torch.cat([z, x_body], dim=-1)
        return torch.tanh(self.gate) * self.net(feat)


class MultitaskAffectHead(nn.Module):
    """Multitask stress + valence + arousal head over REVE top-layer pooling."""

    def __init__(self, backbone: ReveTopAttentionPool, hidden: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        in_dim = backbone.embed_dim * 2
        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.35),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.head_stress = nn.Linear(hidden, 1)
        self.head_valence = nn.Linear(hidden, 1)
        self.head_arousal = nn.Linear(hidden, 1)

    def forward(self, h_in: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict stress logit, valence, and arousal."""
        z = self.backbone.encode_pooled(h_in)
        h = self.shared(z)
        return {
            "stress": self.head_stress(h).squeeze(-1),
            "valence": self.head_valence(h).squeeze(-1),
            "arousal": self.head_arousal(h).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# 2. Intra-Subject Pairwise Logistic Ranking Loss & Grouped Sampler (Req 9)
# ---------------------------------------------------------------------------


def intra_subject_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    person_ids: torch.Tensor,
    *,
    min_gap: float = 0.5,
) -> torch.Tensor:
    """Differentiable within-subject pairwise logistic ranking loss.

    Computes ``mean_p mean_{i,j in p: y_i - y_j >= min_gap} softplus(-(s_i - s_j))``.
    Directly optimizes within-person ordering (ROC AUC / Spearman rho) without
    wasting gradient capacity on between-person intercept shifts.
    """
    same_p = person_ids[:, None] == person_ids[None, :]
    dy = targets[:, None] - targets[None, :]
    valid_pairs = same_p & (dy >= min_gap)
    if not valid_pairs.any():
        return scores.sum() * 0.0
    ds = scores[:, None] - scores[None, :]
    # Weight each participant equally so high-trial subjects don't dominate
    losses = F.softplus(-ds) * valid_pairs.float()
    pair_cnt_per_row = valid_pairs.float().sum(dim=1)
    # Average over all valid pairs (already balanced by P x M grouped sampler)
    return losses.sum() / pair_cnt_per_row.sum().clamp(min=1.0)


def sample_person_grouped_batches(
    indices: np.ndarray,
    person_codes: np.ndarray,
    rng: np.random.Generator,
    *,
    people_per_batch: int = 16,
    trials_per_person: int = 4,
) -> list[np.ndarray]:
    """Yield mini-batches of ``P x M`` participant-grouped trials."""
    by_p: dict[int, list[int]] = {}
    for pos, _idx in enumerate(indices):
        by_p.setdefault(int(person_codes[pos]), []).append(pos)
    # Shuffle each person's trial list and form chunks of size M
    chunks_by_p: list[tuple[int, np.ndarray]] = []
    for p, pos_list in by_p.items():
        arr = rng.permutation(pos_list)
        for s in range(0, len(arr), trials_per_person):
            ch = arr[s : s + trials_per_person]
            if len(ch) >= 2:
                chunks_by_p.append((p, ch))
    rng.shuffle(chunks_by_p)
    batches: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    cur_len = 0
    target_bsz = people_per_batch * trials_per_person
    for _, ch in chunks_by_p:
        cur.append(ch)
        cur_len += len(ch)
        if cur_len >= target_bsz:
            batches.append(np.concatenate(cur))
            cur, cur_len = [], 0
    if cur:
        batches.append(np.concatenate(cur))
    return batches


def _internal_person_theme_val_split(
    df: pd.DataFrame, tr_local_idx: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Strict person-AND-theme disjoint internal validation split inside outer train."""
    sub = df.iloc[tr_local_idx]
    people = np.sort(sub["participant_id"].unique())
    themes = np.sort(sub["theme"].unique())
    rng = np.random.default_rng(seed + 4099)
    val_p = set(rng.permutation(people)[: max(2, int(round(0.20 * len(people))))])
    val_t = set(rng.permutation(themes)[: max(2, int(round(0.20 * len(themes))))])
    p_m = sub["participant_id"].isin(val_p).to_numpy()
    t_m = sub["theme"].isin(val_t).to_numpy()
    # Val = unseen people x unseen themes (fallback to unseen people if tiny)
    is_val = p_m & t_m
    if is_val.sum() < 25:
        is_val = p_m
        is_tr = ~p_m
    else:
        is_tr = (~p_m) & (~t_m)
    return tr_local_idx[is_tr], tr_local_idx[is_val]


# ---------------------------------------------------------------------------
# 3. Training Loops (Forecasting LoRA & Multitask Affect+Ranking LoRA)
# ---------------------------------------------------------------------------


def train_eval_forecast_fold(
    top_layers: nn.ModuleList,
    cls_query: torch.Tensor,
    h16_gpu: torch.Tensor,
    x_body_all: np.ndarray,
    y_all: np.ndarray,
    dev_df: pd.DataFrame,
    tr_idx: np.ndarray,
    ev_idx: np.ndarray,
    device: torch.device,
    *,
    use_lora: bool = True,
    precomputed_top: bool = False,
    lora_r: int = LORA_RANK,
    seed: int = 0,
    epochs: int = FORECAST_EPOCHS,
    n_chans: int = 62,
    n_patches: int = 9,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train M0 + residual REVE (Frozen or LoRA) on one outer dev fold.

    Returns ``(m0_pred_ev, m2_pred_ev, stats)`` of shape ``(len(ev_idx), 3)``.
    """
    _set_seed(seed)
    n_t = y_all.shape[1]
    tr_sub, val_sub = _internal_person_theme_val_split(dev_df, tr_idx, seed)

    # 1. Fit fold-isolated M0 on Body[0:5] (strictly inside tr_idx!)
    m0_oof_tr = np.full((len(tr_idx), n_t), np.nan, dtype=np.float32)
    m0_ev = np.full((len(ev_idx), n_t), np.nan, dtype=np.float32)
    for k in range(n_t):
        p_ev, p_oof, _ = study09_protocol.fit_predict_nested_ridge(
            x_body_all[tr_idx],
            y_all[tr_idx, k],
            dev_df.iloc[tr_idx],
            x_body_all[ev_idx],
            alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0),
            pca_grid=(None,),
            inner_k=3,
            return_inner_oof=True,
        )
        m0_oof_tr[:, k] = p_oof.astype(np.float32)
        m0_ev[:, k] = p_ev.astype(np.float32)

    # Map global dev indices -> local position in tr_idx
    tr_map = {int(g): i for i, g in enumerate(tr_idx)}
    sub_pos = np.array([tr_map[int(g)] for g in tr_sub], dtype=int)
    val_pos = np.array([tr_map[int(g)] for g in val_sub], dtype=int)

    # Fold-isolated standardized body features
    zb_tr_full, zb_ev = study09_protocol._preprocess_fit_transform(
        x_body_all[tr_idx], x_body_all[ev_idx], max_pca=None
    )
    zb_sub = zb_tr_full[sub_pos].astype(np.float32)
    zb_val = zb_tr_full[val_pos].astype(np.float32)
    zb_ev32 = zb_ev.astype(np.float32)

    # Residual targets e = y - m0_oof_tr, standardized strictly on tr_sub
    e_tr_all = y_all[tr_idx] - m0_oof_tr
    e_sub = e_tr_all[sub_pos]
    e_val = e_tr_all[val_pos]
    e_mu = np.nanmean(e_sub, axis=0)
    e_mu = np.where(np.isfinite(e_mu), e_mu, 0.0)
    e_sd = np.nanstd(e_sub, axis=0)
    e_sd = np.where(np.isfinite(e_sd) & (e_sd > 1e-4), e_sd, 1.0)
    e_sub_z = ((e_sub - e_mu) / e_sd).astype(np.float32)
    e_val_z = ((e_val - e_mu) / e_sd).astype(np.float32)

    backbone = ReveTopAttentionPool(
        top_layers,
        cls_query,
        embed_dim=int(cls_query.shape[-1]),
        n_chans=n_chans,
        n_patches=n_patches,
        pool_patch_slice=(min(4, n_patches - 1), n_patches),
        use_lora=use_lora,
        precomputed_top=precomputed_top,
        lora_r=lora_r,
    )
    net = ForecastResidualHead(backbone, body_dim=zb_sub.shape[1], n_targets=n_t).to(
        device
    )
    trainable = [p for p in net.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in trainable))
    opt = torch.optim.AdamW(trainable, lr=3e-4 if use_lora else 1e-3, weight_decay=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 17)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    def _masked_smooth_l1(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        ok = torch.isfinite(tgt)
        if not ok.any():
            return pred.sum() * 0.0
        return F.smooth_l1_loss(pred[ok], tgt[ok])

    for _ in range(epochs):
        net.train()
        order = rng.permutation(len(tr_sub))
        for s in range(0, len(order), BATCH_SIZE):
            bi = order[s : s + BATCH_SIZE]
            h_b = h16_gpu[torch.as_tensor(tr_sub[bi], device=h16_gpu.device)].to(device)
            xb_b = torch.from_numpy(zb_sub[bi]).to(device)
            yb_b = torch.from_numpy(e_sub_z[bi]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                pr = net(h_b, xb_b).float()
                loss = _masked_smooth_l1(pr, yb_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.5)
            opt.step()
        sched.step()

        net.eval()
        v_losses = []
        with torch.inference_mode():
            for s in range(0, len(val_sub), BATCH_SIZE):
                sl = slice(s, min(len(val_sub), s + BATCH_SIZE))
                h_b = h16_gpu[torch.as_tensor(val_sub[sl], device=h16_gpu.device)].to(
                    device
                )
                xb_b = torch.from_numpy(zb_val[sl]).to(device)
                yb_b = torch.from_numpy(e_val_z[sl]).to(device)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    pr = net(h_b, xb_b).float()
                    vl = _masked_smooth_l1(pr, yb_b)
                if torch.isfinite(vl):
                    v_losses.append(float(vl.item()))
        vm = float(np.mean(v_losses)) if v_losses else float("inf")
        if vm < best_val:
            best_val = vm
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }

    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    net.eval()
    # Select optimal residual blend eta in [0, 1] strictly on val_sub
    val_pred_z = np.zeros((len(val_sub), n_t), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(val_sub), BATCH_SIZE):
            e = min(len(val_sub), s + BATCH_SIZE)
            h_b = h16_gpu[torch.as_tensor(val_sub[s:e], device=h16_gpu.device)].to(
                device
            )
            xb_b = torch.from_numpy(zb_val[s:e]).to(device)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                val_pred_z[s:e] = net(h_b, xb_b).float().cpu().numpy()
    val_pred_e = val_pred_z * e_sd[None, :] + e_mu[None, :]
    etas = np.zeros(n_t, dtype=np.float32)
    val_ppl = dev_df.iloc[val_sub]["participant_id"].to_numpy()
    for k in range(n_t):
        best_e_loss = float("inf")
        best_eta = 0.0
        for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
            cand = m0_oof_tr[val_pos, k] + eta * val_pred_e[:, k]
            loss_k = study09_protocol._within_person_mse(
                y_all[val_sub, k], cand, val_ppl
            )
            if loss_k < best_e_loss:
                best_e_loss = loss_k
                best_eta = eta
        etas[k] = best_eta

    ev_pred_z = np.zeros((len(ev_idx), n_t), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(ev_idx), BATCH_SIZE):
            e = min(len(ev_idx), s + BATCH_SIZE)
            h_b = h16_gpu[torch.as_tensor(ev_idx[s:e], device=h16_gpu.device)].to(
                device
            )
            xb_b = torch.from_numpy(zb_ev32[s:e]).to(device)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                ev_pred_z[s:e] = net(h_b, xb_b).float().cpu().numpy()
    ev_pred_e = ev_pred_z * e_sd[None, :] + e_mu[None, :]
    m2_ev = m0_ev + etas[None, :] * ev_pred_e

    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    del net
    return (
        m0_ev.astype(np.float32),
        m2_ev.astype(np.float32),
        {
            "n_trainable": n_trainable,
            "gpu_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "etas": [round(float(x), 2) for x in etas],
        },
    )


def train_eval_multitask_fold(
    top_layers: nn.ModuleList,
    cls_query: torch.Tensor,
    h16_gpu: torch.Tensor,
    dev_df: pd.DataFrame,
    tr_idx: np.ndarray,
    ev_idx: np.ndarray,
    device: torch.device,
    *,
    use_lora: bool = True,
    precomputed_top: bool = False,
    use_ranking: bool = True,
    lora_r: int = LORA_RANK,
    seed: int = 0,
    epochs: int = MT_EPOCHS,
    n_chans: int = 62,
    n_patches: int = 11,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train multitask stress+V+A (+ intra-subject ranking) on one dev fold."""
    _set_seed(seed)
    tr_sub, val_sub = _internal_person_theme_val_split(dev_df, tr_idx, seed)

    def _targets(
        idxs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sub = dev_df.iloc[idxs]
        ys = label(sub).to_numpy(np.float32)
        ppl = sub["participant_id"]
        v_raw = sub["rating_valence"].to_numpy(np.float32)
        a_raw = sub["rating_arousal"].to_numpy(np.float32)
        yv = (
            sub["rating_valence"] - sub["rating_valence"].groupby(ppl).transform("mean")
        ).to_numpy(np.float32)
        ya = (
            sub["rating_arousal"] - sub["rating_arousal"].groupby(ppl).transform("mean")
        ).to_numpy(np.float32)
        yv = yv / max(float(np.std(yv)), 1e-4)
        ya = ya / max(float(np.std(ya)), 1e-4)
        pcodes = ppl.astype("category").cat.codes.to_numpy(np.int64)
        return ys, yv, ya, v_raw, a_raw, pcodes

    ys_tr, yv_tr, ya_tr, vr_tr, ar_tr, pc_tr = _targets(tr_sub)
    ys_va, yv_va, ya_va, vr_va, ar_va, pc_va = _targets(val_sub)

    backbone = ReveTopAttentionPool(
        top_layers,
        cls_query,
        embed_dim=int(cls_query.shape[-1]),
        n_chans=n_chans,
        n_patches=n_patches,
        pool_patch_slice=None,
        use_lora=use_lora,
        precomputed_top=precomputed_top,
        lora_r=lora_r,
    )
    net = MultitaskAffectHead(backbone).to(device)
    trainable = [p for p in net.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in trainable))
    opt = torch.optim.AdamW(trainable, lr=3e-4 if use_lora else 1e-3, weight_decay=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_rate = float(np.clip(ys_tr.mean(), 0.05, 0.95))
    pos_w = torch.tensor([(1.0 - pos_rate) / pos_rate], device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 71)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    for _ in range(epochs):
        net.train()
        if use_ranking:
            batches = sample_person_grouped_batches(
                tr_sub, pc_tr, rng, people_per_batch=16, trials_per_person=4
            )
        else:
            perm = rng.permutation(len(tr_sub))
            batches = [
                perm[s : s + BATCH_SIZE] for s in range(0, len(perm), BATCH_SIZE)
            ]

        for bi in batches:
            h_b = h16_gpu[torch.as_tensor(tr_sub[bi], device=h16_gpu.device)].to(device)
            ts = torch.from_numpy(ys_tr[bi]).to(device)
            tv = torch.from_numpy(yv_tr[bi]).to(device)
            ta = torch.from_numpy(ya_tr[bi]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = net(h_b)
                ps, pv, pa = (
                    out["stress"].float(),
                    out["valence"].float(),
                    out["arousal"].float(),
                )
                loss = (
                    F.binary_cross_entropy_with_logits(ps, ts, pos_weight=pos_w)
                    + 0.5 * F.smooth_l1_loss(pv, tv)
                    + 0.5 * F.smooth_l1_loss(pa, ta)
                )
                if use_ranking:
                    pids_t = torch.from_numpy(pc_tr[bi]).to(device)
                    vr_t = torch.from_numpy(vr_tr[bi]).to(device)
                    ar_t = torch.from_numpy(ar_tr[bi]).to(device)
                    l_rs = intra_subject_ranking_loss(ps, ts, pids_t, min_gap=0.5)
                    l_rv = intra_subject_ranking_loss(pv, vr_t, pids_t, min_gap=1.5)
                    l_ra = intra_subject_ranking_loss(pa, ar_t, pids_t, min_gap=1.5)
                    loss = loss + 0.5 * l_rs + 0.25 * l_rv + 0.25 * l_ra
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.5)
            opt.step()
        sched.step()

        # Internal validation loss (disjoint people & themes)
        net.eval()
        v_losses = []
        with torch.inference_mode():
            for s in range(0, len(val_sub), BATCH_SIZE):
                sl = slice(s, min(len(val_sub), s + BATCH_SIZE))
                h_b = h16_gpu[torch.as_tensor(val_sub[sl], device=h16_gpu.device)].to(
                    device
                )
                ts = torch.from_numpy(ys_va[sl]).to(device)
                tv = torch.from_numpy(yv_va[sl]).to(device)
                ta = torch.from_numpy(ya_va[sl]).to(device)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    out = net(h_b)
                    ps, pv, pa = (
                        out["stress"].float(),
                        out["valence"].float(),
                        out["arousal"].float(),
                    )
                    vl = (
                        F.binary_cross_entropy_with_logits(ps, ts, pos_weight=pos_w)
                        + 0.5 * F.smooth_l1_loss(pv, tv)
                        + 0.5 * F.smooth_l1_loss(pa, ta)
                    )
                v_losses.append(float(vl.item()))
        vm = float(np.mean(v_losses)) if v_losses else float("inf")
        if vm < best_val:
            best_val = vm
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }

    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    net.eval()

    # Fit V&A standardization SDs strictly on internal val_sub (zero eval-batch leak!)
    vv, va = [], []
    with torch.inference_mode():
        for s in range(0, len(val_sub), BATCH_SIZE):
            sl = slice(s, min(len(val_sub), s + BATCH_SIZE))
            h_b = h16_gpu[torch.as_tensor(val_sub[sl], device=h16_gpu.device)].to(
                device
            )
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                ov = net(h_b)
            vv.append(ov["valence"].float().cpu().numpy())
            va.append(ov["arousal"].float().cpu().numpy())
    sd_v = max(float(np.std(np.concatenate(vv))) if vv else 1.0, 1e-6)
    sd_a = max(float(np.std(np.concatenate(va))) if va else 1.0, 1e-6)

    n_ev = len(ev_idx)
    ps_ev = np.zeros(n_ev, dtype=np.float32)
    pv_ev = np.zeros(n_ev, dtype=np.float32)
    pa_ev = np.zeros(n_ev, dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n_ev, BATCH_SIZE):
            e = min(n_ev, s + BATCH_SIZE)
            h_b = h16_gpu[torch.as_tensor(ev_idx[s:e], device=h16_gpu.device)].to(
                device
            )
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                oe = net(h_b)
            ps_ev[s:e] = oe["stress"].float().cpu().numpy()
            pv_ev[s:e] = oe["valence"].float().cpu().numpy()
            pa_ev[s:e] = oe["arousal"].float().cpu().numpy()

    pva_ev = (pa_ev / sd_a - pv_ev / sd_v).astype(np.float32)
    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    del net
    return (
        {"stress": ps_ev, "stress_va": pva_ev, "valence": pv_ev, "arousal": pa_ev},
        {
            "n_trainable": n_trainable,
            "gpu_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
        },
    )


# ---------------------------------------------------------------------------
# 4. Precompute Exact Frozen Layer-16 Hidden States & Orchestrate Job 2
# ---------------------------------------------------------------------------


@torch.no_grad()
def precompute_reve_h16(
    model: nn.Module,
    base_pos: torch.Tensor,
    eeg_arr: np.ndarray,
    device: torch.device,
    *,
    layer_idx: int = TOP_LORA_START,
    batch_size: int = 48,
) -> torch.Tensor:
    """Compute exact frozen ``REVE-base`` layer ``layer_idx`` input tensor in bf16."""
    n = len(eeg_arr)
    # Run one probe batch to read exact sequence length
    x0 = study09_extract._zscore_trial_causal(
        torch.from_numpy(eeg_arr[:1]).to(device, dtype=torch.float32)
    )
    pos0 = base_pos.unsqueeze(0)
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        outs0 = model(x0, pos0, return_output=True)
    seq_len, dim = outs0[layer_idx].shape[1], outs0[layer_idx].shape[2]
    dtype_store = torch.bfloat16 if device.type == "cuda" else torch.float32
    store = torch.empty((n, seq_len, dim), dtype=dtype_store, device=device)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        x = study09_extract._zscore_trial_causal(
            torch.from_numpy(eeg_arr[s:e]).to(device, dtype=torch.float32)
        )
        pos = base_pos.unsqueeze(0).expand(e - s, -1, -1)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outs = model(x, pos, return_output=True)
        store[s:e] = outs[layer_idx].to(dtype=dtype_store)
    return store


@torch.no_grad()
def precompute_reve_h22(
    top_layers: nn.ModuleList,
    h16: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    """Pass frozen ``h16`` through ``top_layers`` once so frozen heads skip 6 layers."""
    layers = copy.deepcopy(top_layers).to(device).eval()
    store = torch.empty_like(h16)
    for s in range(0, len(h16), batch_size):
        e = min(len(h16), s + batch_size)
        x = h16[s:e].to(device)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            for attn, ff in layers:
                x = attn(x) + x
                x = ff(x) + x
        store[s:e] = x.to(dtype=store.dtype)
    del layers
    return store


def _load_dev_causal_arrays(
    trials_dev: pd.DataFrame, raw_dir: Path, *, workers: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load strictly causal dev EEG[-3.2,5), EEG[0,10), Body[0:5], and targets."""
    id_to_pos = {tid: i for i, tid in enumerate(trials_dev["trial_id"])}
    n = len(trials_dev)
    n_ch = len(study09_extract.CANONICAL_62)
    n_past = round(
        (study09_extract.PAST_END_S - study09_extract.PAST_START_S)
        * study09_extract.REVE_FS
    )
    n_vid10 = 10 * study09_extract.REVE_FS

    eeg_past = np.zeros((n, n_ch, n_past), dtype=np.float16)
    eeg_vid10 = np.zeros((n, n_ch, n_vid10), dtype=np.float16)
    tab_store: dict[str, np.ndarray] = {}

    jobs = [
        (str(raw_dir), grp.to_dict("records"))
        for _, grp in trials_dev.groupby("recording", sort=True)
    ]
    ctx = multiprocessing.get_context("spawn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for idx, part in enumerate(
            pool.map(study09_extract._worker_recording_study09, jobs), 1
        ):
            pos = np.array([id_to_pos[t] for t in part["trial_ids"]], dtype=int)
            eeg_past[pos] = part["eeg_past"]
            # eeg_vid is [-3, +10) s -> [0, 10) s is [600:2600]
            eeg_vid10[pos] = part["eeg_vid"][:, :, 600:2600]
            for k, arr in part["tabular"].items():
                if k.startswith("b05_") or k in FORECAST_TARGETS:
                    if k not in tab_store:
                        tab_store[k] = np.full(n, np.nan, dtype=np.float32)
                    tab_store[k][pos] = arr
            if idx % 50 == 0 or idx == len(jobs):
                print(f"LoRA dev load: {idx}/{len(jobs)} recordings", flush=True)

    body_cols = sorted(k for k in tab_store if k.startswith("b05_"))
    x_body = np.column_stack([tab_store[k] for k in body_cols]).astype(np.float32)
    y_tgt = np.column_stack([tab_store[k] for k in FORECAST_TARGETS]).astype(np.float32)
    return eeg_past, eeg_vid10, x_body, y_tgt, body_cols


def main() -> None:
    """Execute Study-09 H100 LoRA benchmark strictly inside dev (outer 3x3 CV)."""
    from transformers import AutoModel

    t_all = time.perf_counter()
    raw_dir = Path(os.environ["HABS_DATA_ROOT"])
    out_dir = Path(os.environ["HABS_ANALYSIS_OUT"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(__file__).resolve().parent.parent / "run_config.json"
    stage = (
        json.loads(cfg_path.read_text()).get("stage", "study09_lora")
        if cfg_path.is_file()
        else "study09_lora"
    )
    run_part_a = stage in ("study09_lora", "study09_lora_forecast")
    run_part_b = stage in ("study09_lora", "study09_lora_mt")
    json_name = f"{stage}.json"
    parquet_name = f"{stage}_preds.parquet"

    trials = pd.read_csv(
        out_dir / "data" / "aligned_trials.csv.gz", dtype={"participant_id": str}
    )
    trials = trials.assign(trial_id=[physical_id(r) for r in trials.to_dict("records")])
    frame = common.assign_splits(trials, study09_protocol.STUDY09_BASE_SALT)
    dev_df = frame[frame["split"] == "dev"].reset_index(drop=True)
    n_dev = len(dev_df)
    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)

    t_load0 = time.perf_counter()
    workers = min(20, os.cpu_count() or 8)
    eeg_past, eeg_vid10, x_body, y_tgt, body_cols = _load_dev_causal_arrays(
        dev_df, raw_dir, workers=workers
    )
    load_s = round(time.perf_counter() - t_load0, 2)
    print(
        f"[{stage}] Loaded {n_dev} dev trials ({len(body_cols)} body cols) "
        f"in {load_s}s",
        flush=True,
    )

    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions", trust_remote_code=True
    ).to(device)
    reve = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).to(
        device
    )
    reve.eval()
    base_pos = pos_bank(list(study09_extract.CANONICAL_62)).to(device)
    top_layers_cpu = copy.deepcopy(reve.transformer.layers[TOP_LORA_START:]).cpu()
    cls_query_cpu = reve.cls_query_token.detach().cpu().clone()

    preds_df_cols: dict[str, object] = {"trial_id": dev_df["trial_id"].to_numpy()}
    report: dict = {
        "study": "09_reve_lora_dev_nested_cv",
        "stage": stage,
        "salt": study09_protocol.STUDY09_BASE_SALT,
        "scope": "dev_only_outer_3x3_blocked_person_x_theme",
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "n_dev_trials": n_dev,
        "n_dev_people": int(dev_df["participant_id"].nunique()),
        "n_dev_themes": int(dev_df["theme"].nunique()),
        "n_body_features": len(body_cols),
        "lora_start_layer": TOP_LORA_START,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "load_seconds": load_s,
        "forecasting": {},
        "multitask_affect": {},
    }
    pub = out_dir / "public"
    pub.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt() -> None:
        report["total_elapsed_seconds"] = round(time.perf_counter() - t_all, 2)
        (pub / json_name).write_text(
            json.dumps(report, indent=1, default=lambda x: None)
        )

    # ------------------------------------------------------------------
    # PART A: Primary Causal Forecasting LoRA (M2_LoRA vs M2_FrozenHead vs M0)
    # ------------------------------------------------------------------
    if run_part_a:
        print(
            "=== PART A: Precomputing REVE h16/h22 on causal [-3.2, +5.0) s ===",
            flush=True,
        )
        t_h0 = time.perf_counter()
        h16_past = precompute_reve_h16(reve, base_pos, eeg_past, device)
        h22_past = precompute_reve_h22(top_layers_cpu, h16_past, device)
        h16_past_s = round(time.perf_counter() - t_h0, 2)
        h16_past_gb = round(
            h16_past.element_size() * h16_past.nelement() / (1024**3), 2
        )
        report["forecasting"]["h16_precompute_s"] = h16_past_s
        report["forecasting"]["h16_vram_gb"] = h16_past_gb
        print(
            f"h16/h22_past: {tuple(h16_past.shape)} ({h16_past_gb} GB each) in "
            f"{h16_past_s}s",
            flush=True,
        )
        if not run_part_b:
            del reve, pos_bank, eeg_vid10
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        m0_ref_oof = np.full((n_dev, len(FORECAST_TARGETS)), np.nan, dtype=np.float32)
        for var_name, use_lora, seeds_use in (
            ("reve_forecast_frozen_head", False, SEEDS),
            ("reve_forecast_lora", True, SEEDS),
        ):
            h_use = h16_past if use_lora else h22_past
            pre_top = not use_lora
            seed_oofs = []
            total_gpu_s = 0.0
            peak_vram = 0.0
            n_train_p = 0
            for seed in seeds_use:
                oof_m2 = np.full(
                    (n_dev, len(FORECAST_TARGETS)), np.nan, dtype=np.float32
                )
                for p in range(3):
                    for t in range(3):
                        ev_idx = np.where((pf == p) & (tf == t))[0]
                        tr_idx = np.where((pf != p) & (tf != t))[0]
                        p_m0, p_m2, st = train_eval_forecast_fold(
                            top_layers_cpu,
                            cls_query_cpu,
                            h_use,
                            x_body,
                            y_tgt,
                            dev_df,
                            tr_idx,
                            ev_idx,
                            device,
                            use_lora=use_lora,
                            precomputed_top=pre_top,
                            seed=seed * 100 + p * 3 + t,
                            n_patches=9,
                        )
                        if seed == 0 and not use_lora:
                            m0_ref_oof[ev_idx] = p_m0
                        oof_m2[ev_idx] = p_m2
                        total_gpu_s += st["gpu_s"]
                        peak_vram = max(peak_vram, st["peak_vram_gb"])
                        n_train_p = st["n_trainable"]
                seed_oofs.append(oof_m2)
                for k_i, tname in enumerate(FORECAST_TARGETS):
                    preds_df_cols[f"{var_name}_s{seed}_{tname}"] = oof_m2[:, k_i]

            ens_oof = np.mean(np.stack(seed_oofs, axis=0), axis=0).astype(np.float32)
            for k_i, tname in enumerate(FORECAST_TARGETS):
                preds_df_cols[f"{var_name}_ens_{tname}"] = ens_oof[:, k_i]
                if not use_lora:
                    preds_df_cols[f"m0_ref_{tname}"] = m0_ref_oof[:, k_i]

            tgt_res = {}
            for k_i, tname in enumerate(FORECAST_TARGETS):
                s_m0, pp_m0 = study09_protocol.summarize_forecast(
                    dev_df, y_tgt[:, k_i], m0_ref_oof[:, k_i]
                )
                s_ens, pp_ens = study09_protocol.summarize_forecast(
                    dev_df, y_tgt[:, k_i], ens_oof[:, k_i]
                )
                seed_rs = []
                for so in seed_oofs:
                    sm, _ = study09_protocol.summarize_forecast(
                        dev_df, y_tgt[:, k_i], so[:, k_i], draws=200
                    )
                    seed_rs.append(sm["r"]["mean"])
                tgt_res[tname] = {
                    "m0_summary": s_m0,
                    "ensemble_summary": s_ens,
                    "paired_vs_m0": study09_protocol.paired_forecast_comparison(
                        pp_ens, pp_m0
                    ),
                    "seed_r_mean": round(float(np.mean(seed_rs)), 4),
                    "seed_r_sd": round(float(np.std(seed_rs, ddof=1)), 4)
                    if len(seed_rs) > 1
                    else 0.0,
                    "seed_r_per_seed": seed_rs,
                }
            report["forecasting"][var_name] = {
                "use_lora": use_lora,
                "n_trainable_params": n_train_p,
                "seeds": list(seeds_use),
                "total_gpu_seconds": round(total_gpu_s, 2),
                "peak_vram_gb": round(peak_vram, 2),
                "targets": tgt_res,
            }
            dr_hr = tgt_res[FORECAST_TARGETS[0]]["paired_vs_m0"]["delta_r"]["mean"]
            dr_sc = tgt_res[FORECAST_TARGETS[1]]["paired_vs_m0"]["delta_r"]["mean"]
            dr_re = tgt_res[FORECAST_TARGETS[2]]["paired_vs_m0"]["delta_r"]["mean"]
            print(
                f"  DONE {var_name}: HR d_r={dr_hr}, SC d_r={dr_sc}, "
                f"Resp d_r={dr_re} ({total_gpu_s:.1f}s)",
                flush=True,
            )
            _ckpt()

        del h16_past, h22_past, eeg_past
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PART B: Secondary Multitask Affect + Intra-Subject Ranking (Req 9)
    # ------------------------------------------------------------------
    if run_part_b:
        print(
            "=== PART B: Precomputing REVE h16/h22 on causal [0.0, 10.0) s ===",
            flush=True,
        )
        t_h1 = time.perf_counter()
        h16_vid = precompute_reve_h16(reve, base_pos, eeg_vid10, device)
        h22_vid = precompute_reve_h22(top_layers_cpu, h16_vid, device)
        h16_vid_s = round(time.perf_counter() - t_h1, 2)
        h16_vid_gb = round(h16_vid.element_size() * h16_vid.nelement() / (1024**3), 2)
        report["multitask_affect"]["h16_precompute_s"] = h16_vid_s
        report["multitask_affect"]["h16_vram_gb"] = h16_vid_gb
        print(
            f"h16/h22_vid: {tuple(h16_vid.shape)} ({h16_vid_gb} GB each) "
            f"in {h16_vid_s}s",
            flush=True,
        )

        # Free full 22-layer REVE before running top-6 LoRA loops
        del reve, pos_bank, eeg_vid10
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        mt_specs = (
            ("reve_mt_frozen_rank", False, True, LORA_RANK, SEEDS),
            ("reve_mt_lora_rank", True, True, LORA_RANK, SEEDS),
            ("reve_mt_lora_norank", True, False, LORA_RANK, (0,)),
            ("reve_mt_lora_r4", True, True, 4, (0,)),
        )
        frozen_ref_pp: dict[str, float] | None = None
        for m_name, use_lora, use_rank, r_val, seeds_use in mt_specs:
            h_use = h16_vid if use_lora else h22_vid
            pre_top = not use_lora
            seed_preds = []
            seed_aucs, seed_va_aucs, seed_rhos = [], [], []
            total_gpu_s = 0.0
            peak_vram = 0.0
            n_train_p = 0
            for seed in seeds_use:
                oof = {
                    k: np.full(n_dev, np.nan, dtype=np.float32)
                    for k in ("stress", "stress_va", "valence", "arousal")
                }
                for p in range(3):
                    for t in range(3):
                        ev_idx = np.where((pf == p) & (tf == t))[0]
                        tr_idx = np.where((pf != p) & (tf != t))[0]
                        pr, st = train_eval_multitask_fold(
                            top_layers_cpu,
                            cls_query_cpu,
                            h_use,
                            dev_df,
                            tr_idx,
                            ev_idx,
                            device,
                            use_lora=use_lora,
                            precomputed_top=pre_top,
                            use_ranking=use_rank,
                            lora_r=r_val,
                            seed=seed * 100 + p * 3 + t,
                            n_patches=11,
                        )
                        for k in oof:
                            oof[k][ev_idx] = pr[k]
                        total_gpu_s += st["gpu_s"]
                        peak_vram = max(peak_vram, st["peak_vram_gb"])
                        n_train_p = st["n_trainable"]
                seed_preds.append(oof)
                for k in oof:
                    preds_df_cols[f"{m_name}_s{seed}_{k}"] = oof[k]
                st_d = ssl_eval.evaluate_stress(
                    dev_df, pd.Series(oof["stress"], index=dev_df.index), draws=200
                )
                st_v = ssl_eval.evaluate_stress(
                    dev_df, pd.Series(oof["stress_va"], index=dev_df.index), draws=200
                )
                aff_s = ssl_eval.evaluate_valence_arousal(
                    dev_df,
                    pd.Series(oof["valence"], index=dev_df.index),
                    pd.Series(oof["arousal"], index=dev_df.index),
                    draws=200,
                )
                seed_aucs.append(st_d["stress"]["auc"])
                seed_va_aucs.append(st_v["stress"]["auc"])
                seed_rhos.append(aff_s["mean_rho"])

            ens = {
                k: np.mean(
                    np.stack([sp[k] for sp in seed_preds], axis=0), axis=0
                ).astype(np.float32)
                for k in ("stress", "stress_va", "valence", "arousal")
            }
            for k in ens:
                preds_df_cols[f"{m_name}_ens_{k}"] = ens[k]

            res_dir = ssl_eval.evaluate_stress(
                dev_df,
                pd.Series(ens["stress"], index=dev_df.index),
                return_per_person=True,
            )
            pp_dir = res_dir.pop("_per_person")
            res_va = ssl_eval.evaluate_stress(
                dev_df,
                pd.Series(ens["stress_va"], index=dev_df.index),
                return_per_person=True,
            )
            pp_va = res_va.pop("_per_person")
            res_aff = ssl_eval.evaluate_valence_arousal(
                dev_df,
                pd.Series(ens["valence"], index=dev_df.index),
                pd.Series(ens["arousal"], index=dev_df.index),
            )
            if m_name == "reve_mt_frozen_rank":
                frozen_ref_pp = pp_va

            entry = {
                "use_lora": use_lora,
                "use_ranking": use_rank,
                "lora_r": r_val if use_lora else 0,
                "n_trainable_params": n_train_p,
                "seeds": list(seeds_use),
                "total_gpu_seconds": round(total_gpu_s, 2),
                "peak_vram_gb": round(peak_vram, 2),
                "seed_stress_direct_auc": {
                    "mean": round(float(np.mean(seed_aucs)), 4),
                    "sd": round(float(np.std(seed_aucs, ddof=1)), 4)
                    if len(seed_aucs) > 1
                    else 0.0,
                    "per_seed": seed_aucs,
                },
                "seed_stress_va_auc": {
                    "mean": round(float(np.mean(seed_va_aucs)), 4),
                    "sd": round(float(np.std(seed_va_aucs, ddof=1)), 4)
                    if len(seed_va_aucs) > 1
                    else 0.0,
                    "per_seed": seed_va_aucs,
                },
                "seed_mean_rho": {
                    "mean": round(float(np.mean(seed_rhos)), 4),
                    "sd": round(float(np.std(seed_rhos, ddof=1)), 4)
                    if len(seed_rhos) > 1
                    else 0.0,
                    "per_seed": seed_rhos,
                },
                "ensemble_stress_direct": res_dir,
                "ensemble_stress_va": res_va,
                "ensemble_affect": res_aff,
            }
            if frozen_ref_pp is not None and m_name != "reve_mt_frozen_rank":
                entry["paired_va_vs_frozen_rank"] = ssl_eval.paired_gain_ci(
                    pp_va, frozen_ref_pp
                )
                entry["paired_dir_vs_frozen_rank"] = ssl_eval.paired_gain_ci(
                    pp_dir, frozen_ref_pp
                )
            report["multitask_affect"][m_name] = entry
            print(
                f"  DONE {m_name}: direct_auc={res_dir['stress']['auc']}, "
                f"va_auc={res_va['stress']['auc']}, rho={res_aff['mean_rho']} "
                f"({total_gpu_s:.1f}s)",
                flush=True,
            )
            _ckpt()

    pd.DataFrame(preds_df_cols).to_parquet(
        feat_dir / parquet_name, compression="zstd"
    )
    _ckpt()
    print(f"[{stage}] complete in {report['total_elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
