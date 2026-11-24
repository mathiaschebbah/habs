"""PyTorch unit tests for Study-09 REVE LoRA, ranking loss, and training loops."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from decoding import common, study09_lora, study09_protocol  # noqa: E402


class _MockAttn(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        w = torch.softmax((q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1)
        return self.to_out(w @ v)


class _MockFF(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mock_layers(depth: int = 2, dim: int = 16) -> nn.ModuleList:
    return nn.ModuleList(
        [nn.ModuleList([_MockAttn(dim), _MockFF(dim)]) for _ in range(depth)]
    )


def test_lora_zero_init_and_exact_top_equivalence():
    """At step 0 (B=0), LoRA top layers match frozen layers identically."""
    torch.manual_seed(0)
    dim, ch, patches = 16, 4, 3
    layers = _mock_layers(depth=2, dim=dim)
    cls_q = torch.randn(1, 1, dim)
    h_in = torch.randn(5, ch * patches, dim)

    frozen = study09_lora.ReveTopAttentionPool(
        layers, cls_q, embed_dim=dim, n_chans=ch, n_patches=patches, use_lora=False
    )
    lora = study09_lora.ReveTopAttentionPool(
        layers,
        cls_q,
        embed_dim=dim,
        n_chans=ch,
        n_patches=patches,
        use_lora=True,
        lora_r=4,
    )
    z_f = frozen.encode_pooled(h_in)
    z_l = lora.encode_pooled(h_in)
    assert torch.allclose(z_f, z_l, atol=1e-5)

    # Verify only LoRA A/B (and pool query/norm) require grad inside lora.layers
    base_grads = [
        n
        for n, p in lora.layers.named_parameters()
        if p.requires_grad and "lora_" not in n
    ]
    lora_grads = [
        n for n, p in lora.layers.named_parameters() if p.requires_grad and "lora_" in n
    ]
    assert len(base_grads) == 0
    assert len(lora_grads) == 2 * 2 * 2  # 2 layers x (to_qkv, to_out) x (A, B)


def test_intra_subject_ranking_loss_and_fold_loops():
    """Verify ranking loss gradients and 2-epoch synthetic forecast/multitask folds."""
    scores = torch.tensor([0.1, 0.9, 0.8, 0.2], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pids = torch.tensor([0, 0, 1, 1])
    loss = study09_lora.intra_subject_ranking_loss(scores, targets, pids, min_gap=0.5)
    assert float(loss.item()) > 0.0
    loss.backward()
    # Wrong order for person 0 (score[0] < score[1]): grad pushes score[0] up
    assert float(scores.grad[0].item()) < 0.0
    assert float(scores.grad[1].item()) > 0.0

    # End-to-end synthetic fold smoke test for forecasting & multitask LoRA
    rng = np.random.default_rng(12)
    rows = []
    for p in range(18):
        for t in range(10):
            rows.append(
                {
                    "trial_id": f"P{p:02d}-{t:02d}",
                    "participant_id": f"P{p:02d}",
                    "recorded_day": 1,
                    "event_order": t,
                    "onset_s": float(10 + t * 20),
                    "video": f"theme{t % 6}_{t}.mp4",
                    "video_theme": f"theme{t % 6}",
                    "rating_valence": float(rng.uniform(1, 9)),
                    "rating_arousal": float(rng.uniform(1, 9)),
                }
            )
    frame = common.assign_splits(pd.DataFrame(rows), study09_protocol.STUDY09_BASE_SALT)
    dev_df = frame[frame["split"] == "dev"].reset_index(drop=True)
    n = len(dev_df)
    dim, ch, patches = 16, 4, 5
    layers = _mock_layers(depth=2, dim=dim)
    cls_q = torch.randn(1, 1, dim)
    h16 = torch.randn(n, ch * patches, dim)
    x_body = rng.normal(0, 1, (n, 6)).astype(np.float32)
    y_tgt = rng.normal(0, 1, (n, 3)).astype(np.float32)
    y_tgt[:5, 1] = np.nan

    pf, tf = study09_protocol.outer_blocked_folds(dev_df, k=3)
    tr_idx = np.where((pf != 0) & (tf != 0))[0]
    ev_idx = np.where((pf == 0) & (tf == 0))[0]
    dev_t = torch.device("cpu")

    m0, m2, st_f = study09_lora.train_eval_forecast_fold(
        layers,
        cls_q,
        h16,
        x_body,
        y_tgt,
        dev_df,
        tr_idx,
        ev_idx,
        dev_t,
        use_lora=True,
        lora_r=4,
        seed=0,
        epochs=2,
        n_chans=ch,
        n_patches=patches,
    )
    assert m0.shape == (len(ev_idx), 3)
    assert m2.shape == (len(ev_idx), 3)
    assert np.isfinite(m2).all()
    assert st_f["n_trainable"] > 0

    pr_mt, st_m = study09_lora.train_eval_multitask_fold(
        layers,
        cls_q,
        h16,
        dev_df,
        tr_idx,
        ev_idx,
        dev_t,
        use_lora=True,
        use_ranking=True,
        lora_r=4,
        seed=0,
        epochs=2,
        n_chans=ch,
        n_patches=patches,
    )
    assert pr_mt["stress_va"].shape == (len(ev_idx),)
    assert np.isfinite(pr_mt["stress_va"]).all()
    assert st_m["n_trainable"] > 0
