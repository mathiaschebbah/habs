"""Neural architectures for Study 08: EEGNet, EEG Conformer, and Multimodal PhysioJEPA.

All architectures are deliberately compact (10k to ~1.5M parameters) to match
HABS's sample size (~12k-16k development trials across ~250 participants) and
support strict zero-leakage ablations over modalities, temporal windows,
self-supervised objectives, and training-set scale.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITY_NAMES = ("eeg", "ecg", "eda", "resp")
MODALITY_INDEX = {name: i for i, name in enumerate(MODALITY_NAMES)}


def effective_rank(z: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Roy-Vetterli effective rank exp(H(sigma)) of latent embeddings.

    Parameters
    ----------
    z : torch.Tensor
        Matrix of shape ``(N, D)``.
    eps : float
        Numerical stability floor for singular values.

    Returns
    -------
    float
        Effective rank in ``[1, min(N, D)]``; detects dimensional collapse.
    """
    with torch.no_grad():
        x = z.detach().float()
        x = x - x.mean(dim=0, keepdim=True)
        if x.shape[0] < 2 or x.shape[1] < 2:
            return 1.0
        sv = torch.linalg.svdvals(x)
        p = (sv**2) / (sv.pow(2).sum() + eps)
        p = p.clamp(min=eps)
        entropy = -(p * p.log()).sum()
        return float(torch.exp(entropy).item())


def vicreg_loss(
    z: torch.Tensor,
    *,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance and covariance anti-collapse terms (Bardes et al., 2022).

    Parameters
    ----------
    z : torch.Tensor
        Flattened valid token embeddings of shape ``(N, D)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Weighted ``(var_loss, cov_loss)``.
    """
    if z.shape[0] < 4:
        zero = z.new_zeros(())
        return zero, zero
    x = z.float()
    x = x - x.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()
    cov = (x.T @ x) / (x.shape[0] - 1)
    d = cov.shape[0]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_weight * var_loss, cov_weight * cov_loss


class EEGNet(nn.Module):
    """Compact depthwise-separable CNN for 62-ch EEG (Lawhern et al., 2018).

    Accepts ``(B, C, T)`` at 128 Hz and produces multi-task affective predictions
    ``(stress_logit, valence_pred, arousal_pred)`` plus a trial embedding.
    Uses mathematically identical 1D temporal and grouped spatial-projection
    convolutions so H100 Tensor Cores execute without slow 2D ``(1, 63)``
    cuDNN fallbacks.
    """

    def __init__(
        self,
        n_chans: int = 62,
        f1: int = 8,
        d_mult: int = 2,
        f2: int = 16,
        kernel_length: int = 63,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.f1 = f1
        self.temp_conv = nn.Conv1d(
            1, f1, kernel_length, padding=kernel_length // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.block1_post = nn.Sequential(
            nn.Conv1d(f1 * n_chans, f1 * d_mult, 1, groups=f1, bias=False),
            nn.BatchNorm1d(f1 * d_mult),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(
                f1 * d_mult,
                f1 * d_mult,
                15,
                padding=7,
                groups=f1 * d_mult,
                bias=False,
            ),
            nn.Conv1d(f1 * d_mult, f2, 1, bias=False),
            nn.BatchNorm1d(f2),
            nn.ELU(),
            nn.AvgPool1d(8),
            nn.Dropout(dropout),
        )
        self.embed_dim = f2 * 3
        self.head = nn.Linear(self.embed_dim, 3)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pooled_embed, seq_tokens)`` from ``x`` of shape ``(B, C, T)``."""
        bsz, n_ch, t_len = x.shape
        h = self.temp_conv(x.reshape(bsz * n_ch, 1, t_len))
        h = h.reshape(bsz, n_ch, self.f1, t_len).permute(0, 2, 1, 3).contiguous()
        h = self.bn1(h).reshape(bsz, self.f1 * n_ch, t_len)
        h = self.block1_post(h)
        h = self.block2(h)  # (B, F2, T_out)
        seq = h.transpose(1, 2)  # (B, T_out, F2)
        t_mid = max(1, h.shape[-1] // 2)
        early = h[:, :, :t_mid].mean(dim=-1)
        late = h[:, :, t_mid:].mean(dim=-1)
        std = h.std(dim=-1, unbiased=False)
        pooled = torch.cat([early, late, std], dim=-1)
        return pooled, seq

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict stress logit, valence, and arousal from raw EEG."""
        pooled, seq = self.encode(x)
        out = self.head(pooled)
        return {
            "stress": out[:, 0],
            "valence": out[:, 1],
            "arousal": out[:, 2],
            "embed": pooled,
            "seq": seq,
        }


class EEGConformer(nn.Module):
    """Convolutional Transformer for 62-ch EEG (Song et al., IEEE TNSRE 2023).

    Combines a temporal-spatial convolutional patch stem with a lightweight
    Pre-Norm Transformer encoder over time tokens. Uses mathematically identical
    1D temporal + 1x1 spatial GEMM convolutions for fast Tensor Core execution.
    """

    def __init__(
        self,
        n_chans: int = 62,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_filters = 40
        self.temp_conv = nn.Conv1d(1, self.n_filters, 25, padding=12, bias=False)
        self.stem_post = nn.Sequential(
            nn.Conv1d(self.n_filters * n_chans, self.n_filters, 1, bias=False),
            nn.BatchNorm1d(self.n_filters),
            nn.ELU(),
            nn.AvgPool1d(32, stride=16),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(self.n_filters, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.embed_dim = d_model * 3
        self.head = nn.Linear(self.embed_dim, 3)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pooled_embed, seq_tokens)`` from ``x`` of shape ``(B, C, T)``."""
        bsz, n_ch, t_len = x.shape
        h = self.temp_conv(x.reshape(bsz * n_ch, 1, t_len))
        h = (
            h.reshape(bsz, n_ch, self.n_filters, t_len)
            .permute(0, 2, 1, 3)
            .reshape(bsz, self.n_filters * n_ch, t_len)
        )
        h = self.stem_post(h).transpose(1, 2)  # (B, L, 40)
        h = self.proj(h)
        seq_len = h.shape[1]
        if seq_len <= self.pos_emb.shape[1]:
            h = h + self.pos_emb[:, :seq_len, :]
        else:
            pos = F.interpolate(
                self.pos_emb.transpose(1, 2), size=seq_len, mode="linear"
            ).transpose(1, 2)
            h = h + pos
        seq = self.norm(self.encoder(h))
        mid = max(1, seq_len // 2)
        early = seq[:, :mid, :].mean(dim=1)
        late = seq[:, mid:, :].mean(dim=1)
        diff = late - early
        pooled = torch.cat([early, late, diff], dim=-1)
        return pooled, seq

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict stress logit, valence, and arousal from raw EEG."""
        pooled, seq = self.encode(x)
        out = self.head(pooled)
        return {
            "stress": out[:, 0],
            "valence": out[:, 1],
            "arousal": out[:, 2],
            "embed": pooled,
            "seq": seq,
        }


class EEGPatchTokenizer(nn.Module):
    """Dual time-frequency 1-s patch tokenizer for 62-channel 128 Hz EEG."""

    def __init__(self, n_chans: int = 62, d_model: int = 128, n_fft_bins: int = 46):
        super().__init__()
        self.n_chans = n_chans
        self.n_fft_bins = n_fft_bins
        half = d_model // 2
        # Spatial compression + multi-scale temporal convolution
        self.spatial = nn.Conv1d(n_chans, 32, kernel_size=1, bias=False)
        self.t_conv1 = nn.Conv1d(32, 32, kernel_size=15, stride=4, padding=7)
        self.t_conv2 = nn.Conv1d(32, half, kernel_size=7, stride=4, padding=3)
        self.t_norm = nn.GroupNorm(4, half)
        # Differentiable spectral log-amplitude branch (1-45 Hz)
        self.spec_spatial = nn.Linear(n_chans, 16, bias=False)
        self.spec_mlp = nn.Sequential(
            nn.Linear(16 * n_fft_bins, half),
            nn.LayerNorm(half),
            nn.GELU(),
            nn.Linear(half, half),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, T_sec, 128)`` into ``(B, T_sec, d_model)``."""
        bsz, n_ch, t_sec, n_samp = eeg.shape
        x = eeg.permute(0, 2, 1, 3).reshape(bsz * t_sec, n_ch, n_samp)
        # Temporal branch
        ht = F.gelu(self.spatial(x))
        ht = F.gelu(self.t_conv1(ht))
        ht = F.gelu(self.t_norm(self.t_conv2(ht)))
        ht = ht.mean(dim=-1)  # (B*T, half)
        # Spectral branch (log1p rfft magnitude on 0..45 Hz)
        spec = torch.fft.rfft(x.float(), dim=-1).abs()[:, :, : self.n_fft_bins]
        spec = torch.log1p(spec).to(x.dtype)  # (B*T, C, F)
        hs = self.spec_spatial(spec.transpose(1, 2)).reshape(bsz * t_sec, -1)
        hs = self.spec_mlp(hs)
        tok = self.out_norm(torch.cat([ht, hs], dim=-1))
        return tok.reshape(bsz, t_sec, -1)


class PeripheralPatchTokenizer(nn.Module):
    """Lightweight 1-s patch tokenizer for ECG (128 Hz), EDA or Resp (16 Hz)."""

    def __init__(self, in_chans: int, d_model: int = 128, kernel: int = 5):
        super().__init__()
        mid = max(32, d_model // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_chans, mid, kernel_size=kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv1d(mid, mid, kernel_size=kernel, padding=kernel // 2),
            nn.GELU(),
        )
        # Pool mean, std, and max + linear slope proxy
        self.proj = nn.Sequential(
            nn.Linear(mid * 3, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C_in, T_sec, N_samp)`` into ``(B, T_sec, d_model)``."""
        bsz, n_ch, t_sec, n_samp = sig.shape
        x = sig.permute(0, 2, 1, 3).reshape(bsz * t_sec, n_ch, n_samp)
        h = self.conv(x)
        pooled = torch.cat(
            [
                h.mean(dim=-1),
                h.std(dim=-1, unbiased=False),
                h.amax(dim=-1),
            ],
            dim=-1,
        )
        return self.proj(pooled).reshape(bsz, t_sec, -1)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block supporting arbitrary boolean key-padding masks."""

    def __init__(
        self, d_model: int, n_heads: int, mlp_ratio: float = 3.0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply attention + MLP; ``key_padding_mask`` True marks ignored tokens."""
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class PhysioJEPAConfig:
    """Configuration for the Multimodal Physiological JEPA World Model."""

    n_eeg_chans: int = 62
    max_seconds: int = 15
    d_model: int = 128
    n_heads: int = 4
    enc_layers: int = 4
    pred_layers: int = 2
    dropout: float = 0.1
    ema_start: float = 0.996
    ema_end: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04


class MultimodalEncoder(nn.Module):
    """Shared multimodal tokenizer + Transformer backbone (context or EMA target)."""

    def __init__(self, cfg: PhysioJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.eeg_tok = EEGPatchTokenizer(cfg.n_eeg_chans, cfg.d_model)
        self.ecg_tok = PeripheralPatchTokenizer(2, cfg.d_model, kernel=9)
        self.eda_tok = PeripheralPatchTokenizer(2, cfg.d_model, kernel=5)
        self.resp_tok = PeripheralPatchTokenizer(2, cfg.d_model, kernel=5)
        self.mod_emb = nn.Parameter(torch.randn(4, cfg.d_model) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(cfg.max_seconds, cfg.d_model) * 0.02)
        self.missing_tok = nn.Parameter(torch.randn(4, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
                for _ in range(cfg.enc_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def tokenize(
        self,
        batch: dict[str, torch.Tensor],
        active_modalities: tuple[int, ...] = (0, 1, 2, 3),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw multimodal batch to ``(tokens, valid, mod_ids, time_ids)``."""
        t_sec = batch["eeg"].shape[2]
        device = batch["eeg"].device

        raw_toks = [
            self.eeg_tok(batch["eeg"]),
            self.ecg_tok(batch["ecg"]),
            self.eda_tok(batch["eda"]),
            self.resp_tok(batch["resp"]),
        ]
        valid_full = batch["valid"]  # (B, 4, T) bool
        tok_list, val_list, mod_list, time_list = [], [], [], []
        t_grid = torch.arange(t_sec, device=device)

        for m in active_modalities:
            tm = raw_toks[m]  # (B, T, D)
            vm = valid_full[:, m, :t_sec]  # (B, T)
            miss = self.missing_tok[m].view(1, 1, -1).expand_as(tm)
            tm = torch.where(vm.unsqueeze(-1), tm, miss)
            tm = (
                tm + self.mod_emb[m].view(1, 1, -1) + self.time_emb[:t_sec].unsqueeze(0)
            )
            tok_list.append(tm)
            val_list.append(vm)
            mod_list.append(torch.full((t_sec,), m, device=device, dtype=torch.long))
            time_list.append(t_grid)

        tokens = torch.cat(tok_list, dim=1)  # (B, M*T, D)
        valid = torch.cat(val_list, dim=1)  # (B, M*T)
        mod_ids = torch.cat(mod_list, dim=0)  # (M*T,)
        time_ids = torch.cat(time_list, dim=0)  # (M*T,)
        return tokens, valid, mod_ids, time_ids

    def encode_tokens(
        self, tokens: torch.Tensor, visible_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode tokens attending only to ``visible_mask`` (True = visible)."""
        # Safeguard against all-False rows to avoid NaN softmax in MultiheadAttention
        any_vis = visible_mask.any(dim=1, keepdim=True)
        safe_vis = visible_mask | (~any_vis)
        key_padding_mask = ~safe_vis
        h = tokens
        for blk in self.blocks:
            h = blk(h, key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return h * visible_mask.unsqueeze(-1).to(h.dtype)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        active_modalities: tuple[int, ...] = (0, 1, 2, 3),
        visible_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize and encode; returns ``(z, valid, mod_ids, time_ids)``."""
        tokens, valid, mod_ids, time_ids = self.tokenize(batch, active_modalities)
        vis = valid if visible_mask is None else (valid & visible_mask)
        z = self.encode_tokens(tokens, vis)
        return z, valid, mod_ids, time_ids


def sample_jepa_mask(
    valid: torch.Tensor,
    mod_ids: torch.Tensor,
    time_ids: torch.Tensor,
    strategy: str = "jepa_crossmodal",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``(context_mask, target_mask)`` boolean tensors of shape ``(B, L)``.

    Strategies
    ----------
    - ``jepa_future``: causal World Model predicting future horizon ``t >= t0``
      from past context ``t < t0`` across all active modalities.
    - ``jepa_masked``: contiguous temporal block masking (~45% masked).
    - ``jepa_crossmodal``: joint temporal + cross-modal masking (randomly masks
      temporal blocks/future AND drops either EEG or peripheral streams on
      target steps, forcing cross-modal brain-body latent prediction).
    - ``mae_recon``: same mask geometry as ``jepa_masked`` for raw reconstruction.
    """
    bsz, seq_len = valid.shape
    device = valid.device
    t_max = int(time_ids.max().item()) + 1
    target_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)

    for i in range(bsz):
        if strategy == "jepa_future":
            lo = max(3, t_max // 3)
            hi = max(lo + 1, t_max - 2)
            t0 = int(torch.randint(lo, hi, (1,), generator=generator).item())
            t_sel = time_ids >= t0
            target_mask[i] = t_sel
        elif strategy in ("jepa_masked", "mae_recon"):
            n_mask = max(2, int(round(t_max * 0.45)))
            hi_start = max(1, t_max - n_mask + 1)
            start = int(torch.randint(0, hi_start, (1,), generator=generator).item())
            t_sel = (time_ids >= start) & (time_ids < start + n_mask)
            target_mask[i] = t_sel
        elif strategy == "jepa_crossmodal":
            mode = int(torch.randint(0, 3, (1,), generator=generator).item())
            if mode == 0:
                # Causal future rollout across all modalities
                lo = max(3, t_max // 3)
                hi = max(lo + 1, t_max - 2)
                t0 = int(torch.randint(lo, hi, (1,), generator=generator).item())
                target_mask[i] = time_ids >= t0
            elif mode == 1:
                # Temporal block mask across all modalities
                n_mask = max(2, int(round(t_max * 0.40)))
                start = int(
                    torch.randint(
                        0, max(1, t_max - n_mask + 1), (1,), generator=generator
                    ).item()
                )
                target_mask[i] = (time_ids >= start) & (time_ids < start + n_mask)
            else:
                # Cross-modal prediction across whichever modalities are active
                t0 = 3 if t_max >= 8 else max(1, t_max // 3)
                uniq_mods = torch.unique(mod_ids)
                if len(uniq_mods) >= 2:
                    if (0 in uniq_mods) and len(uniq_mods) == 4:
                        if (
                            int(torch.randint(0, 2, (1,), generator=generator).item())
                            == 0
                        ):
                            target_mask[i] = (mod_ids != 0) & (time_ids >= t0)
                        else:
                            target_mask[i] = (mod_ids == 0) & (time_ids >= t0)
                    else:
                        pick = int(
                            torch.randint(
                                0, len(uniq_mods), (1,), generator=generator
                            ).item()
                        )
                        target_mask[i] = (mod_ids == uniq_mods[pick]) & (time_ids >= t0)
                else:
                    target_mask[i] = time_ids >= t0
        else:
            raise ValueError(f"Unknown SSL strategy: {strategy}")

    target_mask = target_mask & valid
    context_mask = valid & (~target_mask)
    # Guarantee at least 2 context tokens if a sample has sparse validity
    bad_ctx = context_mask.sum(dim=1) < 2
    if bad_ctx.any():
        for idx in torch.where(bad_ctx)[0].tolist():
            val_pos = torch.where(valid[idx])[0]
            if len(val_pos) >= 2:
                half = len(val_pos) // 2
                context_mask[idx] = False
                target_mask[idx] = False
                context_mask[idx, val_pos[:half]] = True
                target_mask[idx, val_pos[half:]] = True
    return context_mask, target_mask


class PhysioJEPA(nn.Module):
    """Multimodal Physiological Joint-Embedding Predictive Architecture.

    Supports:
    - Latent JEPA pretraining (``jepa_future``, ``jepa_masked``, ``jepa_crossmodal``)
      with EMA target encoder + VICReg anti-collapse regularization.
    - Raw-signal MAE reconstruction control (``mae_recon``) on the exact same
      backbone.
    - End-to-end supervised training from scratch (``MultimodalConformer``
      control) using ``supervised_forward``.
    - Frozen representation extraction returning both token-level ``(B, M, T, D)``
      and phase-structured trial embeddings.
    """

    def __init__(self, cfg: PhysioJEPAConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PhysioJEPAConfig()
        self.context_encoder = MultimodalEncoder(self.cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Latent predictor (conditioned on target modality & temporal position)
        d = self.cfg.d_model
        self.mask_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pred_blocks = nn.ModuleList(
            [
                TransformerBlock(d, self.cfg.n_heads, dropout=self.cfg.dropout)
                for _ in range(self.cfg.pred_layers)
            ]
        )
        self.pred_norm = nn.LayerNorm(d)
        self.pred_head = nn.Linear(d, d)

        # Lightweight signal-space reconstruction decoders for the MAE control
        self.mae_eeg_head = nn.Linear(d, 62 * 32)  # 32-pt downsampled 1-s patch
        self.mae_ecg_head = nn.Linear(d, 2 * 32)
        self.mae_eda_head = nn.Linear(d, 2 * 16)
        self.mae_resp_head = nn.Linear(d, 2 * 16)

        # Supervised multi-task head (stress logit, valence, arousal)
        self.sup_head = nn.Linear(d * 3, 3)

    @torch.no_grad()
    def update_ema(self, progress: float) -> float:
        """Cosine-scheduled exponential moving average update of target encoder."""
        tau = self.cfg.ema_end - (self.cfg.ema_end - self.cfg.ema_start) * 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
        ctx_params = list(self.context_encoder.parameters())
        tgt_params = list(self.target_encoder.parameters())
        for i, p_ctx in enumerate(ctx_params):
            tgt_params[i].data.mul_(tau).add_(p_ctx.data, alpha=1.0 - tau)
        return tau

    def ssl_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        strategy: str = "jepa_crossmodal",
        active_modalities: tuple[int, ...] = (0, 1, 2, 3),
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute one self-supervised training step (JEPA or MAE control)."""
        tokens, valid, mod_ids, time_ids = self.context_encoder.tokenize(
            batch, active_modalities
        )
        ctx_mask, tgt_mask = sample_jepa_mask(
            valid, mod_ids, time_ids, strategy=strategy, generator=generator
        )
        z_ctx = self.context_encoder.encode_tokens(tokens, ctx_mask)

        # Assemble predictor input: context latents + positional mask queries
        pos_q = (
            self.mask_query
            + self.context_encoder.mod_emb[mod_ids].unsqueeze(0)
            + self.context_encoder.time_emb[time_ids].unsqueeze(0)
        )
        pred_in = torch.where(ctx_mask.unsqueeze(-1), z_ctx, pos_q.expand_as(z_ctx))
        active_pred = ctx_mask | tgt_mask
        any_act = active_pred.any(dim=1, keepdim=True)
        kpm = ~(active_pred | (~any_act))
        h = pred_in
        for blk in self.pred_blocks:
            h = blk(h, key_padding_mask=kpm)
        pred_latents = self.pred_head(self.pred_norm(h))

        if strategy == "mae_recon":
            loss_pred = self._mae_reconstruction_loss(
                pred_latents, batch, tgt_mask, mod_ids, time_ids
            )
            var_l, cov_l = vicreg_loss(
                z_ctx[ctx_mask],
                var_weight=0.1 * self.cfg.var_weight,
                cov_weight=0.1 * self.cfg.cov_weight,
            )
            return {
                "loss": loss_pred + var_l + cov_l,
                "pred_loss": loss_pred.detach(),
                "var_loss": var_l.detach(),
                "cov_loss": cov_l.detach(),
                "z_ctx_flat": z_ctx[ctx_mask].detach(),
            }

        with torch.no_grad():
            z_tgt, _, _, _ = self.target_encoder(
                batch, active_modalities=active_modalities, visible_mask=valid
            )
            # Layer-normalize target representations per token (I-JEPA / V-JEPA)
            z_tgt = F.layer_norm(z_tgt.float(), (z_tgt.shape[-1],))

        if tgt_mask.any():
            loss_pred = F.smooth_l1_loss(
                pred_latents[tgt_mask].float(), z_tgt[tgt_mask].float()
            )
        else:
            loss_pred = pred_latents.new_zeros(())

        var_l, cov_l = vicreg_loss(
            z_ctx[ctx_mask],
            var_weight=self.cfg.var_weight,
            cov_weight=self.cfg.cov_weight,
        )
        return {
            "loss": loss_pred + var_l + cov_l,
            "pred_loss": loss_pred.detach(),
            "var_loss": var_l.detach(),
            "cov_loss": cov_l.detach(),
            "z_ctx_flat": z_ctx[ctx_mask].detach(),
        }

    def _mae_reconstruction_loss(
        self,
        pred_latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        tgt_mask: torch.Tensor,
        mod_ids: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        """MSE reconstruction of masked raw signal patches for the MAE control."""
        losses = []
        t_sec = batch["eeg"].shape[2]
        bsz = batch["eeg"].shape[0]
        # Downsample 128 Hz -> 32 Hz targets so EEG and peripheral MSE scales match
        targets = {
            0: F.avg_pool1d(
                batch["eeg"].permute(0, 2, 1, 3).reshape(bsz * t_sec, 62, -1), 4
            ).reshape(bsz, t_sec, -1),
            1: F.avg_pool1d(
                batch["ecg"].permute(0, 2, 1, 3).reshape(bsz * t_sec, 2, -1), 4
            ).reshape(bsz, t_sec, -1),
            2: batch["eda"].permute(0, 2, 1, 3).reshape(bsz, t_sec, -1),
            3: batch["resp"].permute(0, 2, 1, 3).reshape(bsz, t_sec, -1),
        }
        heads = {
            0: self.mae_eeg_head,
            1: self.mae_ecg_head,
            2: self.mae_eda_head,
            3: self.mae_resp_head,
        }
        for m in range(4):
            sel = (mod_ids == m).unsqueeze(0) & tgt_mask
            if not sel.any():
                continue
            b_idx, l_idx = torch.where(sel)
            t_idx = time_ids[l_idx]
            p_m = heads[m](pred_latents[b_idx, l_idx]).float()
            y_m = targets[m][b_idx, t_idx].float()
            losses.append(F.mse_loss(p_m, y_m))
        if not losses:
            return pred_latents.new_zeros(())
        return torch.stack(losses).mean()

    def encode_structured(
        self,
        batch: dict[str, torch.Tensor],
        *,
        active_modalities: tuple[int, ...] = (0, 1, 2, 3),
        use_target_encoder: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Extract token-grid and phase-structured trial representations.

        Returns both raw video embeddings (``embed_vid``) and baseline-referenced
        phase-delta embeddings (``embed_delta``) so downstream probes can directly
        contrast static identity vs temporal physiological dynamics.
        """
        enc = self.target_encoder if use_target_encoder else self.context_encoder
        z, valid, mod_ids, time_ids = enc(batch, active_modalities=active_modalities)
        bsz = z.shape[0]
        n_mod = len(active_modalities)
        t_sec = batch["eeg"].shape[2]
        d = self.cfg.d_model

        z_grid = z.reshape(bsz, n_mod, t_sec, d)
        v_grid = valid.reshape(bsz, n_mod, t_sec).float()

        def _masked_mean(z_sub: torch.Tensor, v_sub: torch.Tensor) -> torch.Tensor:
            denom = v_sub.sum(dim=(1, 2), keepdim=False).clamp(min=1.0).unsqueeze(-1)
            return (z_sub * v_sub.unsqueeze(-1)).sum(dim=(1, 2)) / denom

        if t_sec >= 13:
            # Canonical 15-s layout: ref [0:3], early [3:8], late [8:13], post [13:]
            z_ref = _masked_mean(z_grid[:, :, :3, :], v_grid[:, :, :3])
            z_early = _masked_mean(z_grid[:, :, 3:8, :], v_grid[:, :, 3:8])
            z_late = _masked_mean(z_grid[:, :, 8:13, :], v_grid[:, :, 8:13])
            z_vid = _masked_mean(z_grid[:, :, 3:13, :], v_grid[:, :, 3:13])
        elif t_sec >= 8:
            mid = t_sec // 2
            z_ref = _masked_mean(z_grid[:, :, :1, :], v_grid[:, :, :1])
            z_early = _masked_mean(z_grid[:, :, :mid, :], v_grid[:, :, :mid])
            z_late = _masked_mean(z_grid[:, :, mid:, :], v_grid[:, :, mid:])
            z_vid = _masked_mean(z_grid, v_grid)
        else:
            mid = max(1, t_sec // 2)
            z_ref = _masked_mean(z_grid[:, :, :1, :], v_grid[:, :, :1])
            z_early = _masked_mean(z_grid[:, :, :mid, :], v_grid[:, :, :mid])
            z_late = _masked_mean(z_grid[:, :, mid:, :], v_grid[:, :, mid:])
            z_vid = _masked_mean(z_grid, v_grid)

        # Phase-delta trial representation (3 * d)
        embed_delta = torch.cat(
            [z_early - z_ref, z_late - z_ref, z_late - z_early], dim=-1
        )
        embed_raw = torch.cat([z_early, z_late, z_late - z_early], dim=-1)
        return {
            "z_grid": z_grid,  # (B, M, T, D)
            "v_grid": v_grid,  # (B, M, T)
            "z_ref": z_ref,
            "z_early": z_early,
            "z_late": z_late,
            "z_vid": z_vid,
            "embed_delta": embed_delta,
            "embed_raw": embed_raw,
        }

    def supervised_forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        active_modalities: tuple[int, ...] = (0, 1, 2, 3),
        use_delta: bool = True,
    ) -> dict[str, torch.Tensor]:
        """End-to-end supervised forward pass (matched Multimodal Conformer control)."""
        rep = self.encode_structured(
            batch, active_modalities=active_modalities, use_target_encoder=False
        )
        feat = rep["embed_delta"] if use_delta else rep["embed_raw"]
        out = self.sup_head(feat)
        return {
            "stress": out[:, 0],
            "valence": out[:, 1],
            "arousal": out[:, 2],
            "embed": feat,
            "z_grid": rep["z_grid"],
        }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count scalar parameters in a PyTorch module."""
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if (not trainable_only) or p.requires_grad
        )
    )
