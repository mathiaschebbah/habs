"""PyTorch smoke tests for Study-08 models and training/ablation helpers."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from decoding import common, ssl_eval, ssl_models, ssl_train  # noqa: E402


def test_ssl_models_and_train_pipeline(monkeypatch):
    """Run 1-epoch synthetic smoke test of EEGNet, Conformer, and PhysioJEPA."""
    monkeypatch.setattr(ssl_eval, "DRAWS", 20)
    monkeypatch.setattr(ssl_train, "SSL_EPOCHS", 2)
    monkeypatch.setattr(ssl_train, "SUP_EPOCHS", 2)
    monkeypatch.setattr(ssl_train, "FT_EPOCHS", 2)
    monkeypatch.setattr(ssl_train, "BATCH_SIZE", 16)

    rng = np.random.default_rng(0)
    rows = []
    for p in range(24):
        for v in range(12):
            rows.append(
                {
                    "trial_id": f"P{p:02d}-{v:02d}",
                    "participant_id": f"P{p:02d}",
                    "recorded_day": 1 if v < 6 else 2,
                    "video": f"theme{v % 8}_{v}.mp4",
                    "video_theme": f"theme{v % 8}",
                    "rating_valence": float(rng.uniform(1, 9)),
                    "rating_arousal": float(rng.uniform(1, 9)),
                }
            )
    frame = common.assign_splits(pd.DataFrame(rows), "test-train-salt")
    n = len(frame)
    data = {
        "eeg": rng.normal(0, 1, (n, 62, 15, 128)).astype(np.float16),
        "eeg_scale": np.full((n, 62), 20.0, dtype=np.float16),
        "ecg": rng.normal(0, 1, (n, 2, 15, 128)).astype(np.float16),
        "eda": rng.normal(0, 1, (n, 2, 15, 16)).astype(np.float16),
        "resp": rng.normal(0, 1, (n, 2, 15, 16)).astype(np.float16),
        "valid": np.ones((n, 4, 15), dtype=bool),
    }
    data["valid"][:10, 2, :] = False  # missing EDA trials
    resting = {
        "pid": np.array(["P00", "P01", "P02", "P03"]),
        "eeg": rng.normal(0, 1, (4, 62, 15, 128)).astype(np.float16),
        "ecg": rng.normal(0, 1, (4, 2, 15, 128)).astype(np.float16),
        "eda": rng.normal(0, 1, (4, 2, 15, 16)).astype(np.float16),
        "resp": rng.normal(0, 1, (4, 2, 15, 16)).astype(np.float16),
        "valid": np.ones((4, 4, 15), dtype=bool),
    }
    device = torch.device("cpu")

    # 1. EEGNet & EEGConformer supervised step
    tr = np.where(frame["split"] == "dev")[0][:60]
    ev = np.where(frame["split"] != "dev")[0][:20]
    for kind in ("eegnet", "eeg_conformer"):
        pr, st = ssl_train.train_supervised_eeg_model(
            kind, data, frame, tr, ev, device, seed=0, epochs=2
        )
        assert pr["stress_va"].shape == (len(ev),)
        assert np.isfinite(pr["stress_va"]).all()
        assert st["n_params"] > 1000

    # 2. PhysioJEPA strategies + body-only + window slice + resting + fixed norm + FT
    for strat, mods, tsl, fix_eeg, rst in (
        ("jepa_crossmodal", (0, 1, 2, 3), (0, 15), False, resting),
        ("jepa_future", (0, 1, 2, 3), (3, 13), False, None),
        ("jepa_masked", (0,), (0, 15), True, None),
        ("jepa_crossmodal", (1, 2, 3), (3, 8), False, None),
        ("mae_recon", (0, 1, 2, 3), (0, 15), False, None),
    ):
        r_idx = np.arange(len(rst["pid"])) if rst is not None else None
        mdl, st_ssl = ssl_train.pretrain_physio_jepa(
            data,
            tr,
            device,
            strategy=strat,
            active_modalities=mods,
            time_slice=tsl,
            use_fixed_eeg=fix_eeg,
            resting_data=rst,
            resting_idx=r_idx,
            seed=0,
            epochs=2,
        )
        assert st_ssl["ctx_effective_rank"] >= 1.0
        rep = ssl_train.extract_jepa_features(
            mdl,
            data,
            ev,
            device,
            active_modalities=mods,
            time_slice=tsl,
            use_fixed_eeg=fix_eeg,
            shuffle_time_seed=7,
        )
        assert np.isfinite(rep["embed_delta"]).all()
        assert np.isfinite(rep["embed_raw"]).all()

    # Verify fine-tuning from pretrained JEPA
    pr_ft, _ = ssl_train.train_supervised_multimodal(
        data, frame, tr, ev, device, pretrained_jepa=mdl, seed=0, epochs=2
    )
    assert pr_ft["stress_va"].shape == (len(ev),)
    assert ssl_models.count_parameters(mdl.context_encoder) > 100000
