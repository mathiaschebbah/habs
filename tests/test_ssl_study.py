"""Unit tests for Study-08 CPU study orchestrator (ssl_study) on synthetic data."""

import numpy as np
import pandas as pd

from decoding import ssl_eval, ssl_study
from tests.test_decoding_revision import sim_folder  # noqa: F401


def test_ssl_study_end_to_end(sim_folder, monkeypatch):  # noqa: F811
    """Verify Study-07 baselines, FM probing, fusion, and paired gains on sim data."""
    monkeypatch.setattr(ssl_eval, "DRAWS", 30)
    frame = ssl_study.load_study08_table(sim_folder)
    rng = np.random.default_rng(7)
    n = len(frame)
    tids = frame["trial_id"].to_numpy()

    # Write one synthetic foundation model parquet + jepa embeddings + jepa preds
    fm_cols = {"trial_id": tids, "cbramod_valid": np.ones(n, dtype=bool)}
    for ph in ("ref", "early", "late"):
        for j in range(12):
            fm_cols[f"cbramod_{ph}_{j}"] = rng.normal(0, 1, n).astype(np.float32)
    pd.DataFrame(fm_cols).to_parquet(sim_folder / "fm_cbramod.parquet")

    jepa_cols = {"trial_id": tids}
    for k in ("z_ref", "z_early", "z_late", "z_eeg_early"):
        for j in range(12):
            jepa_cols[f"jepa_{k}_{j}"] = rng.normal(0, 1, n).astype(np.float32)
    pd.DataFrame(jepa_cols).to_parquet(sim_folder / "jepa_embeddings.parquet")

    pred_cols = {
        "trial_id": tids,
        "physio_jepa_frozen_ens_stress_va": rng.normal(0, 1, n).astype(np.float32),
        "eegnet_ens_stress_va": rng.normal(0, 1, n).astype(np.float32),
    }
    pd.DataFrame(pred_cols).to_parquet(sim_folder / "jepa_preds.parquet")

    s07, ref_full, ref_eeg, x_body, x_enr = ssl_study.run_study07_baselines(frame)
    assert "enriched_full" in s07
    assert "phase4_eeg_only" in s07
    assert "study07_handcrafted_diagnostics" in s07

    fm = ssl_study.evaluate_foundation_models(
        frame, sim_folder, ref_full, ref_eeg, x_body, x_enr
    )
    assert fm["cbramod"]["status"] == "ok"
    assert "fusion_plus_study07_enriched" in fm["cbramod"]
    assert fm["sjepa"]["status"] == "missing"

    ji = ssl_study.evaluate_jepa_fusion_and_paired_gains(
        frame, sim_folder, ref_full, ref_eeg, x_enr
    )
    assert "physio_jepa_plus_study07_enriched" in ji
    assert "physio_jepa_frozen_ens_stress_va" in ji["paired_gains_vs_study07"]
