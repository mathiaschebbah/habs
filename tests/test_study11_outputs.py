"""Check that the published aggregates represent the stated paired experiments."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"


def load(name):
    return json.loads((ROOT / "study11" / f"{name}.json").read_text())


def test_temporal_curves_reproduce_historical_split_and_count_finite_ids():
    old = json.loads((ROOT / "timecourse.json").read_text())["curves"]["curves"]["dev"]["stress"]
    cov = load("controls")["figure1_per_bin_coverage"]
    for name, record in cov["original_timecourse"]["signals"].items():
        for key in ("d", "low", "high"):
            assert record[key] == old[name][key]
        assert record["eligible_ids"] == old[name]["people"]
    assert cov["audited_decoding"]["development_ids"] == 228
    for record in cov["audited_decoding"]["signals"].values():
        assert len(record["finite_ids_per_bin"]) == 15
        assert max(record["finite_ids_per_bin"]) <= record["eligible_ids"]
    strict = load("finite_curves")
    for split, result in strict.items():
        for name, signal in result["signals"].items():
            historical = cov[split]["signals"][name]["finite_ids_per_bin"]
            assert all(
                a <= b for a, b in zip(signal["finite_ids_per_bin"], historical, strict=True)
            )
            assert all(
                a <= x <= b
                for a, x, b in zip(signal["low"], signal["d"], signal["high"], strict=True)
            )


def test_training_budget_and_direction_of_paired_contrasts():
    for name in ("ladder_ridge", "ladder_reve", "same_ridge", "same_reve"):
        report = load(name)
        rows = {
            v["n_rows"]
            for k, v in report["pool_audit_mean_across_9_cells"].items()
            if k.startswith("matched_")
        }
        assert len(rows) == 1
        assert report["n_repeats"] == 10
        for model in report["models"].values():
            for head in ("stress_va", "stress_logreg"):
                m = model[head]["matched_n_train"]
                diff = m["contrasts"]["R4_minus_R1"]["diff"]
                assert abs(diff + m["contrasts"]["R1_minus_R4"]["diff"]) < 1e-7
                direct = (
                    m["regimes"]["R4_double_unseen"]["within_id_auc"]
                    - m["regimes"]["R1_random_pool"]["within_id_auc"]
                )
                assert abs(diff - direct) < 0.00016


def test_primary_cardiac_comparison_is_on_common_rows():
    coverage = load("raw_coverage")
    raw = load("raw_scores")
    assert coverage["source"] == "CURATED" and coverage["calibration_overlap_ids"] == 0
    assert (
        coverage["dev_trials"]
        == coverage["curated_grid_trials"] + coverage["non_curated_alignment_trials_excluded"]
    )
    assert (
        coverage["curated_grid_trials"]
        == coverage["calibration_trials"] + coverage["evaluation_trials_before_qc"]
    )
    assert coverage["bit_invariance_at_0_and_5"] == {"0.0": True, "5.0": True}
    assert raw["common_evaluation_trials"] <= coverage["evaluation_trials_after_qc"]
    primary = raw["paired"]["strong_net_after-body"]
    assert primary["n_people"] == raw["models"]["body_strong"]["n_people"]
    assert primary["n_rows"] == raw["common_evaluation_trials"]
    difference = (
        raw["models"]["body_strong"]["rmse"]["mean"]
        - raw["models"]["strong_net_after"]["rmse"]["mean"]
    )
    assert abs(primary["reduction_rmse"]["mean"] - difference) < 0.00016
