"""Check Study-07 revision analyses on simulated data (never research evidence)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoding import common, person, revision, timecourse

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def fast_test_boots(monkeypatch):
    """Keep unit tests fast without changing production parameters."""
    monkeypatch.setattr(revision, "DRAWS", 40)
    monkeypatch.setattr(person, "DRAWS", 40)
    monkeypatch.setattr(timecourse, "DRAWS", 40)
    orig_boot = common.bootstrap
    orig_rboot = revision._boot_ci

    def _fast_common_boot(stats, value, draws=40, seed=0):
        return orig_boot(stats, value, draws=min(draws, 40), seed=seed)

    def _fast_rev_boot(values, draws=40, seed=0):
        return orig_rboot(values, draws=min(draws, 40), seed=seed)

    def _fast_balanced_model():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=200),
        )

    monkeypatch.setattr(common, "bootstrap", _fast_common_boot)
    monkeypatch.setattr(revision, "_boot_ci", _fast_rev_boot)
    monkeypatch.setattr(revision, "_balanced_model", _fast_balanced_model)


@pytest.fixture
def sim_folder(tmp_path):
    """Build synthetic trials + timecourse + hep + rich tables."""
    rng = np.random.default_rng(7)
    n_people, n_videos = 36, 25
    video_val = rng.uniform(1.5, 8.5, n_videos)
    video_aro = rng.uniform(2.0, 8.5, n_videos)

    trials, tc_rows, hep_rows, rich_rows = [], [], [], []
    chs = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")
    for p in range(n_people):
        for v in range(n_videos):
            tid = f"{p}-{v}"
            val = float(np.clip(video_val[v] + rng.normal(0, 1.5), 1.0, 9.0))
            aro = float(np.clip(video_aro[v] + rng.normal(0, 1.5), 1.0, 9.0))
            stressed = val <= 4.0 and aro >= 6.0
            if val <= 3.5 and aro >= 5.5:
                e1 = rng.choice(["fear", "anger", "disgust"])
            elif val <= 4.0:
                e1 = "sadness"
            elif val >= 6.0:
                e1 = rng.choice(["joy", "relaxation"])
            else:
                e1 = rng.choice(["neutral", "surprise", "curiosity"])
            tr = {
                "trial_id": tid,
                "participant_id": f"P{p:02d}",
                "cohort": 2 + (p % 3),
                "recorded_day": 1,
                "event_order": v,
                "start_marker": 1,
                "onset_s": float(v * 15),
                "video": f"theme{v % 10}_{v}.mp4",
                "video_theme": f"theme{v % 10}",
                "default_emotion_label": e1,
                "rating_valence": val,
                "rating_arousal": aro,
                "rating_dominance": 5.0,
                "recording": "x",
                "age": 30.0 + p,
                "gender": "F" if p % 2 == 0 else "M",
            }
            for ch in chs:
                for b in common.BANDS:
                    tr[f"bp_vid_{ch}_{b}"] = float(rng.normal(0, 0.5))
                    tr[f"bp_pre_{ch}_{b}"] = float(rng.normal(0, 0.5))
            trials.append(tr)

            tc = {
                "trial_id": tid,
                "final_emotion_label": e1,
                "self_reported_emotion_1": e1,
            }
            for sig in ("hr", "sc", "resp_amp", "resp_rate", *timecourse.EEG):
                arr = rng.normal(0, 0.5, timecourse.BINS)
                if sig == "hr" and stressed:
                    arr[4:11] -= 0.8
                if sig == "sc" and stressed:
                    arr[7:13] += 0.8
                for k, x in enumerate(arr):
                    tc[f"tc_{sig}_{k}"] = float(x)
            tc_rows.append(tc)

            hp = {
                "trial_id": tid,
                "beats_video": 12.0,
                "rr_video": float(rng.normal(0.85, 0.05)),
            }
            for w in ("pre", "video"):
                for k in range(16):
                    hp[f"ecg_{w}_{k}"] = float(rng.normal(0, 1))
                    for r in common.REGIONS:
                        hp[f"hep_{w}_{r}_{k}"] = float(rng.normal(0, 1))
            hep_rows.append(hp)

            rc = {
                "trial_id": tid,
                "self_reported_emotion_2": "fear" if stressed else "neutral",
                "self_reported_emotion_3": None,
                "expected_emotion_label": e1,
                "expected_emotion_source": "emotinder",
                "flagged_outliers": 1.0 if v == 0 else 0.0,
                "ecg_polarity": -1.0,
                "ecg_good_qrs_frac": 0.98,
                "hep_raw_beats_video": 11.0,
                "hep_raw200_beats_video": 12.0,
                "raw_fp_ptp_ref": 40.0,
                "raw_fp_ptp_0_2": 45.0 + (10.0 if stressed else 0.0),
                "raw_fp_ptp_early": 45.0,
                "raw_fp_ptp_late": 42.0,
            }
            for w in ("ref", "early", "late", "post", "vid"):
                shift = -3.0 if (stressed and w in ("early", "late", "vid")) else 0.0
                rc[f"ecg_hr_{w}"] = float(70.0 + shift + rng.normal(0, 3))
                rc[f"ecg_rr_{w}"] = float(850.0 - shift * 8 + rng.normal(0, 30))
                rc[f"ecg_rmssd_{w}"] = float(40.0 + rng.normal(0, 5))
                rc[f"ecg_pr_pk_{w}"] = float(125.0 + rng.normal(0, 4))
                rc[f"ecg_pr_on_{w}"] = float(150.0 + rng.normal(0, 5))
                rc[f"ecg_pr_seg_{w}"] = float(55.0 + rng.normal(0, 4))
                rc[f"ecg_qrs_{w}"] = float(80.0 + rng.normal(0, 4))
                rc[f"ecg_qt_{w}"] = float(360.0 + rng.normal(0, 10))
                rc[f"ecg_qtc_baz_{w}"] = float(400.0 + rng.normal(0, 12))
                rc[f"ecg_qtc_frid_{w}"] = float(385.0 + rng.normal(0, 10))
                rc[f"ecg_rt_pk_{w}"] = float(270.0 + rng.normal(0, 8))
                rc[f"ecg_t_amp_{w}"] = float(0.4 + rng.normal(0, 0.05))
            for w in ("ref", "early", "late", "post"):
                for st in ("mean", "max", "rise", "slope", "std"):
                    rc[f"sc_{st}_{w}"] = float(rng.normal(0, 0.5))
                for st in ("amp", "amp_std", "slope", "rate"):
                    rc[f"resp_{st}_{w}"] = float(rng.normal(0, 0.5))
                for b in common.BANDS:
                    for r in common.REGIONS:
                        rc[f"eeg_ep_{w}_{r}_{b}"] = float(rng.normal(0, 0.4))
                    for fam in (
                        "sp_std",
                        "sp_iqr",
                        "sp_ent",
                        "reg_std",
                        "ap_grad",
                        "faa",
                        "temp_asym",
                    ):
                        rc[f"eeg_{fam}_{w}_{b}"] = float(rng.normal(0, 0.2))
            for b in common.BANDS:
                for st in (
                    "tvar_ref",
                    "tvar_early",
                    "tvar_late",
                    "tvar_vid",
                    "tslope_early",
                    "tslope_late",
                    "tslope_vid",
                    "maxjump_early",
                    "maxjump_late",
                    "maxjump_vid",
                ):
                    rc[f"eeg_{st}_{b}"] = float(rng.normal(0, 0.2))
            for w in ("pre", "video"):
                for r in common.REGIONS:
                    for k in range(16):
                        bump = 1.2 if (w == "video" and stressed and k >= 9) else 0.0
                        rc[f"hep_raw_{w}_{r}_{k}"] = float(rng.normal(0, 1) + bump)
            for r in ("frontal", "central"):
                for k in range(16):
                    bump = 1.2 if (stressed and k >= 9) else 0.0
                    rc[f"hep_raw200_video_{r}_{k}"] = float(rng.normal(0, 1) + bump)
            rich_rows.append(rc)

    pd.DataFrame(trials).to_parquet(tmp_path / "trials.parquet")
    pd.DataFrame(tc_rows).to_parquet(tmp_path / "timecourse.parquet")
    pd.DataFrame(hep_rows).to_parquet(tmp_path / "hep.parquet")
    pd.DataFrame(rich_rows).to_parquet(tmp_path / "rich.parquet")
    return tmp_path


def test_track_a_splits(sim_folder):
    """Criterion, emotion_models, fixed_video, and sensitivity run cleanly."""
    table = revision.load_base(sim_folder)
    c = revision.criterion(table)
    assert c["threshold_sweep"]["v_le4_a_ge6"]["emotion_1"]["negative_purity"] > 0.8
    assert "fear" in c["emotion_1_va_profiles"]

    em = revision.emotion_models(table)
    assert "fear" in em["binary_classifiers"]
    assert em["multiclass_5class"]["balanced_accuracy"] > 0.0
    assert "kmeans_k4_profiles" in em["unsupervised_all"]

    fv = revision.fixed_video(table)
    assert fv["video_distribution"]["n_videos"] == 25
    assert "stress_vs_oklm_5" in fv["fixed_video_contrasts"]
    assert "high_consensus_ge50" in fv["matrix_2x2"]
    assert "valence" in fv["dose_response"]

    sn = revision.sensitivity(table)
    assert "frontal" in sn["regional_coupling"]
    assert "trait_stress_bias" in sn["trait_correlates_cohort_adjusted"]


def test_track_b_splits(sim_folder):
    """ecg_morphology, hep_raw, sequence_models, and criterion_full run cleanly."""
    table = revision.load_rich(sim_folder, with_hep=True)

    cf = revision.criterion_full(table)
    assert cf["flagged_outliers"]["flagged_fraction_all"] > 0

    em = revision.ecg_morphology(table)
    assert em["ecg_audit"]["inverted_lead_fraction"] == 1.0
    assert "pr_pk" in em["ecg_effects"]
    assert "delta_arousal_own_controlled" in em["ocular_proxy_control"]

    hr = revision.hep_raw(table)
    # Simulated raw HEP carries +1.2 stress bump while ICA HEP is pure noise
    assert hr["variants"]["raw_curated_100uV"]["contrasts_d"]["stress"][1] > 0.5
    assert (
        abs(hr["variants"]["ica_preprocessed_100uV"]["contrasts_d"]["stress"][0]) < 0.3
    )

    sm = revision.sequence_models(table)
    assert "sequence_plus_enriched_logreg" in sm["models"]
    assert "enriched_full_gbdt" in sm["models"]
