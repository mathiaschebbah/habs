"""Study-08 CPU scoring, fusion, cross-modal forecasting, and Study-07 comparison.

Runs on the ``decoding.yml`` CPU runner (``--study ssl``) after GPU Job 1
(``fm_*.parquet``) and GPU Job 2 (``jepa_embeddings.parquet``,
``jepa_preds.parquet``) complete. Joins Study-07's audited physiological tables
with Study-08's foundation-model and PhysioJEPA outputs on the exact frozen
fresh split ``STUDY08_SALT = "habs-ssl-2026-11-24-b"``.

Outputs aggregate JSON metrics only (no trial-level data ever leaves the runner).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from decoding import common, revision, ssl_eval, timecourse
from decoding.stress import contrasts

STUDY08_SALT = "habs-ssl-2026-11-24-b"
FM_NAMES = ("sjepa", "cbramod", "labram", "eegpt", "reve")


def load_study08_table(folder: Path | None = None) -> pd.DataFrame:
    """Load Study-07 tables (trials + timecourse + rich) on the Study-08 split."""
    base = Path(folder or os.environ["TRIAL_DATA"])
    table = pd.read_parquet(base / "trials.parquet")
    if (base / "body.parquet").is_file():
        body = pd.read_parquet(base / "body.parquet")
        table = table.drop(columns=[c for c in body if c != "trial_id" and c in table])
        table = table.merge(body, on="trial_id", how="left", validate="1:1")
    for fname in ("timecourse.parquet", "rich.parquet"):
        p = base / fname
        if p.is_file():
            extra = pd.read_parquet(p)
            keep = ["trial_id"] + [c for c in extra.columns if c not in table.columns]
            table = table.merge(extra[keep], on="trial_id", how="left", validate="1:1")
    return common.assign_splits(table, STUDY08_SALT)


def _body_late_targets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Late [5, 10) s autonomic targets (baseline-referenced, person-centered)."""
    targets = {}
    pairs = (
        ("hr_late_chg", "ecg_hr_late", "ecg_hr_ref"),
        ("sc_late_chg", "sc_mean_late", "sc_mean_ref"),
        ("resp_amp_late_chg", "resp_amp_late", "resp_amp_ref"),
    )
    ppl = frame["participant_id"]
    for name, c_late, c_ref in pairs:
        if c_late in frame and c_ref in frame:
            chg = frame[c_late] - frame[c_ref]
            chg = chg - chg.groupby(ppl).transform("mean")
            targets[name] = chg.to_numpy(float)
    return targets


def _study07_phase4_modality(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Extract EEG-only or Body-only 4-phase Study-07 features (person-centered)."""
    full = revision._build_phase_features(frame, add_deltas=True)
    if modality == "eeg":
        cols = [c for c in full.columns if c.startswith("eeg_")]
    elif modality == "body":
        cols = [c for c in full.columns if c.startswith(("ecg_", "sc_", "resp_"))]
    else:
        raise ValueError(modality)
    return full[cols]


def _study07_enriched_eeg(frame: pd.DataFrame) -> pd.DataFrame:
    """Strongest handcrafted EEG-only Study-07 representation (4-phase+spatial+tvar)."""
    full = revision._build_phase_features(
        frame,
        add_deltas=True,
        add_ecg_morph=False,
        add_eeg_spatial=True,
        add_eeg_tvar=True,
    )
    cols = [c for c in full.columns if c.startswith("eeg_")]
    return full[cols]


def _study07_phase_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble raw (ref, early, late) handcrafted multimodal feature vectors."""
    ref_cols, early_cols, late_cols = [], [], []
    for prefix, names in (
        ("ecg", ("hr", "rr")),
        ("sc", ("mean", "rise", "slope")),
        ("resp", ("amp", "rate", "slope")),
    ):
        for nm in names:
            if f"{prefix}_{nm}_ref" in frame:
                ref_cols.append(f"{prefix}_{nm}_ref")
                early_cols.append(f"{prefix}_{nm}_early")
                late_cols.append(f"{prefix}_{nm}_late")
    for r in common.REGIONS:
        for b in common.BANDS:
            if f"eeg_ep_ref_{r}_{b}" in frame:
                ref_cols.append(f"eeg_ep_ref_{r}_{b}")
                early_cols.append(f"eeg_ep_early_{r}_{b}")
                late_cols.append(f"eeg_ep_late_{r}_{b}")
    # Median-impute per column cleanly using dev medians only for geometry probes
    dev_m = (frame["split"] == "dev").to_numpy()

    def _mat(cols: list[str]) -> np.ndarray:
        arr = frame[cols].to_numpy(dtype=float, copy=True)
        med = np.nanmedian(arr[dev_m], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(~np.isfinite(arr))
        arr[inds] = np.take(med, inds[1])
        return arr

    return _mat(ref_cols), _mat(early_cols), _mat(late_cols)


def _strip_private(d: dict) -> dict:
    """Remove internal per-person maps before JSON serialization."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def run_study07_baselines(
    frame: pd.DataFrame,
) -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    """Re-evaluate Study-07 linear & sequence baselines on the Study-08 split."""
    x_1s = timecourse._features(frame, "all", 10)
    x_p4 = revision._build_phase_features(frame)
    x_p4_eeg = _study07_phase4_modality(frame, "eeg")
    x_p4_body = _study07_phase4_modality(frame, "body")
    x_enr_eeg = _study07_enriched_eeg(frame)
    x_enr_full = revision._build_phase_features(
        frame,
        add_deltas=True,
        add_ecg_morph=True,
        add_eeg_spatial=True,
        add_eeg_tvar=True,
    )
    x_seq = revision._build_multiscale_temporal_features(frame)
    x_seq_full = pd.concat([x_enr_full, x_seq], axis=1)

    # First evaluate primary references so all other models get paired gain CIs
    enr_full_res, _ = ssl_eval.probe_representation_all_regimes(
        frame, x_enr_full.to_numpy(np.float32), person_center=False, max_pca=512
    )
    ref_full_pp = enr_full_res["_per_person_va"]

    enr_eeg_res, _ = ssl_eval.probe_representation_all_regimes(
        frame,
        x_enr_eeg.to_numpy(np.float32),
        person_center=False,
        max_pca=512,
        ref_per_person=ref_full_pp,
    )
    ref_eeg_pp = enr_eeg_res["_per_person_va"]

    baselines = {
        "enriched_full": _strip_private(enr_full_res),
        "enriched_eeg_only": _strip_private(enr_eeg_res),
    }

    for name, df_x in (
        ("baseline_1s", x_1s),
        ("phase4_core", x_p4),
        ("phase4_eeg_only", x_p4_eeg),
        ("phase4_body_only", x_p4_body),
        ("sequence_plus_enriched", x_seq_full),
    ):
        res, _ = ssl_eval.probe_representation_all_regimes(
            frame,
            df_x.to_numpy(np.float32),
            person_center=False,
            max_pca=512,
            ref_per_person=ref_full_pp,
        )
        baselines[name] = _strip_private(res)

    # Geometry & cross-modal diagnostics for Study-07 handcrafted features
    zr, ze, zl = _study07_phase_arrays(frame)
    zv = 0.5 * (ze + zl)
    unseen_m = frame["split"].isin(["test_new_both", "test_new_person"]).to_numpy()
    unseen_df = frame[unseen_m]
    body_targets = _body_late_targets(frame)
    eeg_early_cols = [
        f"eeg_ep_early_{r}_{b}" for r in common.REGIONS for b in common.BANDS
    ]
    eeg_ref_cols = [f"eeg_ep_ref_{r}_{b}" for r in common.REGIONS for b in common.BANDS]
    eeg_early_chg = frame[eeg_early_cols].to_numpy(float) - frame[
        eeg_ref_cols
    ].to_numpy(float)
    eeg_early_chg = np.nan_to_num(eeg_early_chg, nan=0.0)

    baselines["study07_handcrafted_diagnostics"] = {
        "subject_identity_raw_vid": ssl_eval.probe_subject_identity(
            unseen_df, zv[unseen_m]
        ),
        "subject_identity_delta": ssl_eval.probe_subject_identity(
            unseen_df, (zv - zr)[unseen_m]
        ),
        "subject_identity_all_crossday_raw": ssl_eval.probe_subject_identity(frame, zv),
        "subject_identity_all_crossday_delta": ssl_eval.probe_subject_identity(
            frame, zv - zr
        ),
        "temporal_dynamics": ssl_eval.probe_temporal_dynamics(frame, zr, ze, zl),
        "cross_modal_eeg_early_to_late_body": ssl_eval.probe_cross_modal_forecast(
            frame, eeg_early_chg, body_targets
        ),
    }
    return baselines, ref_full_pp, ref_eeg_pp, x_p4_body, x_enr_full


def evaluate_foundation_models(
    frame: pd.DataFrame,
    folder: Path,
    ref_full_pp: dict,
    ref_eeg_pp: dict,
    x_body: pd.DataFrame,
    x_enr_full: pd.DataFrame,
) -> dict:
    """Evaluate all available pretrained EEG foundation model embeddings."""
    out = {}
    body_targets = _body_late_targets(frame)
    unseen_m = frame["split"].isin(["test_new_both", "test_new_person"]).to_numpy()
    unseen_df = frame[unseen_m]
    x_body_np = x_body.to_numpy(np.float32)
    x_enr_np = x_enr_full.to_numpy(np.float32)

    for name in FM_NAMES:
        pq = folder / f"fm_{name}.parquet"
        if not pq.is_file():
            out[name] = {"status": "missing"}
            continue
        df_fm = pd.read_parquet(pq)
        merged = frame[["trial_id"]].merge(
            df_fm, on="trial_id", how="left", validate="1:1"
        )
        ref_cols = [c for c in merged.columns if c.startswith(f"{name}_ref_")]
        early_cols = [c for c in merged.columns if c.startswith(f"{name}_early_")]
        late_cols = [c for c in merged.columns if c.startswith(f"{name}_late_")]
        if not ref_cols:
            out[name] = {"status": "empty"}
            continue

        zr = merged[ref_cols].to_numpy(np.float32)
        ze = merged[early_cols].to_numpy(np.float32)
        zl = merged[late_cols].to_numpy(np.float32)
        zv = 0.5 * (ze + zl)
        valid_col = f"{name}_valid"
        valid_frac = (
            round(float(merged[valid_col].fillna(False).mean()), 4)
            if valid_col in merged
            else 1.0
        )

        # 1. Baseline-referenced delta embedding (primary, person-centered)
        x_delta = np.hstack([ze - zr, zl - zr, zl - ze])
        res_delta, _ = ssl_eval.probe_representation_all_regimes(
            frame, x_delta, person_center=True, max_pca=128, ref_per_person=ref_eeg_pp
        )
        # Also compute paired gain vs full Study-07 enriched multimodal baseline
        res_delta_vs_full = {}
        pp_fm = res_delta["_per_person_va"]
        for reg in ("dev", *common.REGIMES):
            if reg in pp_fm and reg in ref_full_pp:
                res_delta_vs_full[reg] = ssl_eval.paired_gain_ci(
                    pp_fm[reg], ref_full_pp[reg]
                )

        # 2. Raw video embedding without ref subtraction (person-centered & uncentered)
        x_raw = np.hstack([ze, zl, zl - ze])
        res_raw_pc, _ = ssl_eval.probe_representation_all_regimes(
            frame, x_raw, person_center=True, max_pca=128, ref_per_person=ref_eeg_pp
        )
        res_raw_nopc, _ = ssl_eval.probe_representation_all_regimes(
            frame, x_raw, person_center=False, max_pca=128, ref_per_person=ref_eeg_pp
        )

        # 3. Multimodal fusion: FM delta + Study-07 Body and + Study-07 Enriched Full
        x_delta_c = ssl_eval.person_center_matrix(
            x_delta, frame["participant_id"].to_numpy()
        )
        res_plus_body, _ = ssl_eval.probe_representation_all_regimes(
            frame,
            np.hstack([x_delta_c, x_body_np]),
            person_center=False,
            max_pca=192,
            ref_per_person=ref_full_pp,
        )
        res_plus_enr, _ = ssl_eval.probe_representation_all_regimes(
            frame,
            np.hstack([x_delta_c, x_enr_np]),
            person_center=False,
            max_pca=256,
            ref_per_person=ref_full_pp,
        )

        # 4. Representation dynamics vs identity + cross-modal brain->body forecast
        diag = {
            "subject_identity_raw_vid": ssl_eval.probe_subject_identity(
                unseen_df, zv[unseen_m]
            ),
            "subject_identity_delta": ssl_eval.probe_subject_identity(
                unseen_df, (zv - zr)[unseen_m]
            ),
            "subject_identity_all_crossday_raw": ssl_eval.probe_subject_identity(
                frame, zv
            ),
            "subject_identity_all_crossday_delta": ssl_eval.probe_subject_identity(
                frame, zv - zr
            ),
            "temporal_dynamics": ssl_eval.probe_temporal_dynamics(frame, zr, ze, zl),
            "cross_modal_eeg_early_to_late_body": ssl_eval.probe_cross_modal_forecast(
                frame, ze - zr, body_targets
            ),
        }

        out[name] = {
            "status": "ok",
            "embed_dim": int(zr.shape[1]),
            "valid_fraction": valid_frac,
            "probe_delta_pc": _strip_private(res_delta),
            "gain_delta_vs_study07_full": res_delta_vs_full,
            "probe_raw_pc": _strip_private(res_raw_pc),
            "probe_raw_uncentered": _strip_private(res_raw_nopc),
            "fusion_plus_body": _strip_private(res_plus_body),
            "fusion_plus_study07_enriched": _strip_private(res_plus_enr),
            "diagnostics": diag,
        }
        d_auc = out[name]["probe_delta_pc"]["dev"]["stress_ridge_va"]["stress"]["auc"]
        t_auc = out[name]["probe_delta_pc"]["test_new_both"]["stress_ridge_va"][
            "stress"
        ]["auc"]
        f_auc = out[name]["fusion_plus_study07_enriched"]["test_new_both"][
            "stress_ridge_va"
        ]["stress"]["auc"]
        print(
            f"  FM {name}: dev_delta_va={d_auc}, new_both={t_auc}, fusion_enr={f_auc}",
            flush=True,
        )
    return out


def evaluate_jepa_fusion_and_paired_gains(
    frame: pd.DataFrame,
    folder: Path,
    ref_full_pp: dict,
    ref_eeg_pp: dict,
    x_enr_full: pd.DataFrame,
) -> dict:
    """Compute PhysioJEPA+Study07 fusion, brain->body forecast, and exact paired CIs."""
    out: dict = {}
    body_targets = _body_late_targets(frame)

    # 1. PhysioJEPA embeddings fusion & cross-modal forecast
    emb_pq = folder / "jepa_embeddings.parquet"
    if emb_pq.is_file():
        df_e = pd.read_parquet(emb_pq)
        merged = frame[["trial_id"]].merge(
            df_e, on="trial_id", how="left", validate="1:1"
        )
        zr = merged[[c for c in merged if c.startswith("jepa_z_ref_")]].to_numpy(
            np.float32
        )
        ze = merged[[c for c in merged if c.startswith("jepa_z_early_")]].to_numpy(
            np.float32
        )
        zl = merged[[c for c in merged if c.startswith("jepa_z_late_")]].to_numpy(
            np.float32
        )
        ze_eeg = merged[
            [c for c in merged if c.startswith("jepa_z_eeg_early_")]
        ].to_numpy(np.float32)
        x_delta_c = ssl_eval.person_center_matrix(
            np.hstack([ze - zr, zl - zr, zl - ze]), frame["participant_id"].to_numpy()
        )
        res_fuse, _ = ssl_eval.probe_representation_all_regimes(
            frame,
            np.hstack([x_delta_c, x_enr_full.to_numpy(np.float32)]),
            person_center=False,
            max_pca=256,
            ref_per_person=ref_full_pp,
        )
        out["physio_jepa_plus_study07_enriched"] = _strip_private(res_fuse)
        out["physio_jepa_cross_modal_eeg_to_body"] = (
            ssl_eval.probe_cross_modal_forecast(frame, ze_eeg, body_targets)
        )

    # 2. Exact paired bootstrap gains vs Study-07 enriched_full for every GPU model
    paired = {}
    dev_m = (frame["split"] == "dev").to_numpy()
    for pq_name in ("jepa_preds.parquet", "jepa_ablation_preds.parquet"):
        pred_pq = folder / pq_name
        if not pred_pq.is_file():
            continue
        df_p = pd.read_parquet(pred_pq)
        merged_p = frame[["trial_id"]].merge(
            df_p, on="trial_id", how="left", validate="1:1"
        )
        # Score all ensemble and seed0 stress_va / stress columns
        score_cols = [
            c
            for c in merged_p.columns
            if c.endswith(("_ens_stress_va", "_ens_stress", "_s0_stress_va"))
        ]
        for col in score_cols:
            s = pd.Series(merged_p[col].to_numpy(float), index=frame.index)
            entry = {}
            for reg, mask in (
                ("dev", dev_m),
                *((r, (frame["split"] == r).to_numpy()) for r in common.REGIMES),
            ):
                sub = frame[mask]
                ct = contrasts(sub)
                pmap = ssl_eval.person_stress_aucs(
                    sub, s[mask], ct["stress"][0], ct["stress"][1]
                )
                entry[reg] = {
                    "vs_study07_full": ssl_eval.paired_gain_ci(pmap, ref_full_pp[reg]),
                    "vs_study07_eeg": ssl_eval.paired_gain_ci(pmap, ref_eeg_pp[reg]),
                }
            paired[col] = entry
    if paired:
        out["paired_gains_vs_study07"] = paired

    return out


def main() -> None:
    """Run Study-08 CPU benchmark and write aggregate JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="main")
    parser.add_argument("--model", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    folder = Path(os.environ["TRIAL_DATA"])
    frame = load_study08_table(folder)

    print("=== Study 08: Study-07 Baselines on Fresh Split ===", flush=True)
    s07_res, ref_full_pp, ref_eeg_pp, x_body, x_enr_full = run_study07_baselines(frame)

    result: dict = {
        "study": "08_ssl_study",
        "salt": STUDY08_SALT,
        "splits": common.summary(frame),
        "study07_baselines": s07_res,
    }

    if args.split in ("main", "fm_only"):
        print("=== Study 08: Pretrained EEG Foundation Models ===", flush=True)
        result["foundation_models"] = evaluate_foundation_models(
            frame, folder, ref_full_pp, ref_eeg_pp, x_body, x_enr_full
        )

    if args.split in ("main", "jepa_fusion"):
        print("=== Study 08: PhysioJEPA Fusion & Paired Gains ===", flush=True)
        result["jepa_integration"] = evaluate_jepa_fusion_and_paired_gains(
            frame, folder, ref_full_pp, ref_eeg_pp, x_enr_full
        )

    text = json.dumps(result, indent=1, default=lambda x: None)
    print(text[:5000], flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(text)


if __name__ == "__main__":
    main()
