"""Paired evaluation of raw versus corrected EEG on new IDs and new themes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from decoding import study09_protocol as old

from habs_stress.pub_revision import bootstrap_mean, person_stress_aucs, run_r4_3x3, stress_label


def _donors(ids, video, rng):
    """Return a different trial of the same ID and a different ID watching the same video."""
    within_id = np.full(len(ids), -1, int)
    within_video = within_id.copy()
    for pid in np.unique(ids):
        positions = rng.permutation(np.flatnonzero(ids == pid))
        if len(positions) > 1:
            within_id[positions] = np.roll(positions, 1)
    for clip in np.unique(video):
        positions = np.flatnonzero(video == clip)
        for i in positions:
            others = positions[ids[positions] != ids[i]]
            if len(others):
                within_video[i] = rng.choice(others)
    return within_id, within_video


def _stimulus_interaction(frame, y, baseline, augmented):
    label = stress_label(frame).to_numpy()
    error_gain = (y - baseline) ** 2 - (y - augmented) ** 2
    values = []
    for idx in frame.groupby("participant_id").indices.values():
        pos = error_gain[idx[label[idx]]]
        neg = error_gain[idx[~label[idx]]]
        if min(len(pos), len(neg)) >= 3:
            values.append(pos.mean() - neg.mean())
    return bootstrap_mean(np.asarray(values))


def _pre0(pre):
    output = {}
    y = stress_label(pre).to_numpy()
    for name in ("prebody", "net_pre", "joint_pre0"):
        features = (
            np.column_stack([np.stack(pre.prebody), np.stack(pre.net_pre)])
            if name == "joint_pre0"
            else np.stack(pre[name])
        )
        pred = run_r4_3x3(pre, features, fit_logreg=False)["stress_va"]
        pmap = person_stress_aucs(pre, pred, y, ~y)
        output[name] = bootstrap_mean(np.asarray(list(pmap.values())))
    return output


def score(folder):
    grid = pd.read_parquet(folder / "study11_raw.parquet")
    release = pd.read_parquet(folder / "study09_reve_primary.parquet")
    cols = []
    for prefix in ("reve_cs_ctx8_early_", "reve_cs_ctx8_pool_", "reve_cs_ctx8_diff_"):
        cols.extend(c for c in release if c.startswith(prefix))
    if len(cols) != 1536:
        raise ValueError(f"Expected the three historical REVE blocks, got {len(cols)} columns")
    grid = grid.merge(release[["trial_id", *cols]], on="trial_id", how="left", validate="1:1")
    usable = grid[cols].notna().all(axis=1) & np.isfinite(grid["tgt_ecg_hr_chg_5_10"])
    frame = grid.loc[usable].reset_index(drop=True)
    y = frame["tgt_ecg_hr_chg_5_10"].to_numpy(float)
    b57 = frame[sorted(c for c in frame if c.startswith("b05_"))].to_numpy(dtype="float32")
    extra = np.stack(frame.extra)
    strong = np.column_stack([b57, extra])
    blocks = {
        k: np.stack(frame[k])
        for k in ("net_before", "net_after", "hc_before", "hc_after", "net_pre", "net_interior")
    }
    blocks["release"] = frame[cols].to_numpy(dtype="float32")
    n = len(frame)
    pf, tf = frame.person_fold.to_numpy(), frame.theme_fold.to_numpy()
    predictions = {}

    def keep(name, indices, pred):
        if name not in predictions:
            predictions[name] = np.full(n, np.nan)
        predictions[name][indices] = pred

    for p in range(3):
        for t in range(3):
            tr = np.flatnonzero((pf != p) & (tf != t))
            ev = np.flatnonzero((pf == p) & (tf == t))
            training = frame.iloc[tr]
            for name, body in [
                ("body57", b57),
                ("body_strong", strong),
                ("last_rr", extra[:, :17]),
            ]:
                pred, _ = old.fit_predict_nested_ridge(body[tr], y[tr], training, body[ev])
                keep(name, ev, pred)
            keep("body_gbdt", ev, old.fit_predict_gbdt_regressor(strong[tr], y[tr], strong[ev]))
            a, b = _donors(
                frame.participant_id.to_numpy()[ev],
                frame.video.to_numpy()[ev],
                np.random.default_rng(701 + 3 * p + t),
            )
            for key, eeg in blocks.items():
                configs = [("strong", strong)]
                if key in {"release", "net_before", "net_after"}:
                    configs.append(("57", b57))
                for kind, body in configs:
                    xt, xe = eeg[tr], eeg[ev]
                    bv = body[ev]
                    if key == "net_after" and kind == "strong":
                        xe = np.vstack([xe, xe[np.maximum(a, 0)], xe[np.maximum(b, 0)]])
                        bv = np.tile(bv, (3, 1))
                    pred, _ = old.fit_predict_m2_joint_dual_ridge(
                        body[tr], xt, y[tr], training, bv, xe
                    )
                    keep(f"{kind}_{key}", ev, pred[: len(ev)])
                    if len(pred) == 3 * len(ev):
                        for name, donor, vals in zip(
                            ("trial_shuffle", "other_id_same_video"),
                            (a, b),
                            np.split(pred[len(ev) :], 2),
                            strict=True,
                        ):
                            good = donor >= 0
                            keep(name, ev[good], vals[good])
            # Nonlinear reference uses an EEG-only PCA fitted on the outer training rows.
            ze, zv = old._preprocess_fit_transform(
                blocks["net_after"][tr], blocks["net_after"][ev], max_pca=32
            )
            keep(
                "gbdt_corrected",
                ev,
                old.fit_predict_gbdt_regressor(
                    np.column_stack([strong[tr], ze]), y[tr], np.column_stack([strong[ev], zv])
                ),
            )
            print(f"Scored raw fold {p}/{t} ({len(tr)} train, {len(ev)} test)", flush=True)

    summaries = {}
    for k, pred in predictions.items():
        summaries[k], _ = old.summarize_forecast(frame, y, pred)
    comparisons = {}
    for k in blocks:
        comparisons[f"strong_{k}-body"] = (f"strong_{k}", "body_strong")
    for k in ("release", "net_before", "net_after"):
        comparisons[f"57_{k}-body"] = (f"57_{k}", "body57")
    comparisons.update(
        {
            "corrected_minus_uncorrected": ("strong_net_after", "strong_net_before"),
            "raw_minus_release": ("strong_net_before", "strong_release"),
            "strong_body_minus_57": ("body_strong", "body57"),
            "nonlinear_eeg_gain": ("gbdt_corrected", "body_gbdt"),
            "matched_minus_other": ("strong_net_after", "other_id_same_video"),
            "matched_minus_shuffle": ("strong_net_after", "trial_shuffle"),
        }
    )
    paired = {}
    for label, (k, ref) in comparisons.items():
        m = np.isfinite(predictions[k]) & np.isfinite(predictions[ref])
        pa = old.evaluate_forecast_per_person(frame[m], y[m], predictions[k][m])
        pb = old.evaluate_forecast_per_person(frame[m], y[m], predictions[ref][m])
        paired[label] = old.paired_forecast_comparison(pa, pb)
        paired[label]["n_rows"] = int(m.sum())
    pre = pd.read_parquet(folder / "study11_pre0.parquet").reset_index(drop=True)
    return {
        "common_evaluation_trials": n,
        "common_evaluation_ids": frame.participant_id.nunique(),
        "models": summaries,
        "paired": paired,
        "distress_interaction_mse_bpm2": _stimulus_interaction(
            frame, y, predictions["body_strong"], predictions["strong_net_after"]
        ),
        "strict_pre0_classification": _pre0(pre),
        "ecg_coupling": {
            k: bootstrap_mean(frame.groupby("participant_id")[k].median().to_numpy())
            for k in ["median_abs_ecg_r_before", "median_abs_ecg_r_after"]
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(score(args.data_dir), indent=2, allow_nan=False))
