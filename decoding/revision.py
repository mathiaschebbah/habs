"""Study 07 analyses: comprehensive revision addressing all team critiques.

Track A splits (run directly on existing trials + body + timecourse tables):
- ``criterion``: rank-ordered ``self_reported_emotion_1`` audit inside the
  stress criterion, threshold purification sweep, purity/recall curves, and 2D
  valence-arousal profiles per emotion.
- ``emotion_models``: per-emotion physiological signatures, shared-vs-own
  decomposition, K binary classifiers, (K+1)-class multiclass classifier on raw
  ``self_reported_emotion_1`` (cohorts 2-5), and unsupervised PCA/k-means/GMM
  structure exploration.
- ``fixed_video``: per-video class-imbalance audit, non-adjacent fixed-video
  contrasts (stressed vs clearly calm/"oklm", pure valence, pure arousal),
  2x2 stimulus-consensus x viewer-state decomposition, and graded-vs-on/off
  dose-response tests.
- ``sensitivity``: 6-region x 5-band + 62-channel + frontal-asymmetry EEG
  topography of idiosyncratic sensitivity (beta_own) and between-person trait
  stress sensitivity correlates.

Track B splits (require ``rich.parquet`` from ``decoding.rich_extract``):
- ``ecg_morphology``: beat-by-beat ECG P-QRS-T delineation audit (PR peak,
  PR interval, PR segment / AV node, QRS, QT/QTc, RT, T-amp), RR-controlled PR
  tests, 4-phase EEG spatial homogeneity & temporal variation, and raw
  frontopolar ocular-proxy control on early delta.
- ``hep_raw``: direct empirical comparison of unprocessed CURATED HEP (CAR,
  0.5-30 Hz, NO ICA) vs ICA-cleaned PREPROCESSED HEP and exact ICA-removed
  component (CURATED - PREPROCESSED) locked to identical R peaks.
- ``sequence_models``: fresh person x theme split benchmarking linear 1-s
  baseline, 4-phase, 4-phase + explicit t->t+1 deltas, full enriched linear &
  ridge, HistGradientBoosting, and multi-scale temporal-convolution / sequence
  models (within-person AUC and same-video AUC), plus feature-group ablations.
- ``criterion_full``: extends criterion audit with ``self_reported_emotion_2``,
  ``self_reported_emotion_3``, and ``flagged_outliers`` exclusion sensitivity.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoding import common, person, timecourse
from decoding.stress import contrasts, label

SALT = "habs-revision-2026-11-23"
DRAWS = 2000
MIN_EACH = 3
MIN_TRIALS = 10

NEGATIVE_ALL = ("anger", "fear", "disgust", "sadness", "guilt")
ACUTE_THREAT = ("anger", "fear", "disgust")
SAD_GUILT = ("sadness", "guilt")
INCOHERENT = ("joy", "excitement", "relaxation", "sensuality", "neutral", "curiosity")
ALL_12 = (*NEGATIVE_ALL, "surprise", *INCOHERENT)
TOP_NEG = ("fear", "anger", "disgust", "sadness")
PHASES = ("ref", "early", "late", "post")
LEFT_FRONTAL = ("Fp1", "AF3", "AF7", "F1", "F3", "F5", "F7")
RIGHT_FRONTAL = ("Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8")


def _clean_str(series):
    return series.astype("string").str.strip().str.lower()


def load_base(folder=None):
    """Trials + body + 1-s timecourse with Study-07 deterministic splits."""
    folder = Path(folder or os.environ["TRIAL_DATA"])
    table = common.load(SALT, folder)
    bins = pd.read_parquet(folder / "timecourse.parquet")
    return table.merge(bins, on="trial_id", how="inner", validate="1:1")


def load_rich(folder=None, *, with_hep=False):
    """Join ``rich.parquet`` (and optionally ``hep.parquet``) onto base table."""
    folder = Path(folder or os.environ["TRIAL_DATA"])
    table = load_base(folder)
    rich = pd.read_parquet(folder / "rich.parquet")
    table = table.merge(rich, on="trial_id", how="inner", validate="1:1")
    if with_hep and (folder / "hep.parquet").is_file():
        hep = pd.read_parquet(folder / "hep.parquet")
        table = table.merge(hep, on="trial_id", how="inner", validate="1:1")
    return table


def _boot_ci(values, draws=DRAWS, seed=0):
    """Mean and 95% percentile bootstrap CI over rows of a 1D array."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return [None, None, None, 0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), (draws, len(arr)))
    boot = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [
        round(float(arr.mean()), 4),
        round(float(lo), 4),
        round(float(hi), 4),
        len(arr),
    ]


def _person_d(values, pos, neg, people, min_each=MIN_EACH):
    """Within-person Cohen's d between boolean masks ``pos`` and ``neg``."""
    arr = np.asarray(values, dtype=float)
    pos, neg = np.asarray(pos, dtype=bool), np.asarray(neg, dtype=bool)
    people = np.asarray(people)
    ds = []
    for p in np.unique(people[pos | neg]):
        mine = people == p
        a = arr[mine & pos]
        b = arr[mine & neg]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < min_each or len(b) < min_each:
            continue
        pooled = np.sqrt(
            (np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1))
            / (len(a) + len(b) - 2)
        )
        if pooled > 0:
            ds.append((a.mean() - b.mean()) / pooled)
    return _boot_ci(ds)


def _person_diff(values, pos, neg, people, min_each=MIN_EACH):
    """Within-person raw mean difference (pos - neg)."""
    arr = np.asarray(values, dtype=float)
    pos, neg = np.asarray(pos, dtype=bool), np.asarray(neg, dtype=bool)
    people = np.asarray(people)
    diffs = []
    for p in np.unique(people[pos | neg]):
        mine = people == p
        a = arr[mine & pos]
        b = arr[mine & neg]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < min_each or len(b) < min_each:
            continue
        diffs.append(a.mean() - b.mean())
    return _boot_ci(diffs)


def _within_person_auc_masks(frame, score, pos, neg, min_each=MIN_EACH):
    """Within-person ROC AUC between arbitrary ``pos`` and ``neg`` masks."""
    s = np.asarray(score, dtype=float)
    pos, neg = np.asarray(pos, dtype=bool), np.asarray(neg, dtype=bool)
    people = frame["participant_id"].to_numpy()
    aucs = []
    for p in np.unique(people[pos | neg]):
        mine = (people == p) & np.isfinite(s)
        a, b = mine & pos, mine & neg
        if a.sum() < min_each or b.sum() < min_each:
            continue
        keep = a | b
        aucs.append(roc_auc_score(a[keep], s[keep]))
    m, lo, hi, n = _boot_ci(aucs)
    return {"auc": m, "ci95": [lo, hi], "people": n}


def _same_video_stats(frame, score, pos, neg, min_each=MIN_EACH, video_mask=None):
    """Pair-weighted & unweighted same-video AUC plus within-video Cohen's d."""
    s = np.asarray(score, dtype=float)
    pos, neg = np.asarray(pos, dtype=bool), np.asarray(neg, dtype=bool)
    rows = []
    unweighted_aucs = []
    ds = []
    raw_diffs = []
    for v, index in frame.groupby("video").indices.items():
        if video_mask is not None and v not in video_mask:
            continue
        idx = np.asarray(index)
        ok = np.isfinite(s[idx])
        a = s[idx[pos[idx] & ok]]
        b = s[idx[neg[idx] & ok]]
        if len(a) < min_each or len(b) < min_each:
            continue
        diff = a[:, None] - b[None, :]
        greater = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        pairs = len(a) * len(b)
        rows.append((greater, pairs))
        unweighted_aucs.append(greater / pairs)
        raw_diffs.append(float(a.mean() - b.mean()))
        pooled = np.sqrt(
            (np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1))
            / (len(a) + len(b) - 2)
        )
        if pooled > 0:
            ds.append(float((a.mean() - b.mean()) / pooled))
    if not rows:
        return {
            "auc_weighted": None,
            "ci95_weighted": None,
            "auc_unweighted": None,
            "ci95_unweighted": None,
            "d_within_video": None,
            "diff_within_video": None,
            "videos": 0,
            "pairs": 0,
        }
    arr = np.array(rows, dtype=float)
    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(arr), (DRAWS, len(arr)))
    boot_w = arr[bidx, 0].sum(1) / arr[bidx, 1].sum(1)
    lo_w, hi_w = np.percentile(boot_w, [2.5, 97.5])
    uw = _boot_ci(unweighted_aucs)
    return {
        "auc_weighted": round(float(arr[:, 0].sum() / arr[:, 1].sum()), 4),
        "ci95_weighted": [round(float(lo_w), 4), round(float(hi_w), 4)],
        "auc_unweighted": uw[0],
        "ci95_unweighted": [uw[1], uw[2]],
        "d_within_video": _boot_ci(ds),
        "diff_within_video": _boot_ci(raw_diffs),
        "videos": len(rows),
        "pairs": int(arr[:, 1].sum()),
    }


def _safe_mean_ci(df):
    """Wrap ``person._mean_ci`` safely for empty DataFrames."""
    if df is None or len(df) == 0:
        return {"people": 0}
    return person._mean_ci(df)


def _balanced_model():
    """L2 logistic regression with balanced class weights (no fold prior shift)."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-4, 0, 9),
            l1_ratios=(0,),
            class_weight="balanced",
            scoring="neg_log_loss",
            use_legacy_attributes=False,
            max_iter=2000,
        ),
    )


def _cross_fit_binary(frame, x, y, *, model_fn=None):
    """3x3 person x theme cross-fit returning out-of-fold continuous scores."""
    y_arr = np.asarray(y, dtype=int)
    model_fn = model_fn or _balanced_model
    blocks = [(p, t) for p in range(3) for t in range(3)]

    def one(p, t):
        test_m = (frame["person_fold"] == p) & (frame["theme_fold"] == t)
        train_m = (frame["person_fold"] != p) & (frame["theme_fold"] != t)
        n_pos = int(y_arr[train_m].sum())
        if n_pos < 5 or int(train_m.sum()) - n_pos < 5:
            return pd.Series(np.nan, index=frame.index[test_m])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = model_fn().fit(x.loc[train_m], y_arr[train_m])
            if hasattr(clf, "decision_function"):
                sc = clf.decision_function(x.loc[test_m])
            else:
                sc = clf.predict_proba(x.loc[test_m])[:, 1]
        return pd.Series(sc, index=frame.index[test_m])

    parts = Parallel(n_jobs=-1)(delayed(one)(p, t) for p, t in blocks)
    return pd.concat(parts).loc[frame.index]


def _base_responses(frame):
    """Return the five Study-06 physiological summary responses."""
    return {name: person.response(frame, name) for name in person.RESPONSES}


# ---------------------------------------------------------------------------
# SPLIT 1: CRITERION SANITY CHECK & THRESHOLD PURIFICATION
# ---------------------------------------------------------------------------


def _purity_stats(emo_series, pos_mask):
    """Rank-ordered emotion breakdown, purity, and recall inside ``pos_mask``."""
    emo = _clean_str(emo_series)
    known = emo.notna() & (emo != "") & (emo != "nan") & (emo != "none")
    pos_known = pos_mask & known
    n_pos_k = int(pos_known.sum())
    counts_pos = emo[pos_known].value_counts().to_dict()
    counts_all = emo[known].value_counts().to_dict()
    if n_pos_k == 0:
        return {"n_known": 0}
    pct_pos = {k: round(v / n_pos_k, 4) for k, v in counts_pos.items()}
    neg_n = sum(counts_pos.get(e, 0) for e in NEGATIVE_ALL)
    threat_n = sum(counts_pos.get(e, 0) for e in ACUTE_THREAT)
    sad_g_n = sum(counts_pos.get(e, 0) for e in SAD_GUILT)
    inc_n = sum(counts_pos.get(e, 0) for e in INCOHERENT)
    surp_n = counts_pos.get("surprise", 0)
    recall = {
        e: round(counts_pos.get(e, 0) / counts_all[e], 4)
        for e in (*NEGATIVE_ALL, "surprise")
        if counts_all.get(e, 0) > 0
    }
    return {
        "n_known": n_pos_k,
        "counts_ranked": {str(k): int(v) for k, v in counts_pos.items()},
        "pct_ranked": {str(k): v for k, v in pct_pos.items()},
        "negative_purity": round(neg_n / n_pos_k, 4),
        "acute_threat_purity": round(threat_n / n_pos_k, 4),
        "sadness_guilt_rate": round(sad_g_n / n_pos_k, 4),
        "surprise_rate": round(surp_n / n_pos_k, 4),
        "incoherent_rate": round(inc_n / n_pos_k, 4),
        "recall_by_emotion": recall,
    }


def criterion(frame):
    """Sanity check of V/A stress criterion + threshold purification sweep."""
    v = frame["rating_valence"]
    a = frame["rating_arousal"]
    people = frame["participant_id"].to_numpy()
    resps = _base_responses(frame)
    x_10s = timecourse._features(frame, "all", 10)

    thresholds = {
        "v_lt5_a_gt5": (v < 5.0) & (a > 5.0),
        "v_le4.5_a_ge5.5": (v <= 4.5) & (a >= 5.5),
        "v_le4_a_ge6": (v <= 4.0) & (a >= 6.0),
        "v_le3.5_a_ge6.5": (v <= 3.5) & (a >= 6.5),
        "v_le3_a_ge7": (v <= 3.0) & (a >= 7.0),
        "v_le2.5_a_ge7.5": (v <= 2.5) & (a >= 7.5),
        "v_le2_a_ge8": (v <= 2.0) & (a >= 8.0),
        "asym_v4_a5": (v <= 4.0) & (a > 5.0),
        "asym_v4_a7": (v <= 4.0) & (a >= 7.0),
        "asym_v3_a6": (v <= 3.0) & (a >= 6.0),
    }

    sweep = {}
    for name, mask in thresholds.items():
        pos = mask.to_numpy(bool)
        neg = ~pos
        sc = _cross_fit_binary(frame, x_10s, pos)
        sweep[name] = {
            "n_trials": int(pos.sum()),
            "fraction": round(float(pos.mean()), 4),
            "emotion_1": _purity_stats(frame["self_reported_emotion_1"], pos),
            "final_emotion": _purity_stats(frame["final_emotion_label"], pos),
            "effect_d": {
                rname: _person_d(rval, pos, neg, people)
                for rname, rval in resps.items()
            },
            "within_person_auc": _within_person_auc_masks(frame, sc, pos, neg),
            "same_video": _same_video_stats(frame, sc, pos, neg),
        }

    # Per-emotion 2D valence-arousal distribution (raw self_reported_emotion_1)
    emo1 = _clean_str(frame["self_reported_emotion_1"])
    known = emo1.notna() & (emo1 != "") & (emo1 != "nan")
    profiles = {}
    for em in ALL_12:
        m = known & (emo1 == em)
        if m.sum() == 0:
            continue
        vm, am = v[m], a[m]
        profiles[em] = {
            "n": int(m.sum()),
            "pct_of_known": round(float(m.sum() / known.sum()), 4),
            "valence_mean": round(float(vm.mean()), 3),
            "valence_sd": round(float(vm.std()), 3),
            "valence_q": [round(float(x), 2) for x in vm.quantile([0.25, 0.5, 0.75])],
            "arousal_mean": round(float(am.mean()), 3),
            "arousal_sd": round(float(am.std()), 3),
            "arousal_q": [round(float(x), 2) for x in am.quantile([0.25, 0.5, 0.75])],
            "in_v_lt5_a_gt5": round(float(((vm < 5) & (am > 5)).mean()), 4),
            "in_v_le4_a_ge6": round(float(((vm <= 4) & (am >= 6)).mean()), 4),
            "in_v_le3_a_ge7": round(float(((vm <= 3) & (am >= 7)).mean()), 4),
        }

    return {
        "n_total": len(frame),
        "n_with_emotion_1": int(known.sum()),
        "cohort_emotion_1_coverage": {
            str(int(c)): round(float(known[frame["cohort"] == c].mean()), 4)
            for c in sorted(frame["cohort"].dropna().unique())
        },
        "threshold_sweep": sweep,
        "emotion_1_va_profiles": profiles,
    }


def criterion_full(frame):
    """Audit emotion_2/3 ranks and flagged_outliers sensitivity."""
    v, a = frame["rating_valence"], frame["rating_arousal"]
    stressed = ((v <= 4.0) & (a >= 6.0)).to_numpy(bool)
    e1 = _clean_str(frame["self_reported_emotion_1"])
    e2 = _clean_str(frame["self_reported_emotion_2"])
    e3 = _clean_str(frame["self_reported_emotion_3"])
    known = e1.notna() & (e1 != "") & (e1 != "nan")

    # For stressed trials with incoherent #1 emotion, check if #2 or #3 is negative
    inc_stress = stressed & known & e1.isin(INCOHERENT)
    surp_stress = stressed & known & (e1 == "surprise")
    any_neg_23 = e2.isin(NEGATIVE_ALL) | e3.isin(NEGATIVE_ALL)
    any_threat_123 = (
        e1.isin(ACUTE_THREAT) | e2.isin(ACUTE_THREAT) | e3.isin(ACUTE_THREAT)
    )
    any_neg_123 = e1.isin(NEGATIVE_ALL) | e2.isin(NEGATIVE_ALL) | e3.isin(NEGATIVE_ALL)

    # Outlier sensitivity: exclude flagged_outliers == 1
    clean_mask = (frame["flagged_outliers"].fillna(0) == 0).to_numpy(bool)
    clean_frame = frame[clean_mask]
    resps_clean = _base_responses(clean_frame)
    pos_c = label(clean_frame).to_numpy(bool)
    people_c = clean_frame["participant_id"].to_numpy()

    return {
        "any_rank_123_in_stress": {
            "n_stress_known": int((stressed & known).sum()),
            "pct_any_negative_123": round(
                float(any_neg_123[stressed & known].mean()), 4
            ),
            "pct_any_acute_threat_123": round(
                float(any_threat_123[stressed & known].mean()), 4
            ),
            "n_incoherent_e1": int(inc_stress.sum()),
            "pct_incoherent_e1_with_neg_e23": round(
                float(any_neg_23[inc_stress].mean()), 4
            )
            if inc_stress.any()
            else None,
            "n_surprise_e1": int(surp_stress.sum()),
            "pct_surprise_e1_with_neg_e23": round(
                float(any_neg_23[surp_stress].mean()), 4
            )
            if surp_stress.any()
            else None,
        },
        "flagged_outliers": {
            "flagged_fraction_all": round(float((~clean_mask).mean()), 4),
            "flagged_fraction_stress": round(float((~clean_mask)[stressed].mean()), 4),
            "clean_n": int(clean_mask.sum()),
            "clean_purity": _purity_stats(
                clean_frame["self_reported_emotion_1"], pos_c
            ),
            "clean_effect_d": {
                rname: _person_d(rval, pos_c, ~pos_c, people_c)
                for rname, rval in resps_clean.items()
            },
            "clean_coupling_heart_valence": _safe_mean_ci(
                person._person_betas(
                    clean_frame, resps_clean["heart"], clean_frame["rating_valence"]
                )
            ),
            "clean_coupling_skin_arousal": _safe_mean_ci(
                person._person_betas(
                    clean_frame, resps_clean["skin"], clean_frame["rating_arousal"]
                )
            ),
        },
    }


# ---------------------------------------------------------------------------
# SPLIT 2: EMOTION-SPECIFIC MODELS & UNSUPERVISED EXPLORATION
# ---------------------------------------------------------------------------


def _unsupervised_block(frame):
    """Zero-label PCA, k-means, and GMM structure on person-centred physiology."""
    resps = _base_responses(frame)
    # Compact, interpretable person-centred physiological matrix (36 features)
    reg_cols = {}
    for sig in ("hr", "sc", "resp_amp", "resp_rate"):
        arr = timecourse.corrected(frame, sig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reg_cols[f"{sig}_early"] = np.nanmean(arr[:, 3:8], axis=1)
            reg_cols[f"{sig}_late"] = np.nanmean(arr[:, 8:13], axis=1)
    for r in common.REGIONS:
        for b in common.BANDS:
            arr = timecourse.corrected(frame, f"eeg_{r}_{b}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                reg_cols[f"eeg_{r}_{b}"] = np.nanmean(arr[:, 3:13], axis=1)
    feat = pd.DataFrame(reg_cols, index=frame.index)
    # Person-centre every feature so clusters capture state, not person identity
    feat = feat - feat.groupby(frame["participant_id"].to_numpy()).transform("mean")
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    z = pipe.fit_transform(feat)

    pca = PCA(n_components=10, random_state=0)
    pcs = pca.fit_transform(z)
    fnames = list(feat.columns)
    top_loadings = {}
    for k in range(3):
        order = np.argsort(np.abs(pca.components_[k]))[::-1][:6]
        top_loadings[f"PC{k + 1}"] = {
            fnames[i]: round(float(pca.components_[k, i]), 3) for i in order
        }

    # Within-person correlations of PC1-PC5 with valence, arousal, stress
    people = frame["participant_id"].to_numpy()
    pc_corrs = {}
    targets = {
        "valence": frame["rating_valence"].to_numpy(float),
        "arousal": frame["rating_arousal"].to_numpy(float),
        "stress": label(frame).to_numpy(float),
    }
    for k in range(5):
        pc_corrs[f"PC{k + 1}"] = {}
        for tname, tvals in targets.items():
            rs = []
            for p in np.unique(people):
                m = (people == p) & np.isfinite(tvals)
                if m.sum() >= MIN_TRIALS and np.std(tvals[m]) > 0:
                    rs.append(np.corrcoef(pcs[m, k], tvals[m])[0, 1])
            pc_corrs[f"PC{k + 1}"][tname] = _boot_ci(rs, draws=1000)

    # Clustering on top 8 PCs (denoises high-dim Euclidean distance)
    z_pca = pcs[:, :8]
    emo1 = _clean_str(frame["self_reported_emotion_1"])
    emo5 = np.where(emo1.isin(TOP_NEG), emo1.astype(str), "other")
    known = emo1.notna() & (emo1 != "") & (emo1 != "nan")
    stress_y = label(frame).to_numpy(int)
    themes = frame["theme"].astype(str).to_numpy()
    videos = frame["video"].astype(str).to_numpy()

    rng = np.random.default_rng(0)
    sil_idx = rng.choice(len(z_pca), size=min(4000, len(z_pca)), replace=False)

    km_res, gmm_res = {}, {}
    k4_profiles = {}
    for k in (2, 3, 4, 5, 6):
        cl = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(z_pca)
        km_res[str(k)] = {
            "silhouette": round(
                float(silhouette_score(z_pca[sil_idx], cl[sil_idx])), 4
            ),
            "ari_stress": round(float(adjusted_rand_score(stress_y, cl)), 5),
            "nmi_stress": round(float(normalized_mutual_info_score(stress_y, cl)), 5),
            "ari_emotion5": round(
                float(adjusted_rand_score(emo5[known], cl[known])), 5
            ),
            "nmi_emotion5": round(
                float(normalized_mutual_info_score(emo5[known], cl[known])), 5
            ),
            "nmi_theme": round(float(normalized_mutual_info_score(themes, cl)), 5),
            "nmi_video": round(float(normalized_mutual_info_score(videos, cl)), 5),
        }
        if k == 4:
            for c in range(4):
                m = cl == c
                mk = m & known
                em_cnt = pd.Series(emo1[mk]).value_counts(normalize=True).head(5)
                k4_profiles[f"cluster_{c}"] = {
                    "n": int(m.sum()),
                    "stress_rate": round(float(stress_y[m].mean()), 4),
                    "valence_mean": round(float(frame["rating_valence"][m].mean()), 3),
                    "arousal_mean": round(float(frame["rating_arousal"][m].mean()), 3),
                    "mean_responses": {
                        rn: round(float(np.nanmean(rv[m])), 4)
                        for rn, rv in resps.items()
                    },
                    "top_emotions": {
                        str(ek): round(float(ev), 4) for ek, ev in em_cnt.items()
                    },
                }
    for k in (2, 3, 4, 5):
        cl = GaussianMixture(
            n_components=k, covariance_type="diag", random_state=0
        ).fit_predict(z_pca)
        gmm_res[str(k)] = {
            "ari_stress": round(float(adjusted_rand_score(stress_y, cl)), 5),
            "nmi_stress": round(float(normalized_mutual_info_score(stress_y, cl)), 5),
            "nmi_emotion5": round(
                float(normalized_mutual_info_score(emo5[known], cl[known])), 5
            ),
            "nmi_video": round(float(normalized_mutual_info_score(videos, cl)), 5),
        }

    return {
        "pca_explained_variance": [
            round(float(x), 4) for x in pca.explained_variance_ratio_
        ],
        "pca_top_loadings": top_loadings,
        "pca_within_person_corrs": pc_corrs,
        "kmeans": km_res,
        "gmm": gmm_res,
        "kmeans_k4_profiles": k4_profiles,
    }


def emotion_models(frame):
    """Evaluate binary, (K+1) multiclass, and unsupervised emotion models."""
    emo1 = _clean_str(frame["self_reported_emotion_1"])
    known = emo1.notna() & (emo1 != "") & (emo1 != "nan") & (emo1 != "none")
    sub = frame[known].copy()
    em = emo1[known]
    people = sub["participant_id"].to_numpy()
    resps = _base_responses(sub)

    # 1. Per-emotion physiological signatures (vs neutral AND vs all other trials)
    is_neutral = (em == "neutral").to_numpy(bool)
    signatures = {}
    for target in (
        *TOP_NEG,
        "surprise",
        "joy",
        "excitement",
        "relaxation",
        "curiosity",
    ):
        pos = (em == target).to_numpy(bool)
        signatures[target] = {
            "n": int(pos.sum()),
            "vs_other_d": {
                rn: _person_d(rv, pos, ~pos, people) for rn, rv in resps.items()
            },
            "vs_neutral_d": {
                rn: _person_d(rv, pos, is_neutral, people) for rn, rv in resps.items()
            },
        }

    # 2. Binary classifiers (K one-vs-rest + core_threat + all_negative)
    x_10s = timecourse._features(sub, "all", 10)
    binary_targets = {
        **{
            e: (em == e).to_numpy(bool)
            for e in (*TOP_NEG, "joy", "relaxation", "neutral")
        },
        "core_threat": em.isin(ACUTE_THREAT).to_numpy(bool),
        "all_negative": em.isin(NEGATIVE_ALL).to_numpy(bool),
    }
    binary_out = {}
    for name, pos in binary_targets.items():
        neg = ~pos
        sc = _cross_fit_binary(sub, x_10s, pos)
        # Crowd leave-one-out consensus for this emotion
        y_f = pd.Series(pos.astype(float), index=sub.index)
        tot = y_f.groupby(sub["video"]).transform("sum")
        cnt = y_f.groupby(sub["video"]).transform("count")
        cons = ((tot - y_f) / (cnt - 1)).where(cnt - 1 >= person.MIN_OTHERS)
        keep_c = cons.notna().to_numpy(bool)
        # Shared vs own coupling of heart & skin with this binary emotion
        coup_hr = _safe_mean_ci(person._person_betas(sub, resps["heart"], y_f))
        coup_sc = _safe_mean_ci(person._person_betas(sub, resps["skin"], y_f))
        coup_dl = _safe_mean_ci(person._person_betas(sub, resps["delta"], y_f))
        binary_out[name] = {
            "n_pos": int(pos.sum()),
            "base_rate": round(float(pos.mean()), 4),
            "within_person_ovr": _within_person_auc_masks(sub, sc, pos, neg),
            "within_person_vs_neutral": _within_person_auc_masks(
                sub, sc, pos, is_neutral
            )
            if name != "neutral"
            else None,
            "same_video_ovr": _same_video_stats(sub, sc, pos, neg),
            "consensus_within_person": _within_person_auc_masks(
                sub[keep_c], cons[keep_c], pos[keep_c], neg[keep_c]
            ),
            "coupling_heart": {
                "beta_shared": coup_hr["beta_shared"],
                "beta_own": coup_hr["beta_own"],
            },
            "coupling_skin": {
                "beta_shared": coup_sc["beta_shared"],
                "beta_own": coup_sc["beta_own"],
            },
            "coupling_delta": {
                "beta_shared": coup_dl["beta_shared"],
                "beta_own": coup_dl["beta_own"],
            },
        }

    # 3. Unified (K+1)-class multiclass classifier: fear, anger, disgust, sadness, other
    classes = [*TOP_NEG, "other"]
    y_mc = np.where(em.isin(TOP_NEG), em.astype(str), "other")
    blocks = [(p, t) for p in range(3) for t in range(3)]

    def one_mc(p, t):
        test_m = (sub["person_fold"] == p) & (sub["theme_fold"] == t)
        train_m = (sub["person_fold"] != p) & (sub["theme_fold"] != t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    C=0.01,
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ).fit(x_10s.loc[train_m], y_mc[train_m])
            proba = clf.predict_proba(x_10s.loc[test_m])
            order = [list(clf.classes_).index(c) for c in classes]
            proba = proba[:, order]
        return pd.DataFrame(proba, index=sub.index[test_m], columns=classes)

    proba_mc = pd.concat(
        Parallel(n_jobs=-1)(delayed(one_mc)(p, t) for p, t in blocks)
    ).loc[sub.index]
    preds_mc = np.array(classes)[np.argmax(proba_mc.to_numpy(), axis=1)]
    cm = confusion_matrix(y_mc, preds_mc, labels=classes, normalize="true")
    mc_per_class = {}
    for c in classes:
        pos_c = y_mc == c
        mc_per_class[c] = {
            "within_person_ovr": _within_person_auc_masks(
                sub, proba_mc[c], pos_c, ~pos_c
            ),
            "same_video_ovr": _same_video_stats(sub, proba_mc[c], pos_c, ~pos_c),
        }
    multiclass_out = {
        "classes": classes,
        "balanced_accuracy": round(float(balanced_accuracy_score(y_mc, preds_mc)), 4),
        "chance_balanced_accuracy": round(1.0 / len(classes), 4),
        "per_class": mc_per_class,
        "confusion_matrix_row_norm": np.round(cm, 4).tolist(),
    }

    return {
        "n_trials_emotion_1": len(sub),
        "n_people_emotion_1": int(sub["participant_id"].nunique()),
        "signatures": signatures,
        "binary_classifiers": binary_out,
        "multiclass_5class": multiclass_out,
        "unsupervised_all": _unsupervised_block(frame),
    }


# ---------------------------------------------------------------------------
# SPLIT 3: FIXED-VIDEO DISTRIBUTION, 2x2 CONSENSUS x STATE, & ON/OFF TESTS
# ---------------------------------------------------------------------------


def fixed_video(frame):
    """Fixed-video class balance, non-adjacent contrasts, 2x2 table, on/off tests."""
    v = frame["rating_valence"]
    a = frame["rating_arousal"]
    stressed = label(frame).to_numpy(bool)
    resps = _base_responses(frame)
    x_10s = timecourse._features(frame, "all", 10)
    model_sc = _cross_fit_binary(frame, x_10s, stressed)

    # 1. Per-video rating distribution & class imbalance audit
    by_vid = (
        frame.assign(stressed=stressed)
        .groupby("video")
        .agg(
            n=("stressed", "size"),
            n_pos=("stressed", "sum"),
            rate=("stressed", "mean"),
            v_mean=("rating_valence", "mean"),
            v_sd=("rating_valence", "std"),
            a_mean=("rating_arousal", "mean"),
            a_sd=("rating_arousal", "std"),
        )
    )
    hist_counts, hist_edges = np.histogram(by_vid["rate"], bins=np.linspace(0, 1, 11))
    polarizing_vids = set(
        by_vid.index[(by_vid["rate"] >= 0.20) & (by_vid["rate"] <= 0.80)]
    )
    balanced_vids = set(
        by_vid.index[(by_vid["rate"] >= 0.25) & (by_vid["rate"] <= 0.75)]
    )
    vid_audit = {
        "n_videos": len(by_vid),
        "rate_quantiles": [
            round(float(x), 4)
            for x in by_vid["rate"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        ],
        "rate_histogram_edges": [round(float(x), 2) for x in hist_edges],
        "rate_histogram_counts": [int(x) for x in hist_counts],
        "n_low_stress_lt10": int((by_vid["rate"] < 0.10).sum()),
        "n_high_stress_gt80": int((by_vid["rate"] > 0.80).sum()),
        "n_polarizing_20_80": len(polarizing_vids),
        "n_balanced_25_75": len(balanced_vids),
        "median_within_video_valence_sd": round(float(by_vid["v_sd"].median()), 3),
        "median_within_video_arousal_sd": round(float(by_vid["a_sd"].median()), 3),
    }

    # 2. Clean non-adjacent & axis-isolated fixed-video contrasts
    oklm_5 = ((v >= 5.0) & (a <= 5.0)).to_numpy(bool)
    oklm_strict = ((v >= 5.5) & (a <= 4.5)).to_numpy(bool)
    fv_contrasts = {
        "all_stress_vs_rest": (stressed, ~stressed, None),
        "polarizing_20_80": (stressed, ~stressed, polarizing_vids),
        "balanced_25_75": (stressed, ~stressed, balanced_vids),
        "stress_vs_oklm_5": (stressed, oklm_5, None),
        "stress_vs_oklm_strict": (stressed, oklm_strict, None),
        "pure_valence_fixed_video": (
            (v <= 3.5).to_numpy(bool),
            (v >= 5.5).to_numpy(bool),
            None,
        ),
        "pure_valence_high_arousal": (
            ((v <= 3.5) & (a >= 5.0)).to_numpy(bool),
            ((v >= 5.5) & (a >= 5.0)).to_numpy(bool),
            None,
        ),
        "pure_arousal_fixed_video": (
            (a >= 6.5).to_numpy(bool),
            (a <= 4.0).to_numpy(bool),
            None,
        ),
        "pure_arousal_neg_valence": (
            ((a >= 6.5) & (v <= 4.5)).to_numpy(bool),
            ((a <= 4.0) & (v <= 4.5)).to_numpy(bool),
            None,
        ),
    }
    scores_map = {
        "model": model_sc,
        **{nm: person.STRESS_SIGN[nm] * rv for nm, rv in resps.items()},
    }
    fv_results = {}
    for cname, (pos, neg, vmask) in fv_contrasts.items():
        fv_results[cname] = {
            sname: _same_video_stats(frame, sc, pos, neg, video_mask=vmask)
            for sname, sc in scores_map.items()
        }

    # 3. Definitive 2x2 / 3x3 Stimulus Consensus x Viewer Subjective State table
    st_f = pd.Series(stressed.astype(float), index=frame.index)
    tot = st_f.groupby(frame["video"]).transform("sum")
    cnt = st_f.groupby(frame["video"]).transform("count")
    cons = ((tot - st_f) / (cnt - 1)).where(cnt - 1 >= person.MIN_OTHERS)

    stim_strata = {
        "low_consensus_le10": (cons <= 0.10).to_numpy(bool),
        "mid_polarizing_20_55": ((cons >= 0.20) & (cons <= 0.55)).to_numpy(bool),
        "high_consensus_ge55": (cons >= 0.55).to_numpy(bool),
        "high_consensus_ge50": (cons >= 0.50).to_numpy(bool),
    }
    state_strata = {
        "oklm_strict": oklm_strict,
        "oklm_5": oklm_5,
        "borderline": (~stressed) & (~oklm_5),
        "stressed": stressed,
    }
    people = frame["participant_id"].to_numpy()
    matrix_2x2 = {}
    for sname, smask in stim_strata.items():
        matrix_2x2[sname] = {}
        for vname, vmask in state_strata.items():
            cell = smask & vmask
            cell_res = {"n_trials": int(cell.sum())}
            for rname, rval in resps.items():
                # Average within each person first, then bootstrap over people
                vals_p = []
                rv = rval.to_numpy(float)
                for p in np.unique(people[cell]):
                    x = rv[(people == p) & cell]
                    x = x[np.isfinite(x)]
                    if len(x) >= 1:
                        vals_p.append(float(x.mean()))
                cell_res[rname] = _boot_ci(vals_p)
            matrix_2x2[sname][vname] = cell_res

    # Direct paired within-person contrasts isolating stimulus vs appraisal
    paired_2x2 = {}
    pairs_def = {
        "stimulus_effect_in_oklm_5": (
            stim_strata["high_consensus_ge50"] & oklm_5,
            stim_strata["low_consensus_le10"] & oklm_5,
        ),
        "stimulus_effect_in_oklm_strict": (
            stim_strata["high_consensus_ge50"] & oklm_strict,
            stim_strata["low_consensus_le10"] & oklm_strict,
        ),
        "appraisal_effect_in_high_consensus_vs_oklm_5": (
            stim_strata["high_consensus_ge50"] & stressed,
            stim_strata["high_consensus_ge50"] & oklm_5,
        ),
        "appraisal_effect_in_mid_polarizing_vs_oklm_5": (
            stim_strata["mid_polarizing_20_55"] & stressed,
            stim_strata["mid_polarizing_20_55"] & oklm_5,
        ),
        "appraisal_effect_in_mid_polarizing_vs_oklm_strict": (
            stim_strata["mid_polarizing_20_55"] & stressed,
            stim_strata["mid_polarizing_20_55"] & oklm_strict,
        ),
    }
    for pname, (m_pos, m_neg) in pairs_def.items():
        paired_2x2[pname] = {
            rname: {
                "diff": _person_diff(rval, m_pos, m_neg, people, min_each=2),
                "d": _person_d(rval, m_pos, m_neg, people, min_each=2),
            }
            for rname, rval in resps.items()
        }

    # 4. Graded proportionality vs On/Off threshold dose-response curves
    dose = {}
    for axis_name, axis_vals, edges in (
        ("valence", v.to_numpy(float), [1, 2, 3, 4, 5, 6, 7, 8, 9.01]),
        ("arousal", a.to_numpy(float), [1, 2, 3, 4, 5, 6, 7, 8, 9.01]),
        (
            "consensus_stress_rate",
            cons.to_numpy(float),
            [0.0, 0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 1.01],
        ),
    ):
        dose[axis_name] = {"edges": edges, "bins": []}
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            bm = np.isfinite(axis_vals) & (axis_vals >= lo) & (axis_vals < hi)
            entry = {"lo": lo, "hi": hi, "n_trials": int(bm.sum())}
            for rname, rval in resps.items():
                # Person-centred response so between-person baseline shifts cancel
                rc = (
                    rval - rval.groupby(frame["participant_id"]).transform("mean")
                ).to_numpy(float)
                pmeans = []
                for p in np.unique(people[bm]):
                    vals = rc[(people == p) & bm]
                    vals = vals[np.isfinite(vals)]
                    if len(vals) >= 1:
                        pmeans.append(float(vals.mean()))
                entry[rname] = _boot_ci(pmeans)
            dose[axis_name]["bins"].append(entry)

    # Within-person linear vs on/off step model test
    functional = {}
    for rname, rval in resps.items():
        functional[rname] = {}
        for xname, xseries, step_mask in (
            ("valence", v, (v <= 4.0).astype(float)),
            ("arousal", a, (a >= 6.0).astype(float)),
            ("consensus_rate", cons, (cons >= 0.40).astype(float)),
        ):
            df_t = pd.DataFrame(
                {
                    "y": rval,
                    "x": xseries,
                    "step": step_mask,
                    "p": frame["participant_id"],
                }
            ).dropna()
            rows_f = []
            for _, g in df_t.groupby("p"):
                if (
                    len(g) < 15
                    or g["y"].std() == 0
                    or g["x"].std() == 0
                    or g["step"].std() == 0
                ):
                    continue
                yz = (g["y"] - g["y"].mean()) / g["y"].std()
                xz = (g["x"] - g["x"].mean()) / g["x"].std()
                sz = (g["step"] - g["step"].mean()) / g["step"].std()
                y_np = yz.to_numpy()
                x_np = xz.to_numpy()
                s_np = sz.to_numpy()
                # R2 of linear, step, and joint (linear + step)
                r2_lin = float(np.corrcoef(x_np, y_np)[0, 1] ** 2)
                r2_step = float(np.corrcoef(s_np, y_np)[0, 1] ** 2)
                des = np.column_stack([np.ones(len(y_np)), x_np, s_np])
                b_joint = np.linalg.lstsq(des, y_np, rcond=None)[0]
                yhat = des @ b_joint
                r2_joint = float(1.0 - np.var(y_np - yhat) / np.var(y_np))
                rows_f.append(
                    {
                        "r2_linear": r2_lin,
                        "r2_step": r2_step,
                        "r2_joint": r2_joint,
                        "delta_r2_step_over_lin": r2_joint - r2_lin,
                        "delta_r2_lin_over_step": r2_joint - r2_step,
                        "beta_lin_joint": float(b_joint[1]),
                        "beta_step_joint": float(b_joint[2]),
                    }
                )
            df_f = pd.DataFrame(rows_f)
            functional[rname][xname] = (
                {c: _boot_ci(df_f[c]) for c in df_f.columns} if len(df_f) else {}
            )

    return {
        "video_distribution": vid_audit,
        "fixed_video_contrasts": fv_results,
        "matrix_2x2": matrix_2x2,
        "paired_2x2_contrasts": paired_2x2,
        "dose_response": dose,
        "linear_vs_onoff": functional,
    }


# ---------------------------------------------------------------------------
# SPLIT 4: EEG TOPOGRAPHY OF STRESS SENSITIVITY (TRIAL-LEVEL & TRAIT-LEVEL)
# ---------------------------------------------------------------------------


def sensitivity(frame):
    """Map regional and scalp EEG correlates of trial and trait sensitivity."""
    # 1. All 6 regions x 5 bands: shared vs own coupling with valence & arousal
    regional_coupling = {}
    for region in common.REGIONS:
        regional_coupling[region] = {}
        for band in common.BANDS:
            sig = f"eeg_{region}_{band}"
            arr = timecourse.corrected(frame, sig)
            # Early 0-2 s for delta/theta, full 0-10 s for all bands
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                win_full = pd.Series(
                    np.nanmean(arr[:, 3:13], axis=1), index=frame.index
                )
                win_early = pd.Series(
                    np.nanmean(arr[:, 3:5], axis=1), index=frame.index
                )
            entry = {
                "full_valence": _safe_mean_ci(
                    person._person_betas(frame, win_full, frame["rating_valence"])
                ),
                "full_arousal": _safe_mean_ci(
                    person._person_betas(frame, win_full, frame["rating_arousal"])
                ),
            }
            if band in ("delta", "theta"):
                entry["early_valence"] = _safe_mean_ci(
                    person._person_betas(frame, win_early, frame["rating_valence"])
                )
                entry["early_arousal"] = _safe_mean_ci(
                    person._person_betas(frame, win_early, frame["rating_arousal"])
                )
            regional_coupling[region][band] = entry

    # 2. Frontal hemispheric asymmetry (right - left frontal, video minus pre)
    faa_out = {}
    people = frame["participant_id"].to_numpy()
    stressed = label(frame).to_numpy(bool)
    for band in common.BANDS:
        l_vid = [
            f"bp_vid_{c}_{band}" for c in LEFT_FRONTAL if f"bp_vid_{c}_{band}" in frame
        ]
        r_vid = [
            f"bp_vid_{c}_{band}" for c in RIGHT_FRONTAL if f"bp_vid_{c}_{band}" in frame
        ]
        l_pre = [
            f"bp_pre_{c}_{band}" for c in LEFT_FRONTAL if f"bp_pre_{c}_{band}" in frame
        ]
        r_pre = [
            f"bp_pre_{c}_{band}" for c in RIGHT_FRONTAL if f"bp_pre_{c}_{band}" in frame
        ]
        if l_vid and r_vid and l_pre and r_pre:
            faa_vid = frame[r_vid].mean(axis=1) - frame[l_vid].mean(axis=1)
            faa_pre = frame[r_pre].mean(axis=1) - frame[l_pre].mean(axis=1)
            faa_chg = faa_vid - faa_pre
            faa_out[band] = {
                "stress_d_change": _person_d(faa_chg, stressed, ~stressed, people),
                "stress_d_video": _person_d(faa_vid, stressed, ~stressed, people),
                "coupling_valence": _safe_mean_ci(
                    person._person_betas(frame, faa_chg, frame["rating_valence"])
                ),
                "coupling_arousal": _safe_mean_ci(
                    person._person_betas(frame, faa_chg, frame["rating_arousal"])
                ),
            }

    # 3. 62-channel scalp topography of beta_own vs beta_shared (delta & alpha)
    scalp_own = {}
    for band in ("delta", "theta", "alpha", "beta"):
        ch_res = {}
        vid_cols = [
            c for c in frame if c.startswith("bp_vid_") and c.endswith(f"_{band}")
        ]
        for vc in vid_cols:
            ch = vc.split("_")[2]
            pc = f"bp_pre_{ch}_{band}"
            if pc not in frame:
                continue
            chg = frame[vc] - frame[pc]
            if chg.notna().mean() < 0.5:
                continue
            bv = person._person_betas(frame, chg, frame["rating_valence"])
            ba = person._person_betas(frame, chg, frame["rating_arousal"])
            ch_res[ch] = {
                "val_shared": round(float(bv["beta_shared"].mean()), 4),
                "val_own": round(float(bv["beta_own"].mean()), 4),
                "aro_shared": round(float(ba["beta_shared"].mean()), 4),
                "aro_own": round(float(ba["beta_own"].mean()), 4),
            }
        scalp_own[band] = ch_res

    # 4. Participant-level TRAIT stress sensitivity (between-person, N=314)
    st_f = pd.Series(stressed.astype(float), index=frame.index)
    tot_s = st_f.groupby(frame["video"]).transform("sum")
    cnt_s = st_f.groupby(frame["video"]).transform("count")
    cons_s = ((tot_s - st_f) / (cnt_s - 1)).where(cnt_s - 1 >= person.MIN_OTHERS)

    val_s = frame["rating_valence"]
    tot_v = val_s.groupby(frame["video"]).transform("sum")
    cnt_v = val_s.notna().groupby(frame["video"]).transform("sum")
    cons_v = ((tot_v - val_s.fillna(0)) / (cnt_v - val_s.notna())).where(
        cnt_v - val_s.notna() >= person.MIN_OTHERS
    )

    aro_s = frame["rating_arousal"]
    tot_a = aro_s.groupby(frame["video"]).transform("sum")
    cnt_a = aro_s.notna().groupby(frame["video"]).transform("sum")
    cons_a = ((tot_a - aro_s.fillna(0)) / (cnt_a - aro_s.notna())).where(
        cnt_a - aro_s.notna() >= person.MIN_OTHERS
    )

    resps = _base_responses(frame)
    # Pre-stimulus baseline regional relative power (resting/anticipatory tone)
    pre_reg = common.regional(frame, kind="bp_pre")
    pre_reg_key = pre_reg.columns.str.rsplit("_", n=1).str[0]
    pre_tot = np.log10((10**pre_reg).T.groupby(pre_reg_key).transform("sum").T)
    pre_rel = pre_reg - pre_tot

    p_rows = []
    for p, g in frame.groupby("participant_id"):
        idx = g.index
        row = {
            "participant_id": p,
            "cohort": float(g["cohort"].iloc[0]),
            "age": float(g["age"].iloc[0]) if "age" in g else np.nan,
            "female": 1.0
            if str(g["gender"].iloc[0]).strip().lower().startswith("f")
            else 0.0,
            "raw_stress_rate": float(st_f.loc[idx].mean()),
            "trait_stress_bias": float((st_f.loc[idx] - cons_s.loc[idx]).mean()),
            "trait_neg_valence": float((cons_v.loc[idx] - val_s.loc[idx]).mean()),
            "trait_high_arousal": float((aro_s.loc[idx] - cons_a.loc[idx]).mean()),
        }
        # Baseline pre-stimulus regional relative EEG power
        for col in pre_rel.columns:
            row[f"pre_{col}"] = float(pre_rel.loc[idx, col].mean())
        # Baseline frontal alpha & theta asymmetry
        for b in ("delta", "theta", "alpha", "beta"):
            lp = [f"bp_pre_{c}_{b}" for c in LEFT_FRONTAL if f"bp_pre_{c}_{b}" in frame]
            rp = [
                f"bp_pre_{c}_{b}" for c in RIGHT_FRONTAL if f"bp_pre_{c}_{b}" in frame
            ]
            if lp and rp:
                row[f"pre_faa_{b}"] = float(
                    (g[rp].mean(axis=1) - g[lp].mean(axis=1)).mean()
                )
        # Baseline resting HR & SC (bins 0..2 = [-3, 0) s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row["pre_hr"] = float(
                np.nanmean(g[["tc_hr_0", "tc_hr_1", "tc_hr_2"]].to_numpy(float))
            )
            row["pre_sc"] = float(
                np.nanmean(g[["tc_sc_0", "tc_sc_1", "tc_sc_2"]].to_numpy(float))
            )
        # Personal physiological stress reactivity (mean stressed - mean non-stressed)
        sm = stressed[frame.index.get_indexer(idx)]
        for rname, rval in resps.items():
            rv = rval.loc[idx].to_numpy(float)
            pos_v, neg_v = rv[sm], rv[~sm]
            pos_v, neg_v = pos_v[np.isfinite(pos_v)], neg_v[np.isfinite(neg_v)]
            row[f"react_{rname}"] = (
                float(pos_v.mean() - neg_v.mean())
                if len(pos_v) >= 3 and len(neg_v) >= 3
                else np.nan
            )
        p_rows.append(row)
    pdf = pd.DataFrame(p_rows)

    def _corr_ci(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[ok], y[ok]
        if len(xv) < 20 or np.std(xv) == 0 or np.std(yv) == 0:
            return None
        rho = float(spearmanr(xv, yv).statistic)
        r = float(np.corrcoef(xv, yv)[0, 1])
        rng = np.random.default_rng(0)
        bidx = rng.integers(0, len(xv), (1000, len(xv)))
        br = [np.corrcoef(xv[i], yv[i])[0, 1] for i in bidx]
        lo, hi = np.percentile(br, [2.5, 97.5])
        return {
            "r": round(r, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "rho": round(rho, 4),
            "n": len(xv),
        }

    # Cohort-demean trait metrics to prevent cohort scale artifacts
    trait_corrs = {}
    predictors = [
        c
        for c in pdf.columns
        if c.startswith(("pre_", "react_")) or c in ("age", "female")
    ]
    for trait in ("trait_stress_bias", "trait_neg_valence", "trait_high_arousal"):
        t_raw = pdf[trait]
        t_adj = (t_raw - t_raw.groupby(pdf["cohort"]).transform("mean")).to_numpy(float)
        trait_corrs[trait] = {}
        for pred in predictors:
            p_raw = pdf[pred]
            p_adj = (p_raw - p_raw.groupby(pdf["cohort"]).transform("mean")).to_numpy(
                float
            )
            trait_corrs[trait][pred] = _corr_ci(t_adj, p_adj)

    return {
        "regional_coupling": regional_coupling,
        "frontal_asymmetry": faa_out,
        "scalp_62ch_coupling": scalp_own,
        "trait_sensitivity_summary": {
            "n_people": len(pdf),
            "trait_stress_bias_sd": round(float(pdf["trait_stress_bias"].std()), 4),
            "raw_stress_rate_q": [
                round(float(x), 4)
                for x in pdf["raw_stress_rate"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            ],
        },
        "trait_correlates_cohort_adjusted": trait_corrs,
    }


# ---------------------------------------------------------------------------
# TRACK B SPLIT 5: ECG P-QRS-T MORPHOLOGY, 4-PHASE EEG SPATIAL/TVAR, OCULAR
# ---------------------------------------------------------------------------


def ecg_morphology(frame):
    """Analyze beat-by-beat PR/QRS/QT/QTc/T-amp, 4-phase EEG spatial/tvar, ocular."""
    people = frame["participant_id"].to_numpy()
    ctrs = contrasts(frame)
    ecg_feats = (
        "hr",
        "rr",
        "rmssd",
        "pr_pk",
        "pr_on",
        "pr_seg",
        "qrs",
        "qt",
        "qtc_baz",
        "qtc_frid",
        "rt_pk",
        "t_amp",
    )

    # 1. Quality & reliability audit of ECG morphology
    audit_ecg = {
        "inverted_lead_fraction": round(float((frame["ecg_polarity"] < 0).mean()), 4),
        "good_qrs_frac_median": round(float(frame["ecg_good_qrs_frac"].median()), 4),
        "features": {},
    }
    for feat in ecg_feats:
        col_v = f"ecg_{feat}_vid"
        col_e = f"ecg_{feat}_early"
        col_l = f"ecg_{feat}_late"
        s = frame[col_v].dropna()
        # Early-vs-late within-person reliability of the feature across trials
        ok_el = frame[[col_e, col_l, "participant_id"]].dropna()
        r_between = (
            float(np.corrcoef(ok_el[col_e], ok_el[col_l])[0, 1])
            if len(ok_el) > 10
            else None
        )
        w_rs = [
            np.corrcoef(g[col_e], g[col_l])[0, 1]
            for _, g in ok_el.groupby("participant_id")
            if len(g) >= 10 and g[col_e].std() > 0 and g[col_l].std() > 0
        ]
        audit_ecg["features"][feat] = {
            "coverage_vid": round(float(frame[col_v].notna().mean()), 4),
            "coverage_early": round(float(frame[col_e].notna().mean()), 4),
            "coverage_late": round(float(frame[col_l].notna().mean()), 4),
            "median_vid": round(float(s.median()), 2) if len(s) else None,
            "iqr_vid": round(float(s.quantile(0.75) - s.quantile(0.25)), 2)
            if len(s)
            else None,
            "early_late_r_overall": round(r_between, 4)
            if r_between is not None
            else None,
            "early_late_r_within_person": _boot_ci(w_rs, draws=1000),
        }

    # 2. Phase-resolved & baseline-corrected ECG P-QRS-T effects across contrasts
    ecg_effects = {}
    for feat in ecg_feats:
        ref = frame[f"ecg_{feat}_ref"]
        series_map = {
            "vid_change": frame[f"ecg_{feat}_vid"] - ref,
            "early_change": frame[f"ecg_{feat}_early"] - ref,
            "late_change": frame[f"ecg_{feat}_late"] - ref,
            "post_change": frame[f"ecg_{feat}_post"] - ref,
            "delta_late_minus_early": frame[f"ecg_{feat}_late"]
            - frame[f"ecg_{feat}_early"],
            "delta_post_minus_late": frame[f"ecg_{feat}_post"]
            - frame[f"ecg_{feat}_late"],
            "vid_level": frame[f"ecg_{feat}_vid"],
        }
        ecg_effects[feat] = {}
        for sname, sval in series_map.items():
            ecg_effects[feat][sname] = {
                cname: _person_d(sval, pos, neg, people)
                for cname, (pos, neg) in ctrs.items()
                if cname
                in ("stress", "vs_excitement", "vs_calm_negative", "vs_relaxed")
            }
            # Also raw ms/bpm difference for primary stress contrast
            pos, neg = ctrs["stress"]
            ecg_effects[feat][sname]["stress_raw_diff"] = _person_diff(
                sval, pos, neg, people
            )

    # 3. Cardiology check: RR-controlled PR/QT effects & shared-vs-own coupling
    rr_ctrl = {}
    stressed = label(frame).to_numpy(bool)
    for feat in (
        "pr_pk",
        "pr_on",
        "pr_seg",
        "qrs",
        "qt",
        "qtc_baz",
        "qtc_frid",
        "rt_pk",
        "t_amp",
    ):
        y_chg = frame[f"ecg_{feat}_vid"] - frame[f"ecg_{feat}_ref"]
        rr_chg = frame["ecg_rr_vid"] - frame["ecg_rr_ref"]
        df_c = pd.DataFrame(
            {"y": y_chg, "rr": rr_chg, "s": stressed.astype(float), "p": people}
        ).dropna()
        rows_c = []
        for _, g in df_c.groupby("p"):
            if g["s"].sum() < MIN_EACH or (1 - g["s"]).sum() < MIN_EACH:
                continue
            if g["y"].std() == 0 or g["rr"].std() == 0:
                continue
            yz = (g["y"] - g["y"].mean()) / g["y"].std()
            rz = (g["rr"] - g["rr"].mean()) / g["rr"].std()
            s_np = g["s"].to_numpy()
            plain = np.column_stack([np.ones(len(g)), s_np])
            full = np.column_stack([plain, rz.to_numpy()])
            rows_c.append(
                {
                    "beta_plain": float(np.linalg.lstsq(plain, yz, rcond=None)[0][1]),
                    "beta_rr_controlled": float(
                        np.linalg.lstsq(full, yz, rcond=None)[0][1]
                    ),
                }
            )
        dfc = pd.DataFrame(rows_c)
        rr_ctrl[feat] = {
            "plain": _boot_ci(dfc["beta_plain"]) if len(dfc) else None,
            "rr_controlled": _boot_ci(dfc["beta_rr_controlled"]) if len(dfc) else None,
            "coupling_valence": _safe_mean_ci(
                person._person_betas(frame, y_chg, frame["rating_valence"])
            ),
            "coupling_arousal": _safe_mean_ci(
                person._person_betas(frame, y_chg, frame["rating_arousal"])
            ),
        }

    # 4. 4-phase EEG spatial homogeneity, asymmetries, and temporal power variation
    eeg_rich = {}
    for family in (
        "sp_std",
        "sp_iqr",
        "sp_ent",
        "reg_std",
        "ap_grad",
        "faa",
        "temp_asym",
    ):
        eeg_rich[family] = {}
        for band in common.BANDS:
            ref = frame[f"eeg_{family}_ref_{band}"]
            eeg_rich[family][band] = {
                ph: {
                    cname: _person_d(
                        frame[f"eeg_{family}_{ph}_{band}"] - ref, pos, neg, people
                    )
                    for cname, (pos, neg) in ctrs.items()
                    if cname in ("stress", "vs_excitement", "vs_calm_negative")
                }
                for ph in ("early", "late", "post")
            }
            # Also shared vs own coupling for early & late
            chg_el = (
                0.5
                * (
                    frame[f"eeg_{family}_early_{band}"]
                    + frame[f"eeg_{family}_late_{band}"]
                )
                - ref
            )
            eeg_rich[family][band]["coupling_arousal"] = _safe_mean_ci(
                person._person_betas(frame, chg_el, frame["rating_arousal"])
            )
            eeg_rich[family][band]["coupling_valence"] = _safe_mean_ci(
                person._person_betas(frame, chg_el, frame["rating_valence"])
            )

    tvar_out = {}
    for band in common.BANDS:
        tvar_out[band] = {}
        for stat in (
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
            col = f"eeg_{stat}_{band}"
            if col not in frame:
                continue
            vals = frame[col]
            if stat.startswith("tvar_") and f"eeg_tvar_ref_{band}" in frame:
                vals = vals - frame[f"eeg_tvar_ref_{band}"]
            tvar_out[band][stat] = {
                cname: _person_d(vals, pos, neg, people)
                for cname, (pos, neg) in ctrs.items()
                if cname in ("stress", "vs_excitement", "vs_calm_negative")
            }

    # 5. Pre-ICA frontopolar ocular/blink proxy control on early delta!
    delta_0_2 = person.response(frame, "delta")
    ocular_ctrl = {}
    if "raw_fp_ptp_0_2" in frame and "raw_fp_ptp_ref" in frame:
        fp_chg = np.log10(frame["raw_fp_ptp_0_2"].clip(lower=1e-3)) - np.log10(
            frame["raw_fp_ptp_ref"].clip(lower=1e-3)
        )
        # Does blink proxy itself differ in stress, and does delta's beta_own survive?
        ocular_ctrl["fp_ptp_0_2_stress_d"] = _person_d(
            fp_chg, stressed, ~stressed, people
        )
        ocular_ctrl["fp_ptp_coupling_arousal"] = _safe_mean_ci(
            person._person_betas(frame, fp_chg, frame["rating_arousal"])
        )
        # Regress delta_0_2 on shared + own arousal WITH fp_chg as covariate
        y_dl, _, _ = person.components(frame, delta_0_2)
        _, sh_a, own_a = person.components(frame, frame["rating_arousal"])
        df_o = pd.DataFrame(
            {"y": y_dl, "sh": sh_a, "own": own_a, "fp": fp_chg, "p": people}
        ).dropna()
        rows_o = []
        for _, g in df_o.groupby("p"):
            if len(g) < MIN_TRIALS:
                continue
            z = g[["y", "sh", "own", "fp"]].apply(person._z)
            if z.isna().any().any():
                continue
            d_plain = np.column_stack([np.ones(len(z)), z["sh"], z["own"]])
            d_ctrl = np.column_stack([d_plain, z["fp"]])
            bp = np.linalg.lstsq(d_plain, z["y"].to_numpy(), rcond=None)[0]
            bc = np.linalg.lstsq(d_ctrl, z["y"].to_numpy(), rcond=None)[0]
            rows_o.append(
                {
                    "beta_own_plain": float(bp[2]),
                    "beta_own_ocular_controlled": float(bc[2]),
                    "beta_fp_on_delta": float(bc[3]),
                }
            )
        dfo = pd.DataFrame(rows_o)
        ocular_ctrl["delta_arousal_own_controlled"] = (
            {c: _boot_ci(dfo[c]) for c in dfo.columns} if len(dfo) else {}
        )

    return {
        "ecg_audit": audit_ecg,
        "ecg_effects": ecg_effects,
        "ecg_rr_controlled_and_coupling": rr_ctrl,
        "eeg_spatial_homogeneity_and_asymmetry": eeg_rich,
        "eeg_temporal_variation": tvar_out,
        "ocular_proxy_control": ocular_ctrl,
    }


# ---------------------------------------------------------------------------
# TRACK B SPLIT 6: RAW CURATED vs ICA PREPROCESSED HEP CONTROL
# ---------------------------------------------------------------------------


def _hep_window_amp(frame, prefix, regions=("frontal", "central")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vals = np.nanmean(
            [person._curve(frame, f"{prefix}_{r}") for r in regions], axis=0
        )
    return pd.Series(vals[:, person.HEP_BINS].mean(axis=1), index=frame.index)


def hep_raw(frame):
    """Direct test of whether ICA preprocessing removes brain-heart stress HEP."""
    people = frame["participant_id"].to_numpy()
    ctrs = contrasts(frame)
    stressed = label(frame)

    variants = {
        "ica_preprocessed_100uV": _hep_window_amp(frame, "hep_video"),
        "raw_curated_100uV": _hep_window_amp(frame, "hep_raw_video"),
        "raw_curated_200uV": _hep_window_amp(frame, "hep_raw200_video"),
        "ica_removed_diff_100uV": _hep_window_amp(frame, "hep_raw_video")
        - _hep_window_amp(frame, "hep_video"),
        "raw_curated_change_vs_pre": _hep_window_amp(frame, "hep_raw_video")
        - _hep_window_amp(frame, "hep_raw_pre"),
    }

    summary = {
        "coverage_ica_100uV": round(
            float(variants["ica_preprocessed_100uV"].notna().mean()), 4
        ),
        "coverage_raw_100uV": round(
            float(variants["raw_curated_100uV"].notna().mean()), 4
        ),
        "coverage_raw_200uV": round(
            float(variants["raw_curated_200uV"].notna().mean()), 4
        ),
        "median_beats_ica": round(float(frame["beats_video"].median()), 1),
        "median_beats_raw_100uV": round(
            float(frame["hep_raw_beats_video"].median()), 1
        ),
        "median_beats_raw_200uV": round(
            float(frame["hep_raw200_beats_video"].median()), 1
        ),
        "variants": {},
    }

    for vname, series in variants.items():
        summary["variants"][vname] = {
            "contrasts_d": {
                cname: _person_d(series, pos, neg, people)
                for cname, (pos, neg) in ctrs.items()
                if cname
                in ("stress", "vs_excitement", "vs_calm_negative", "vs_relaxed")
            },
            "ecg_rr_controlled": person._controlled(frame, series, stressed)
            if frame["rr_video"].std() > 0
            else {"people": 0},
            "coupling_valence": _safe_mean_ci(
                person._person_betas(frame, series, frame["rating_valence"])
            ),
            "coupling_arousal": _safe_mean_ci(
                person._person_betas(frame, series, frame["rating_arousal"])
            ),
        }

    # Regional 50-ms bin curves for RAW CURATED, ICA PREPROCESSED, and ICA-REMOVED DIFF
    pos, neg = (m.to_numpy(bool) for m in ctrs["stress"])
    curves = {}
    for region in common.REGIONS:
        raw_c = person._curve(frame, f"hep_raw_video_{region}")
        ica_c = person._curve(frame, f"hep_video_{region}")
        diff_c = raw_c - ica_c
        for tag, arr in (
            (f"raw_{region}", raw_c),
            (f"ica_{region}", ica_c),
            (f"icadiff_{region}", diff_c),
        ):
            d, _ = timecourse._person_d(arr, pos, neg, people)
            mean, lo, hi = timecourse._summary(d)
            curves[tag] = {
                "d": mean.round(4).tolist(),
                "low": lo.round(4).tolist(),
                "high": hi.round(4).tolist(),
                "people": len(d),
            }
    summary["curves"] = curves

    # Within-person correlation of raw HEP & ICA-removed component with ECG field
    ecg_amp = pd.Series(
        person._curve(frame, "ecg_video")[:, person.HEP_BINS].mean(axis=1),
        index=frame.index,
    )
    cfa_check = {}
    for tag, series in (
        ("ica_100uV", variants["ica_preprocessed_100uV"]),
        ("raw_100uV", variants["raw_curated_100uV"]),
        ("icadiff_100uV", variants["ica_removed_diff_100uV"]),
    ):
        rs = []
        df_r = pd.DataFrame({"h": series, "e": ecg_amp, "p": people}).dropna()
        for _, g in df_r.groupby("p"):
            if len(g) >= MIN_TRIALS and g["h"].std() > 0 and g["e"].std() > 0:
                rs.append(np.corrcoef(g["h"], g["e"])[0, 1])
        cfa_check[tag] = _boot_ci(rs, draws=1000)
    summary["ecg_field_correlation_250_600ms"] = cfa_check
    return summary


# ---------------------------------------------------------------------------
# TRACK B SPLIT 7: 4-PHASE, DELTAS, ENRICHED & SEQUENCE-AWARE MODEL BENCHMARK
# ---------------------------------------------------------------------------


def _person_centre(x, people):
    return x - x.groupby(people).transform("mean")


def _build_phase_features(
    frame,
    *,
    add_deltas=False,
    add_ecg_morph=False,
    add_eeg_spatial=False,
    add_eeg_tvar=False,
):
    """Assemble 4-phase physiological features with explicit t->t+1 transitions."""
    cols = {}

    def _add_phase_group(prefix, names):
        for nm in names:
            ref = frame[f"{prefix}_{nm}_ref"]
            early = frame[f"{prefix}_{nm}_early"]
            late = frame[f"{prefix}_{nm}_late"]
            post = frame[f"{prefix}_{nm}_post"]
            # Baseline-referenced levels per phase (ref = 0 by construction)
            cols[f"{prefix}_{nm}_e_chg"] = early - ref
            cols[f"{prefix}_{nm}_l_chg"] = late - ref
            cols[f"{prefix}_{nm}_p_chg"] = post - ref
            if add_deltas:
                # Explicit t -> t+1 step transitions (0 at ref)
                cols[f"{prefix}_{nm}_d_l_e"] = late - early
                cols[f"{prefix}_{nm}_d_p_l"] = post - late

    # Core 4-phase body (HR, SC, Resp)
    _add_phase_group("ecg", ("hr", "rr"))
    _add_phase_group("sc", ("mean", "rise", "slope"))
    _add_phase_group("resp", ("amp", "rate", "slope"))

    # Core 4-phase EEG regional power
    for r in common.REGIONS:
        for b in common.BANDS:
            ref = frame[f"eeg_ep_ref_{r}_{b}"]
            e = frame[f"eeg_ep_early_{r}_{b}"]
            l_ = frame[f"eeg_ep_late_{r}_{b}"]
            p_ = frame[f"eeg_ep_post_{r}_{b}"]
            cols[f"eeg_{r}_{b}_e_chg"] = e - ref
            cols[f"eeg_{r}_{b}_l_chg"] = l_ - ref
            cols[f"eeg_{r}_{b}_p_chg"] = p_ - ref
            if add_deltas:
                cols[f"eeg_{r}_{b}_d_l_e"] = l_ - e
                cols[f"eeg_{r}_{b}_d_p_l"] = p_ - l_

    if add_ecg_morph:
        _add_phase_group(
            "ecg",
            (
                "pr_pk",
                "pr_on",
                "pr_seg",
                "qrs",
                "qt",
                "qtc_baz",
                "qtc_frid",
                "rt_pk",
                "t_amp",
            ),
        )
        for m in ("pr_pk", "pr_on", "pr_seg", "qrs", "qt", "qtc_baz", "rt_pk", "t_amp"):
            cols[f"ecg_{m}_vid_chg"] = frame[f"ecg_{m}_vid"] - frame[f"ecg_{m}_ref"]

    if add_eeg_spatial:
        for fam in (
            "sp_std",
            "sp_iqr",
            "sp_ent",
            "reg_std",
            "ap_grad",
            "faa",
            "temp_asym",
        ):
            for b in common.BANDS:
                ref = frame[f"eeg_{fam}_ref_{b}"]
                e = frame[f"eeg_{fam}_early_{b}"]
                l_ = frame[f"eeg_{fam}_late_{b}"]
                p_ = frame[f"eeg_{fam}_post_{b}"]
                cols[f"eeg_{fam}_{b}_e_chg"] = e - ref
                cols[f"eeg_{fam}_{b}_l_chg"] = l_ - ref
                cols[f"eeg_{fam}_{b}_p_chg"] = p_ - ref
                if add_deltas:
                    cols[f"eeg_{fam}_{b}_d_l_e"] = l_ - e
                    cols[f"eeg_{fam}_{b}_d_p_l"] = p_ - l_

    if add_eeg_tvar:
        for b in common.BANDS:
            tref = frame[f"eeg_tvar_ref_{b}"]
            cols[f"eeg_tvar_e_{b}"] = frame[f"eeg_tvar_early_{b}"] - tref
            cols[f"eeg_tvar_l_{b}"] = frame[f"eeg_tvar_late_{b}"] - tref
            cols[f"eeg_tslope_e_{b}"] = frame[f"eeg_tslope_early_{b}"]
            cols[f"eeg_tslope_l_{b}"] = frame[f"eeg_tslope_late_{b}"]
            cols[f"eeg_maxjump_e_{b}"] = frame[f"eeg_maxjump_early_{b}"]
            cols[f"eeg_maxjump_l_{b}"] = frame[f"eeg_maxjump_late_{b}"]

    x = pd.DataFrame(cols, index=frame.index)
    return _person_centre(x, frame["participant_id"].to_numpy())


def _build_multiscale_temporal_features(frame):
    """Explicit temporal-structure representation over the 15-s multivariate series.

    Applies causal multi-scale temporal operators (first differences t->t+1,
    second differences / acceleration, exponentially weighted moving summaries,
    and fixed multi-scale 1D convolutional kernels of widths 2, 3, 5 s with
    max/mean/positive-proportion pooling — MiniROCKET style) so models explicitly
    see sequential dependencies between t and t+1.
    """
    signals = ("hr", "sc", "resp_amp", "resp_rate", *timecourse.EEG)
    # Build (N, 12, C) post-onset + 1s pre baseline-corrected sequence [t = 0..11]
    mats = []
    for sig in signals:
        arr = timecourse.corrected(frame, sig)[:, 3:15]  # 0..11 s (12 bins)
        mats.append(arr)
    seq = np.stack(mats, axis=-1)  # (N, 12, C)
    # Replace NaNs per signal-time with 0 (neutral baseline change) before conv
    seq_clean = np.nan_to_num(seq, nan=0.0)
    # First-order temporal differences (t+1 - t) and second-order acceleration
    d1 = np.diff(seq_clean, axis=1)  # (N, 11, C)
    d2 = np.diff(d1, axis=1)  # (N, 10, C)

    feats = {}
    for c_i, sig in enumerate(signals):
        s = seq_clean[:, :, c_i]
        v1 = d1[:, :, c_i]
        v2 = d2[:, :, c_i]
        # Temporal trajectory summaries capturing sequential shape
        feats[f"seq_{sig}_early_mean"] = s[:, :5].mean(axis=1)
        feats[f"seq_{sig}_late_mean"] = s[:, 5:10].mean(axis=1)
        feats[f"seq_{sig}_post_mean"] = s[:, 10:12].mean(axis=1)
        feats[f"seq_{sig}_peak_max"] = s[:, :10].max(axis=1)
        feats[f"seq_{sig}_peak_min"] = s[:, :10].min(axis=1)
        feats[f"seq_{sig}_argmax"] = np.argmax(s[:, :10], axis=1).astype(float)
        feats[f"seq_{sig}_argmin"] = np.argmin(s[:, :10], axis=1).astype(float)
        feats[f"seq_{sig}_d1_early_mean"] = v1[:, :4].mean(axis=1)
        feats[f"seq_{sig}_d1_late_mean"] = v1[:, 4:9].mean(axis=1)
        feats[f"seq_{sig}_d1_max"] = v1[:, :9].max(axis=1)
        feats[f"seq_{sig}_d1_min"] = v1[:, :9].min(axis=1)
        feats[f"seq_{sig}_d1_pos_frac"] = (v1[:, :9] > 0).mean(axis=1)
        feats[f"seq_{sig}_d2_energy"] = np.mean(v2[:, :8] ** 2, axis=1)
        # Lag-1 autocorrelation proxy (sequential persistence)
        s0, s1 = s[:, :9], s[:, 1:10]
        feats[f"seq_{sig}_lag1_prod"] = np.mean(s0 * s1, axis=1)

    # Random multi-scale 1D temporal convolutional kernels (deterministic seed)
    rng = np.random.default_rng(20261123)
    n_channels = seq_clean.shape[2]
    for k_i in range(48):
        width = int(rng.choice([2, 3, 4, 5]))
        # Sparse channel weights + zero-mean temporal filter
        ch_sel = rng.choice(n_channels, size=4, replace=False)
        w_t = rng.normal(size=(width, 4))
        w_t -= w_t.mean(axis=0, keepdims=True)
        # 1D convolution across the 12 time steps
        out_len = 12 - width + 1
        conv = np.zeros((len(seq_clean), out_len), dtype=float)
        for u in range(out_len):
            conv[:, u] = (
                seq_clean[:, u : u + width, :][:, :, ch_sel] * w_t[None, :, :]
            ).sum(axis=(1, 2))
        feats[f"tconv_{k_i}_max"] = conv.max(axis=1)
        feats[f"tconv_{k_i}_ppv"] = (conv > 0).mean(axis=1)
        feats[f"tconv_{k_i}_mean"] = conv.mean(axis=1)

    x = pd.DataFrame(feats, index=frame.index)
    return _person_centre(x, frame["participant_id"].to_numpy())


def _eval_model_spec(frame, x, builder_kind="logreg"):
    """Evaluate a feature set + model family on dev 3x3 CV and sealed test regimes."""
    dev = frame[frame["split"] == "dev"]
    test = frame[frame["split"] != "dev"]
    y_all = label(frame).to_numpy(int)

    def fit_score(train_idx, eval_idx):
        xtr, xev = x.loc[train_idx], x.loc[eval_idx]
        ytr = y_all[frame.index.get_indexer(train_idx)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if builder_kind == "logreg":
                clf = _balanced_model().fit(xtr, ytr)
                sc = clf.decision_function(xev)
            elif builder_kind == "ridge_va":
                people_tr = frame.loc[train_idx, "participant_id"]
                yv = frame.loc[train_idx, "rating_valence"]
                yv = yv - yv.groupby(people_tr).transform("mean")
                ya = frame.loc[train_idx, "rating_arousal"]
                ya = ya - ya.groupby(people_tr).transform("mean")
                pipe_v = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    RidgeCV(alphas=np.logspace(0, 4, 9)),
                ).fit(xtr, yv)
                pipe_a = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    RidgeCV(alphas=np.logspace(0, 4, 9)),
                ).fit(xtr, ya)
                pv, pa = pipe_v.predict(xev), pipe_a.predict(xev)
                sc = pa / max(pa.std(), 1e-6) - pv / max(pv.std(), 1e-6)
            elif builder_kind == "gbdt":
                clf = HistGradientBoostingClassifier(
                    max_iter=120,
                    max_depth=4,
                    learning_rate=0.05,
                    min_samples_leaf=40,
                    l2_regularization=10.0,
                    class_weight="balanced",
                    random_state=0,
                ).fit(xtr, ytr)
                sc = clf.predict_proba(xev)[:, 1]
            elif builder_kind == "mlp":
                clf = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=3.0,
                        max_iter=80,
                        early_stopping=True,
                        random_state=0,
                    ),
                ).fit(xtr, ytr)
                prob = np.clip(clf.predict_proba(xev)[:, 1], 1e-6, 1.0 - 1e-6)
                pi = np.clip(float(ytr.mean()), 1e-6, 1.0 - 1e-6)
                sc = np.log(prob / (1.0 - prob)) - np.log(pi / (1.0 - pi))
            else:
                raise ValueError(builder_kind)
        return pd.Series(sc, index=eval_idx)

    # 3x3 person x theme dev cross-fit
    blocks = [(p, t) for p in range(3) for t in range(3)]

    def one_dev(p, t):
        ev = dev[(dev["person_fold"] == p) & (dev["theme_fold"] == t)].index
        tr = dev[(dev["person_fold"] != p) & (dev["theme_fold"] != t)].index
        return fit_score(tr, ev)

    dev_scores = pd.concat(
        Parallel(n_jobs=-1)(delayed(one_dev)(p, t) for p, t in blocks)
    ).loc[dev.index]
    test_scores = fit_score(dev.index, test.index)

    def summarize(sub, sc):
        ct = contrasts(sub)
        return {
            "within_person": _within_person_auc_masks(
                sub, sc, ct["stress"][0], ct["stress"][1]
            ),
            "vs_excitement": _within_person_auc_masks(
                sub, sc, ct["vs_excitement"][0], ct["vs_excitement"][1]
            ),
            "vs_calm_negative": _within_person_auc_masks(
                sub, sc, ct["vs_calm_negative"][0], ct["vs_calm_negative"][1]
            ),
            "vs_relaxed": _within_person_auc_masks(
                sub, sc, ct["vs_relaxed"][0], ct["vs_relaxed"][1]
            ),
            "same_video": _same_video_stats(sub, sc, ct["stress"][0], ct["stress"][1]),
        }

    out = {"n_features": int(x.shape[1]), "dev": summarize(dev, dev_scores)}
    for regime in common.REGIMES:
        m = test["split"] == regime
        out[regime] = summarize(test[m], test_scores[m])
    return out


def sequence_models(frame):
    """Benchmark 4-phase, delta, ECG, EEG spatial, and sequence models."""
    x_1s = timecourse._features(frame, "all", 10)
    x_p4 = _build_phase_features(frame)
    x_p4_delta = _build_phase_features(frame, add_deltas=True)
    x_p4_ecg = _build_phase_features(frame, add_deltas=True, add_ecg_morph=True)
    x_p4_spatial = _build_phase_features(frame, add_deltas=True, add_eeg_spatial=True)
    x_p4_tvar = _build_phase_features(frame, add_deltas=True, add_eeg_tvar=True)
    x_full = _build_phase_features(
        frame,
        add_deltas=True,
        add_ecg_morph=True,
        add_eeg_spatial=True,
        add_eeg_tvar=True,
    )
    x_seq = _build_multiscale_temporal_features(frame)
    x_seq_full = pd.concat([x_full, x_seq], axis=1)

    specs = (
        ("baseline_1s_logreg", x_1s, "logreg"),
        ("phase4_logreg", x_p4, "logreg"),
        ("phase4_deltas_logreg", x_p4_delta, "logreg"),
        ("ablation_plus_ecg_pqrst", x_p4_ecg, "logreg"),
        ("ablation_plus_eeg_spatial", x_p4_spatial, "logreg"),
        ("ablation_plus_eeg_tvar", x_p4_tvar, "logreg"),
        ("enriched_full_logreg", x_full, "logreg"),
        ("enriched_full_ridge_va", x_full, "ridge_va"),
        ("enriched_full_gbdt", x_full, "gbdt"),
        ("sequence_multiscale_conv_logreg", x_seq, "logreg"),
        ("sequence_plus_enriched_logreg", x_seq_full, "logreg"),
        ("sequence_plus_enriched_mlp", x_seq_full, "mlp"),
    )
    models_out = {}
    for name, xmat, kind in specs:
        models_out[name] = _eval_model_spec(frame, xmat, kind)
    return {
        "splits": common.summary(frame),
        "models": models_out,
    }


def _residualize(y, covs):
    """OLS residuals of 1D vector y on 2D covariate matrix covs (handles NaNs)."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(covs, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    ok = np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1)
    out = np.full(len(y_arr), np.nan, dtype=float)
    if ok.sum() < x_arr.shape[1] + 5:
        return out
    design = np.column_stack([np.ones(ok.sum()), x_arr[ok]])
    beta, *_ = np.linalg.lstsq(design, y_arr[ok], rcond=None)
    out[ok] = y_arr[ok] - design @ beta
    return out


def _corr_ci_p(x, y, draws=None):
    """Pearson r, bootstrap 95% CI, exact two-sided p, and Spearman rho."""
    draws = DRAWS if draws is None else draws
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    if len(xv) < 20 or np.std(xv) == 0 or np.std(yv) == 0:
        return None
    r, pval = pearsonr(xv, yv)
    rho = float(spearmanr(xv, yv).statistic)
    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(xv), (draws, len(xv)))
    br = [np.corrcoef(xv[i], yv[i])[0, 1] for i in bidx]
    lo, hi = np.percentile(br, [2.5, 97.5])
    return {
        "r": round(float(r), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p": round(float(pval), 6),
        "rho": round(rho, 4),
        "n": int(len(xv)),
    }


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR q-values for a dict of {key: entry_with_p}."""
    keys = [k for k, v in pvals.items() if isinstance(v, dict) and "p" in v]
    if not keys:
        return pvals
    ps = np.array([pvals[k]["p"] for k in keys], dtype=float)
    order = np.argsort(ps)
    n = len(ps)
    q_ord = np.empty(n, dtype=float)
    running = 1.0
    for rank_rev in range(n - 1, -1, -1):
        idx = order[rank_rev]
        val = ps[idx] * n / (rank_rev + 1)
        running = min(running, val)
        q_ord[rank_rev] = running
    for pos, idx in enumerate(order):
        pvals[keys[idx]]["q_fdr"] = round(float(min(1.0, q_ord[pos])), 5)
    return pvals


INTERIOR_CH = (
    "Cz",
    "FCz",
    "CPz",
    "C1",
    "C2",
    "C3",
    "C4",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "Pz",
)
RIM_CH = (
    "Fp1",
    "Fp2",
    "AF7",
    "AF8",
    "F7",
    "F8",
    "FT7",
    "FT8",
    "T7",
    "T8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "O1",
    "O2",
)


def _group_rel_power(frame, channels, kind="bp_pre"):
    """Relative log10 band power for an arbitrary channel group."""
    abs_cols = {}
    for band in common.BANDS:
        cols = [
            f"{kind}_{ch}_{band}" for ch in channels if f"{kind}_{ch}_{band}" in frame
        ]
        if cols:
            abs_cols[band] = frame[cols].mean(axis=1)
    if not abs_cols:
        return pd.DataFrame(index=frame.index), pd.DataFrame(index=frame.index)
    abs_df = pd.DataFrame(abs_cols, index=frame.index)
    tot = np.log10((10.0**abs_df).sum(axis=1))
    rel_df = abs_df.sub(tot, axis=0)
    return rel_df, abs_df


def sensitivity_controls(frame):
    """Strict confound, micro-EMG, sex-stratified, and cross-day trait controls."""
    stressed = label(frame).to_numpy(bool)
    st_f = pd.Series(stressed.astype(float), index=frame.index)
    tot_s = st_f.groupby(frame["video"]).transform("sum")
    cnt_s = st_f.groupby(frame["video"]).transform("count")
    cons_s = ((tot_s - st_f) / (cnt_s - 1)).where(cnt_s - 1 >= person.MIN_OTHERS)

    val_s = frame["rating_valence"]
    tot_v = val_s.groupby(frame["video"]).transform("sum")
    cnt_v = val_s.notna().groupby(frame["video"]).transform("sum")
    cons_v = ((tot_v - val_s.fillna(0)) / (cnt_v - val_s.notna())).where(
        cnt_v - val_s.notna() >= person.MIN_OTHERS
    )

    aro_s = frame["rating_arousal"]
    tot_a = aro_s.groupby(frame["video"]).transform("sum")
    cnt_a = aro_s.notna().groupby(frame["video"]).transform("sum")
    cons_a = ((tot_a - aro_s.fillna(0)) / (cnt_a - aro_s.notna())).where(
        cnt_a - aro_s.notna() >= person.MIN_OTHERS
    )

    resps = _base_responses(frame)
    pre_reg = common.regional(frame, kind="bp_pre")
    pre_reg_key = pre_reg.columns.str.rsplit("_", n=1).str[0]
    pre_tot = np.log10((10**pre_reg).T.groupby(pre_reg_key).transform("sum").T)
    pre_rel = pre_reg - pre_tot

    int_rel, int_abs = _group_rel_power(frame, INTERIOR_CH, "bp_pre")
    rim_rel, rim_abs = _group_rel_power(frame, RIM_CH, "bp_pre")

    def _agg_subset(df_sub, p_id):
        idx = df_sub.index
        row = {
            "participant_id": p_id,
            "cohort": float(df_sub["cohort"].iloc[0]),
            "age": float(df_sub["age"].iloc[0]) if "age" in df_sub else np.nan,
            "female": 1.0
            if str(df_sub["gender"].iloc[0]).strip().lower().startswith("f")
            else 0.0,
            "n_trials": int(len(df_sub)),
            "trait_stress_bias": float((st_f.loc[idx] - cons_s.loc[idx]).mean()),
            "trait_neg_valence": float((cons_v.loc[idx] - val_s.loc[idx]).mean()),
            "trait_high_arousal": float((aro_s.loc[idx] - cons_a.loc[idx]).mean()),
        }
        for col in pre_rel.columns:
            row[f"pre_{col}"] = float(pre_rel.loc[idx, col].mean())
            row[f"abs_{col}"] = float(pre_reg.loc[idx, col].mean())
        for b in common.BANDS:
            if b in int_rel:
                row[f"pre_interior_{b}"] = float(int_rel.loc[idx, b].mean())
                row[f"abs_interior_{b}"] = float(int_abs.loc[idx, b].mean())
            if b in rim_rel:
                row[f"pre_rim_{b}"] = float(rim_rel.loc[idx, b].mean())
                row[f"abs_rim_{b}"] = float(rim_abs.loc[idx, b].mean())
            if b in int_rel and b in rim_rel:
                row[f"pre_rim_minus_int_{b}"] = (
                    row[f"pre_rim_{b}"] - row[f"pre_interior_{b}"]
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row["pre_hr"] = float(
                np.nanmean(df_sub[["tc_hr_0", "tc_hr_1", "tc_hr_2"]].to_numpy(float))
            )
            row["pre_sc"] = float(
                np.nanmean(df_sub[["tc_sc_0", "tc_sc_1", "tc_sc_2"]].to_numpy(float))
            )
            if "raw_fp_ptp_ref" in df_sub:
                row["raw_fp_ptp_ref"] = float(
                    np.nanmean(df_sub["raw_fp_ptp_ref"].to_numpy(float))
                )
        sm = stressed[frame.index.get_indexer(idx)]
        for rname, rval in resps.items():
            rv = rval.loc[idx].to_numpy(float)
            pos_v, neg_v = rv[sm], rv[~sm]
            pos_v, neg_v = pos_v[np.isfinite(pos_v)], neg_v[np.isfinite(neg_v)]
            row[f"react_{rname}"] = (
                float(pos_v.mean() - neg_v.mean())
                if len(pos_v) >= 3 and len(neg_v) >= 3
                else np.nan
            )
        return row

    pdf = pd.DataFrame([_agg_subset(g, p) for p, g in frame.groupby("participant_id")])

    # Build cohort one-hot + age + female design matrix
    cohorts = sorted(pdf["cohort"].dropna().unique())
    coh_cols = [(pdf["cohort"] == c).astype(float).to_numpy() for c in cohorts[1:]]
    cov_cohort = np.column_stack(coh_cols) if coh_cols else np.zeros((len(pdf), 0))
    cov_full = np.column_stack(
        [*coh_cols, pdf["age"].to_numpy(float), pdf["female"].to_numpy(float)]
    )

    # 1. Partial correlations adjusted for cohort + age + female (+ FDR)
    key_preds = [
        "pre_hr",
        "pre_sc",
        "raw_fp_ptp_ref",
        "pre_interior_gamma",
        "pre_rim_gamma",
        "pre_rim_minus_int_gamma",
        "abs_interior_gamma",
        "abs_rim_gamma",
        *[f"pre_{c}" for c in pre_rel.columns],
        *[f"abs_{c}" for c in pre_reg.columns if c.endswith(("_gamma", "_theta"))],
        *[f"react_{r}" for r in resps],
    ]
    key_preds = [c for c in key_preds if c in pdf]

    adj_full = {}
    adj_cohort = {}
    for trait in ("trait_stress_bias", "trait_high_arousal", "trait_neg_valence"):
        y_coh = _residualize(pdf[trait], cov_cohort)
        y_full = _residualize(pdf[trait], cov_full)
        c_res, f_res = {}, {}
        for pred in key_preds:
            c_res[pred] = _corr_ci_p(y_coh, _residualize(pdf[pred], cov_cohort))
            f_res[pred] = _corr_ci_p(y_full, _residualize(pdf[pred], cov_full))
        adj_cohort[trait] = _bh_fdr(c_res)
        adj_full[trait] = _bh_fdr(f_res)

    # 2. Micro-EMG partial controls (interior vs rim & raw_fp_ptp_ref)
    emg_controls = {}
    for trait in ("trait_stress_bias", "trait_high_arousal"):
        emg_controls[trait] = {}
        if "pre_interior_gamma" in pdf and "pre_rim_gamma" in pdf:
            cov_plus_rim = np.column_stack(
                [cov_full, pdf["pre_rim_gamma"].to_numpy(float)]
            )
            cov_plus_int = np.column_stack(
                [cov_full, pdf["pre_interior_gamma"].to_numpy(float)]
            )
            emg_controls[trait]["interior_gamma_ctrl_rim"] = _corr_ci_p(
                _residualize(pdf[trait], cov_plus_rim),
                _residualize(pdf["pre_interior_gamma"], cov_plus_rim),
            )
            emg_controls[trait]["rim_gamma_ctrl_interior"] = _corr_ci_p(
                _residualize(pdf[trait], cov_plus_int),
                _residualize(pdf["pre_rim_gamma"], cov_plus_int),
            )
        if "raw_fp_ptp_ref" in pdf and "pre_bp_pre_frontal_gamma" in pdf:
            cov_plus_fp = np.column_stack(
                [cov_full, pdf["raw_fp_ptp_ref"].to_numpy(float)]
            )
            emg_controls[trait]["frontal_gamma_ctrl_raw_fp"] = _corr_ci_p(
                _residualize(pdf[trait], cov_plus_fp),
                _residualize(pdf["pre_bp_pre_frontal_gamma"], cov_plus_fp),
            )

    # 2b. Trial-level interior vs rim gamma coupling with valence_own
    vid_int_rel, _ = _group_rel_power(frame, INTERIOR_CH, "bp_vid")
    vid_rim_rel, _ = _group_rel_power(frame, RIM_CH, "bp_vid")
    trial_emg = {}
    if "gamma" in vid_int_rel and "gamma" in int_rel and "gamma" in vid_rim_rel:
        d_int = vid_int_rel["gamma"] - int_rel["gamma"]
        d_rim = vid_rim_rel["gamma"] - rim_rel["gamma"]
        trial_emg["interior_gamma_valence"] = _safe_mean_ci(
            person._person_betas(frame, d_int, frame["rating_valence"])
        )
        trial_emg["rim_gamma_valence"] = _safe_mean_ci(
            person._person_betas(frame, d_rim, frame["rating_valence"])
        )
        # Regress interior gamma on valence_shared + valence_own + rim_gamma
        _, shared_v, own_v = person.components(frame, frame["rating_valence"])
        df_t = pd.DataFrame(
            {
                "p": frame["participant_id"],
                "y": d_int,
                "rim": d_rim,
                "sh": shared_v,
                "ow": own_v,
            }
        ).dropna()
        rows_b = []
        for _, g in df_t.groupby("p"):
            if len(g) < MIN_TRIALS or g["y"].std() == 0 or g["rim"].std() == 0:
                continue
            y_z = (g["y"] - g["y"].mean()) / g["y"].std()
            x_mat = np.column_stack(
                [
                    np.ones(len(g)),
                    (g["sh"] - g["sh"].mean()) / (g["sh"].std() or 1.0),
                    (g["ow"] - g["ow"].mean()) / (g["ow"].std() or 1.0),
                    (g["rim"] - g["rim"].mean()) / g["rim"].std(),
                ]
            )
            b, *_ = np.linalg.lstsq(x_mat, y_z.to_numpy(float), rcond=None)
            rows_b.append(
                {
                    "beta_shared_ctrl_rim": float(b[1]),
                    "beta_own_ctrl_rim": float(b[2]),
                    "beta_rim_cov": float(b[3]),
                }
            )
        trial_emg["interior_gamma_valence_ctrl_rim"] = _safe_mean_ci(
            pd.DataFrame(rows_b)
        )

    # 3. Sex-stratified checks (cohort + age adjusted inside female & male separately)
    sex_strat = {}
    for label_sex, flag in (("female", 1.0), ("male", 0.0)):
        sub = pdf[pdf["female"] == flag].reset_index(drop=True)
        scohs = sorted(sub["cohort"].dropna().unique())
        sc_cols = [(sub["cohort"] == c).astype(float).to_numpy() for c in scohs[1:]]
        scov = np.column_stack([*sc_cols, sub["age"].to_numpy(float)])
        sex_strat[label_sex] = {"n_people": int(len(sub))}
        for trait in ("trait_stress_bias", "trait_high_arousal"):
            ys = _residualize(sub[trait], scov)
            sex_strat[label_sex][trait] = {}
            for pred in (
                "pre_hr",
                "pre_bp_pre_frontal_gamma",
                "pre_interior_gamma",
                "pre_rim_gamma",
                "pre_bp_pre_right_temporal_theta",
            ):
                if pred in sub:
                    sex_strat[label_sex][trait][pred] = _corr_ci_p(
                        ys, _residualize(sub[pred], scov)
                    )

    # 4. Cross-day stability & ZERO-LEAKAGE cross-day prediction (Day 1 <-> Day 2)
    cross_day = {}
    if "recorded_day" in frame:
        d1_rows, d2_rows = [], []
        for p, g in frame.groupby("participant_id"):
            g1 = g[g["recorded_day"] == 1]
            g2 = g[g["recorded_day"] == 2]
            if len(g1) >= 5 and len(g2) >= 5:
                d1_rows.append(_agg_subset(g1, p))
                d2_rows.append(_agg_subset(g2, p))
        if len(d1_rows) >= 20:
            df1 = pd.DataFrame(d1_rows)
            df2 = pd.DataFrame(d2_rows)
            dc_cohs = sorted(df1["cohort"].dropna().unique())
            dc_cols = [
                (df1["cohort"] == c).astype(float).to_numpy() for c in dc_cohs[1:]
            ]
            dcov = np.column_stack(
                [*dc_cols, df1["age"].to_numpy(float), df1["female"].to_numpy(float)]
            )
            cross_day["n_paired_people"] = int(len(df1))
            # Test-retest reliability across days (cohort+age+sex adjusted)
            retest = {}
            for col in (
                "trait_stress_bias",
                "trait_high_arousal",
                "trait_neg_valence",
                "pre_hr",
                "pre_bp_pre_frontal_gamma",
                "pre_interior_gamma",
                "pre_rim_gamma",
                "pre_bp_pre_right_temporal_theta",
            ):
                if col in df1 and col in df2:
                    retest[col] = {
                        "raw": _corr_ci_p(df1[col], df2[col]),
                        "adjusted": _corr_ci_p(
                            _residualize(df1[col], dcov),
                            _residualize(df2[col], dcov),
                        ),
                    }
            cross_day["test_retest_d1_d2"] = retest
            # Strict cross-day prediction: Day 1 resting -> Day 2 trait & Day 2 -> Day 1
            xpred = {}
            for trait in ("trait_stress_bias", "trait_high_arousal"):
                y1 = _residualize(df1[trait], dcov)
                y2 = _residualize(df2[trait], dcov)
                xpred[trait] = {}
                for pred in (
                    "pre_hr",
                    "pre_bp_pre_frontal_gamma",
                    "pre_interior_gamma",
                    "pre_rim_gamma",
                    "pre_bp_pre_right_temporal_theta",
                ):
                    if pred in df1 and pred in df2:
                        p1 = _residualize(df1[pred], dcov)
                        p2 = _residualize(df2[pred], dcov)
                        c12 = _corr_ci_p(p1, y2)
                        c21 = _corr_ci_p(p2, y1)
                        avg_r = (
                            round(0.5 * (c12["r"] + c21["r"]), 4)
                            if c12 and c21
                            else None
                        )
                        xpred[trait][pred] = {
                            "d1_phys_to_d2_trait": c12,
                            "d2_phys_to_d1_trait": c21,
                            "mean_cross_day_r": avg_r,
                        }
            cross_day["cross_day_prediction"] = xpred

    # 5. Joint OLS summary (age, female, pre_hr, frontal gamma, R-temp theta)
    joint_ols = {}
    ols_cols = [
        c
        for c in (
            "age",
            "female",
            "pre_hr",
            "pre_bp_pre_frontal_gamma",
            "pre_bp_pre_right_temporal_theta",
        )
        if c in pdf
    ]
    for trait in ("trait_stress_bias", "trait_high_arousal"):
        y_coh = _residualize(pdf[trait], cov_cohort)
        x_mat = np.column_stack([_residualize(pdf[c], cov_cohort) for c in ols_cols])
        ok = np.isfinite(y_coh) & np.all(np.isfinite(x_mat), axis=1)
        if ok.sum() >= len(ols_cols) + 10:
            yz = (y_coh[ok] - y_coh[ok].mean()) / y_coh[ok].std()
            xz = (x_mat[ok] - x_mat[ok].mean(axis=0)) / x_mat[ok].std(axis=0)
            des = np.column_stack([np.ones(ok.sum()), xz])
            beta, *_ = np.linalg.lstsq(des, yz, rcond=None)
            resid = yz - des @ beta
            r2 = float(1.0 - np.sum(resid**2) / np.sum(yz**2))
            dof = ok.sum() - des.shape[1]
            s2 = float(np.sum(resid**2) / dof)
            se = np.sqrt(np.diag(s2 * np.linalg.pinv(des.T @ des)))
            tstats = beta / se
            joint_ols[trait] = {
                "n": int(ok.sum()),
                "r2_beyond_cohort": round(r2, 4),
                "terms": {
                    col: {
                        "std_beta": round(float(beta[i + 1]), 4),
                        "t": round(float(tstats[i + 1]), 3),
                    }
                    for i, col in enumerate(ols_cols)
                },
            }

    return {
        "n_people": int(len(pdf)),
        "cohort_adjusted": adj_cohort,
        "cohort_age_sex_adjusted": adj_full,
        "emg_partial_controls": emg_controls,
        "trial_level_emg": trial_emg,
        "sex_stratified": sex_strat,
        "cross_day": cross_day,
        "joint_ols": joint_ols,
    }


def main():
    """Run one Study-07 revision split and write aggregate JSON only."""
    parser = argparse.ArgumentParser()
    choices = (
        "criterion",
        "emotion_models",
        "fixed_video",
        "sensitivity",
        "sensitivity_controls",
        "ecg_morphology",
        "hep_raw",
        "sequence_models",
        "criterion_full",
    )
    parser.add_argument("--split", choices=choices, required=True)
    parser.add_argument("--model", nargs="*")  # accepted for shared workflow
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.split in ("criterion", "emotion_models", "fixed_video", "sensitivity"):
        table = load_base()
    elif args.split == "hep_raw":
        table = load_rich(with_hep=True)
    else:
        table = load_rich(with_hep=False)

    result = {args.split: globals()[args.split](table)}
    text = json.dumps(result, indent=1, default=lambda x: None)
    print(text[:5000])
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.split}_metrics.json").write_text(text)


if __name__ == "__main__":
    main()
