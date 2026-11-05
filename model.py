"""Predict future-session ratings after participant-specific calibration."""

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import ECG_FEATURES, EEG_FEATURES, FEATURES, RAW_FEATURES

META = ["video", "expected_emotion_label", "expected_emotion_source"]


class InsufficientSupport(ValueError):
    """The required participant or stimulus support cannot be attained."""


def predictor_frame(frame):
    """Exclude all current-trial ratings and derived emotion labels from X."""
    allowed = [
        "participant_id",
        "order",
        *META,
        *FEATURES,
        *RAW_FEATURES,
        *EEG_FEATURES,
        *ECG_FEATURES,
        "eda_valid",
        "eeg_valid",
        "ecg_valid",
    ]
    return frame[[name for name in allowed if name in frame]].copy()


def center(values, people):
    """Remove participant means from a vector or matrix used in training/scoring."""
    result = np.asarray(values, dtype=float).copy()
    people = np.asarray(people)
    for person in np.unique(people):
        mask = people == person
        result[mask] -= result[mask].mean(axis=0)
    return result


def valid_rows(frame, threshold=0.05, require_rating=True):
    """Select measured trials without consulting their prediction errors."""
    if "any_valid" in frame:
        mask = (frame.join_status == "matched") & frame.any_valid
    else:
        mask = (
            (frame.join_status == "matched")
            & (frame.finite_fraction >= 0.99)
            & ~frame.flat
            & (frame.clamp_fraction < threshold)
        )
    if require_rating:
        mask &= frame.rating_arousal.between(1, 9)
    return frame.loc[mask].copy()


def select_grid(frame, people, videos, threshold=0.05, minimum=20):
    """Select future-session rows with adequate calibration and stimulus support."""
    rows = valid_rows(frame, threshold, require_rating=False)
    rows = rows.loc[(rows.day != 1) | rows.rating_arousal.between(1, 9)]
    rows = rows.loc[rows.participant_id.isin(people) & rows.video.isin(videos)]
    counts = rows.groupby(["participant_id", "day"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[1, 2], fill_value=0)
    eligible = counts.index[(counts[1] >= minimum) & (counts[2] >= minimum)]
    return rows.loc[(rows.day == 2) & rows.participant_id.isin(eligible)].reset_index(
        drop=True
    )


def attach_calibration(grid, source, videos, threshold=0.05):
    """Attach only day-one observations to the prediction rows."""
    history = valid_rows(source, threshold)
    history = history.loc[(history.day == 1) & history.video.isin(videos)]
    histories = {
        person: predictor_frame(rows)
        .assign(rating_arousal=rows.rating_arousal)
        .reset_index(drop=True)
        for person, rows in history.groupby("participant_id")
    }
    result = predictor_frame(grid)
    result["calibration"] = [histories[person] for person in result.participant_id]
    return result


def reference_only(frame, paired_people, threshold=0.05):
    """Keep valid trials from people ineligible for the two-session experiment."""
    rows = valid_rows(frame, threshold)
    rows = rows.loc[~rows.participant_id.isin(paired_people)]
    return predictor_frame(rows).assign(rating_arousal=rows.rating_arousal)


def supported_grid(grid, source, minimum_sources=10, minimum_trials=20):
    """Restrict validation and calibration to adequately represented videos."""
    counts = source.groupby("video").participant_id.nunique()
    videos = counts.index[counts >= minimum_sources]
    result = grid.loc[grid.video.isin(videos)].copy()
    histories = [
        history.loc[history.video.isin(videos)] for history in result.calibration
    ]
    result["calibration"] = histories
    result = result.loc[[len(history) >= minimum_trials for history in histories]]
    counts = result.groupby("participant_id").size()
    return result.loc[
        result.participant_id.isin(counts.index[counts >= minimum_trials])
    ]


class VideoReference:
    """Fit the known-catalogue reference without using held-out participants."""

    def __init__(self, alpha):
        self.alpha = alpha

    def matrix(self, rows, fit=False):
        """Encode stimulus, prior category, source and chronological position."""
        categorical = rows[META].fillna("missing").astype(str)
        if fit:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = self.encoder.fit_transform(categorical)
        else:
            encoded = self.encoder.transform(categorical)
        order = rows.order.to_numpy(dtype=float)
        return np.column_stack([encoded, order, order**2])

    def fit(self, rows):
        """Estimate the population response from source participants only."""
        self.regressor = Ridge(alpha=self.alpha)
        self.regressor.fit(self.matrix(rows, fit=True), rows.rating_arousal)
        return self

    def predict(self, rows):
        """Predict the population response for the supplied trials."""
        return self.regressor.predict(self.matrix(rows))


def population(grid, y):
    """Collect training targets and calibration data for training people only."""
    future = grid.drop(columns="calibration").assign(rating_arousal=np.asarray(y))
    histories = grid.drop_duplicates("participant_id").calibration.tolist()
    return pd.concat([future, *histories], ignore_index=True)


def calibrated(reference, grid, level_penalty, slope_penalty, affine):
    """Estimate a personal offset and nonnegative gain from day one only."""
    prediction = reference.predict(grid)
    for person in grid.participant_id.unique():
        mask = (grid.participant_id == person).to_numpy()
        history = grid.loc[mask, "calibration"].iloc[0]
        mu = reference.predict(history)
        target = history.rating_arousal.to_numpy(dtype=float)
        slope = 1.0
        if affine and np.var(mu) >= 0.1:
            design = np.column_stack([np.ones(len(mu)), mu])
            penalty = np.diag([level_penalty, slope_penalty])
            rhs = design.T @ target + np.array([0.0, slope_penalty])
            offset, slope = np.linalg.solve(design.T @ design + penalty, rhs)
            slope = max(0.0, float(slope))
        offset = float(np.sum(target - slope * mu) / (len(mu) + level_penalty))
        prediction[mask] = offset + slope * prediction[mask]
    return prediction


def feature_columns(mode):
    """Keep modality comparisons fixed before model selection."""
    return {
        "eda": FEATURES,
        "raw": RAW_FEATURES,
        "eeg": EEG_FEATURES,
        "ecg": ECG_FEATURES,
    }.get(mode, FEATURES + EEG_FEATURES + ECG_FEATURES)


def observed_features(rows, columns):
    """Mask measurements that failed their own modality-specific quality test."""
    values = rows.reindex(columns=columns).to_numpy(dtype=float).copy()
    for index, name in enumerate(columns):
        modality = (
            "eda" if name.startswith(("eda_", "voltage_")) else name.split("_")[0]
        )
        flag = f"{modality}_valid"
        if flag in rows:
            values[~rows[flag].to_numpy(dtype=bool), index] = np.nan
    return values


def physiological_matrix(grid, source, mode):
    """Use calibration scaling, training-only fallback and missingness controls."""
    columns = feature_columns(mode)
    result = np.zeros((len(grid), 2 * len(columns)))
    if mode == "person":
        return result
    source_values = observed_features(source, columns)
    means, scales = [], []
    for index in range(len(columns)):
        valid = source_values[np.isfinite(source_values[:, index]), index]
        means.append(float(valid.mean()) if len(valid) else 0.0)
        scales.append(max(float(valid.std()), 1e-6) if len(valid) else 1.0)
    averages = pd.DataFrame(source_values, columns=columns).assign(
        video=source.video.to_numpy()
    )
    averages = averages.groupby("video")[columns].mean()
    for person in grid.participant_id.unique():
        mask = (grid.participant_id == person).to_numpy()
        history = grid.loc[mask, "calibration"].iloc[0]
        values = observed_features(grid.loc[mask], columns)
        calibration = observed_features(history, columns)
        missing = ~np.isfinite(values)
        if mode == "video":
            values = averages.reindex(grid.loc[mask, "video"]).to_numpy()
            values = np.where(missing, np.nan, values)
        for index in range(len(columns)):
            valid = calibration[np.isfinite(calibration[:, index]), index]
            mean = float(valid.mean()) if len(valid) >= 5 else means[index]
            scale = max(float(valid.std()), 1e-6) if len(valid) >= 5 else scales[index]
            normalized = (values[:, index] - mean) / scale
            normalized = np.where(np.isfinite(normalized), normalized, 0.0)
            result[mask, index] = np.clip(normalized, -10, 10)
        result[mask, len(columns) :] = missing.astype(float)
    if mode == "missing":
        result[:, : len(columns)] = 0.0
    return result


class CalibratedArousal(RegressorMixin, BaseEstimator):
    """Add a cross-fitted physiological correction to a calibrated reference.

    Parameters
    ----------
    mode : str
        Baseline, individual modality, 'all', or a signal substitution control.
    base_alpha : float
        Penalty of the population reference.
    level_penalty : float
        Shrinkage of personal offsets towards zero.
    slope_penalty : float
        Shrinkage of personal gains towards one.
    signal_alpha : float
        Ridge penalty of the physiological residual correction.
    minimum_sources : int
        Required source participants per validation video.
    minimum_trials : int
        Required calibration and validation trials per participant.
    """

    def __init__(
        self,
        mode="affine",
        base_alpha=10.0,
        level_penalty=10.0,
        slope_penalty=10.0,
        signal_alpha=10.0,
        minimum_sources=10,
        minimum_trials=20,
    ):
        self.mode = mode
        self.base_alpha = base_alpha
        self.level_penalty = level_penalty
        self.slope_penalty = slope_penalty
        self.signal_alpha = signal_alpha
        self.minimum_sources = minimum_sources
        self.minimum_trials = minimum_trials

    def fit(self, X, y, background=None):
        """Fit population and residual models using training people only."""
        self.source_ = population(X, y)
        if background is not None:
            if set(background.participant_id) & set(X.participant_id):
                raise ValueError("Reference-only and paired participants overlap")
            self.source_ = pd.concat([self.source_, background], ignore_index=True)
        self.reference_ = VideoReference(self.base_alpha).fit(self.source_)
        self.n_features_in_ = X.shape[1]
        if self.mode in ("video_only", "level", "affine"):
            return self
        people = X.participant_id.to_numpy()
        folds = GroupKFold(n_splits=min(5, len(np.unique(people))))
        residual = np.zeros(len(X))
        matrix = np.zeros((len(X), 2 * len(feature_columns(self.mode))))
        retained = np.zeros(len(X), dtype=bool)
        y = np.asarray(y, dtype=float)
        X = X.reset_index(drop=True)
        for train, validation in folds.split(X, y, people):
            source = population(X.iloc[train], y[train])
            if background is not None:
                source = pd.concat([source, background], ignore_index=True)
            reference = VideoReference(self.base_alpha).fit(source)
            grid = supported_grid(
                X.iloc[validation], source, self.minimum_sources, self.minimum_trials
            )
            if not len(grid):
                continue
            positions = grid.index.to_numpy()
            baseline = calibrated(
                reference, grid, self.level_penalty, self.slope_penalty, True
            )
            residual[positions] = y[positions] - baseline
            matrix[positions] = physiological_matrix(grid, source, self.mode)
            retained[positions] = True
        if len(np.unique(people[retained])) < 2:
            raise InsufficientSupport(
                "Insufficient cross-fitted stimulus support for a correction"
            )
        self.correction_people_ = len(np.unique(people[retained]))
        matrix = center(matrix[retained], people[retained])
        residual = center(residual[retained], people[retained])
        self.scaler_ = StandardScaler().fit(matrix)
        self.correction_ = Ridge(alpha=self.signal_alpha, fit_intercept=False)
        self.correction_.fit(self.scaler_.transform(matrix), residual)
        return self

    def predict(self, X):
        """Predict without access to the current-session ratings."""
        if self.mode == "video_only":
            return self.reference_.predict(X)
        baseline = calibrated(
            self.reference_,
            X,
            self.level_penalty,
            self.slope_penalty,
            self.mode != "level",
        )
        if self.mode in ("level", "affine"):
            return baseline
        matrix = physiological_matrix(X, self.source_, self.mode)
        return baseline + self.correction_.predict(self.scaler_.transform(matrix))


def build_learner(estimator, splitter, threshold=0.05, minimum=20):
    """Declare a replayable trial predictor with separate calibration history."""
    source = skrub.var("feature_file").skb.apply_func(pd.read_csv)
    people, videos = skrub.var("people"), skrub.var("videos")
    grid = source.skb.apply_func(select_grid, people, videos, threshold, minimum)
    y = grid["rating_arousal"].skb.mark_as_y()
    X = grid.skb.apply_func(predictor_frame).skb.mark_as_X(
        cv=splitter, split_kwargs={"groups": grid["participant_id"]}
    )
    X = X.skb.apply_func(attach_calibration, source, videos, threshold)
    background = source.skb.apply_func(
        reference_only, skrub.var("paired_people"), threshold
    )
    return X.skb.apply(
        estimator, y=y, no_wrap=True, fit_kwargs={"background": background}
    ).skb.make_learner()
