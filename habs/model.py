"""Regularized stimulus and physiology prediction with explicit J1 calibration."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from habs.data import target_values
from habs.features import ECG, EDA, EEG, PHYSIOLOGY
from habs.profiles import PhysiologyProfiles

CATEGORICAL = ("video", "expected_emotion_label", "expected_emotion_source")
INPUT_COLUMNS = (
    "participant_id",
    "recorded_day",
    "evaluation_eligible",
    "event_order",
    *CATEGORICAL,
    *PHYSIOLOGY,
)


class CalibratedRegressor(RegressorMixin, BaseEstimator):
    """Compare prospective information under the same first-session history.

    Parameters
    ----------
    modality : str, default="context"
        Context, missingness-only, eeg, peripheral, all, no_eda, or history-only.
    alpha : float, default=10.0
        Population ridge regularization strength.
    calibrated : bool, default=True
        Apply the personal J1 median residual offset to later-session predictions.
    shrinkage : float, default=10.0
        Personal-offset shrinkage denominator added to the number of J1 trials.
    target : str, default="score"
        Rating-derived outcome, as defined by target_values.
    nonlinear : bool, default=False
        Add cubic spline bases for continuous numerical inputs.
    profile : str or None, default=None
        Optional cluster or PCA description of J1 physiology.
    n_clusters : int, default=3
        Development-selected profile size; PCA uses K-1 components.
    conditional_arousal : bool, default=False
        Include observed current arousal in the explicitly explanatory analysis.
    """

    def __init__(
        self,
        modality="context",
        alpha=10.0,
        calibrated=True,
        shrinkage=10.0,
        target="score",
        nonlinear=False,
        profile=None,
        n_clusters=3,
        conditional_arousal=False,
    ):
        self.modality = modality
        self.alpha = alpha
        self.calibrated = calibrated
        self.shrinkage = shrinkage
        self.target = target
        self.nonlinear = nonlinear
        self.profile = profile
        self.n_clusters = n_clusters
        self.conditional_arousal = conditional_arousal

    def _columns(self):
        return {
            "context": (),
            "missing": PHYSIOLOGY,
            "eeg": EEG,
            "peripheral": ECG + EDA,
            "all": PHYSIOLOGY,
            "no_eda": EEG + ECG,
            "history": PHYSIOLOGY,
        }[self.modality]

    def _check_history(self, history):
        if history is None or (history["recorded_day"] != 1).any():
            raise ValueError("Calibration must be a separate first-session-only table.")

    def _numeric(self, table, history):
        columns = self._columns()
        order = table["event_order"].to_numpy(dtype=float)
        values = [
            order[:, None],
            (order**2)[:, None],
            table["recorded_day"].to_numpy(dtype=float)[:, None],
        ]
        if self.conditional_arousal:
            values.append(table["observed_arousal"].to_numpy(dtype=float)[:, None])
        if columns:
            observed = table[list(columns)].to_numpy(dtype=float)
            medians, coverage, available = {}, {}, {}
            for pid, group in history.groupby("participant_id"):
                matrix = group[list(columns)].to_numpy(dtype=float)
                medians[str(pid)] = np.asarray(
                    [
                        np.median(column[np.isfinite(column)])
                        if np.isfinite(column).any()
                        else np.nan
                        for column in matrix.T
                    ]
                )
                complete = np.isfinite(matrix).all(axis=1)
                coverage[str(pid)] = float(complete.mean())
                available[str(pid)] = float(complete.sum() >= 5)
            personal = np.vstack(
                [
                    medians.get(str(pid), np.full(len(columns), np.nan))
                    for pid in table["participant_id"]
                ]
            )
            missing = ~np.isfinite(observed)
            if self.modality == "missing":
                values.extend(
                    [missing.astype(float), (~np.isfinite(personal)).astype(float)]
                )
            elif self.modality == "history":
                values.extend([personal, (~np.isfinite(personal)).astype(float)])
            else:
                values.extend(
                    [
                        observed,
                        observed - personal,
                        missing.astype(float),
                        (~np.isfinite(personal)).astype(float),
                    ]
                )
            values.append(
                np.asarray(
                    [coverage.get(str(pid), 0.0) for pid in table["participant_id"]]
                )[:, None]
            )
            values.append(
                np.asarray(
                    [available.get(str(pid), 0.0) for pid in table["participant_id"]]
                )[:, None]
            )
        if self.profile is not None:
            descriptors = self.profiles_.summarize(history, self.profile)
            values.append(
                np.vstack(
                    [
                        descriptors.get(str(pid), np.zeros(self.n_clusters - 1))
                        for pid in table["participant_id"]
                    ]
                )
            )
        return np.column_stack(values)

    def _categorical(self, table):
        context = table[list(CATEGORICAL)].fillna("missing").astype(str)
        context["video_expected"] = (
            context["video"] + "|" + context["expected_emotion_label"]
        )
        return context

    def _matrix(self, table, history, *, fit=False):
        numeric = self._numeric(table, history)
        categorical = self._categorical(table)
        if fit:
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            numeric = self.imputer_.fit_transform(numeric)
            self.scaler_ = StandardScaler().fit(numeric)
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = self.encoder_.fit_transform(categorical)
            if self.nonlinear:
                start = 3 + int(self.conditional_arousal)
                width = len(self._columns()) * (1 if self.modality == "history" else 2)
                self.spline_columns_ = np.arange(start, start + width)
                self.splines_ = SplineTransformer(
                    n_knots=3, degree=3, include_bias=False, extrapolation="constant"
                ).fit(numeric[:, self.spline_columns_])
        else:
            numeric = self.imputer_.transform(numeric)
            encoded = self.encoder_.transform(categorical)
        matrices = [encoded, self.scaler_.transform(numeric)]
        if self.nonlinear:
            matrices.append(self.splines_.transform(numeric[:, self.spline_columns_]))
        return np.column_stack(matrices)

    def fit(self, X, y, *, calibration=None):
        """Fit population transforms and coefficients on the supplied rows only.

        Parameters
        ----------
        X : pandas.DataFrame
            Population training trial predictors without response columns.
        y : array-like
            Population training responses.
        calibration : pandas.DataFrame
            Separate J1 history; any profile fitting is restricted to X's people.

        Returns
        -------
        self
            Fitted population model.
        """
        self._check_history(calibration)
        self.training_people_ = tuple(sorted(set(X["participant_id"])))
        self.profiles_ = None
        if self.profile is not None:
            training_history = calibration.loc[
                calibration["participant_id"].isin(self.training_people_)
            ]
            self.profiles_ = PhysiologyProfiles(self.n_clusters).fit(training_history)
        matrix = self._matrix(X, calibration, fit=True)
        counts = X["participant_id"].value_counts()
        weights = X["participant_id"].map(1 / counts).to_numpy()
        weights = weights * (len(weights) / weights.sum())
        self.regressor_ = Ridge(alpha=self.alpha).fit(matrix, y, sample_weight=weights)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, *, calibration=None):
        """Predict each requested row using only that person's observed J1 history.

        Parameters
        ----------
        X : pandas.DataFrame
            Requested prediction grid. Current response ratings are unnecessary.
        calibration : pandas.DataFrame
            First-session history, never future-session outcomes.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            One bounded response prediction per requested row.
        """
        self._check_history(calibration)
        prediction = self.regressor_.predict(self._matrix(X, calibration))
        if self.calibrated:
            people = set(X.loc[X["recorded_day"] > 1, "participant_id"])
            history = calibration.loc[calibration["participant_id"].isin(people)]
            if len(history):
                historical_X = history.loc[:, list(INPUT_COLUMNS)].copy()
                if self.conditional_arousal:
                    historical_X["observed_arousal"] = (
                        history["rating_arousal"] - 1
                    ) / 8
                base = self.regressor_.predict(self._matrix(historical_X, calibration))
                residual = target_values(history, self.target).to_numpy() - base
                history_people = history["participant_id"].to_numpy()
                for person in people:
                    historical_mask = history_people == person
                    count = int(historical_mask.sum())
                    if count < 5:
                        continue
                    offset = np.median(residual[historical_mask])
                    offset *= count / (count + self.shrinkage)
                    mask = (X["participant_id"] == person) & (X["recorded_day"] > 1)
                    prediction[mask.to_numpy()] += offset
        return np.clip(prediction, 0, 1)
