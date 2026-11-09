"""Compare the same forecaster with and without a personal J1 carryover slope."""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from habs.temporal.preprocess import CATEGORICAL, NUMERIC


class TemporalForecaster(RegressorMixin, BaseEstimator):
    """Forecast the standardized arousal–EEG difference using observed past only.

    Parameters
    ----------
    mode : str, default="full"
        Full personal lagged-arousal correction, its exact ablation, or a
        separately refitted restricted calibration (``refit``).
    population_alpha : float, default=10
        Fixed ridge penalty for the population context/history model.
    personal_penalty : float, default=20
        Fixed penalty for personal slopes; the personal intercept is unpenalized.
    """

    def __init__(self, mode="full", population_alpha=10.0, personal_penalty=20.0):
        self.mode = mode
        self.population_alpha = population_alpha
        self.personal_penalty = personal_penalty

    def _check_history(self, calibration):
        if calibration is None or (calibration["recorded_day"] != 1).any():
            raise ValueError("Personal calibration requires first-session-only data.")
        if not np.isfinite(calibration["target_difference"]).all():
            raise ValueError(
                "Personal calibration contains invalid historical targets."
            )

    def _matrix(self, table, *, fit=False):
        categorical = table[list(CATEGORICAL)].to_numpy(dtype=str)
        numeric = table[list(NUMERIC)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(
                "Predictors must be finite after causal history construction."
            )
        if fit:
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = self.encoder_.fit_transform(categorical)
            self.scaler_ = StandardScaler().fit(numeric)
        else:
            encoded = self.encoder_.transform(categorical)
        return np.column_stack((encoded, self.scaler_.transform(numeric)))

    def fit(self, X, y, *, calibration=None):
        """Fit population coefficients on training IDs and their J1 observations.

        Parameters
        ----------
        X : pandas.DataFrame
            Training J2 predictors without current response values.
        y : array-like of shape (n_samples,)
            Current standardized difference on training rows.
        calibration : pandas.DataFrame
            Observed J1 predictors and targets, including allowed personal history
            for prediction IDs. Only training IDs enter the population fit.

        Returns
        -------
        self
            Fitted population forecaster with a frozen past-only calibration source.

        Raises
        ------
        ValueError
            If history is not J1-only or the calibration mode is unsupported.
        """
        self._check_history(calibration)
        if self.mode not in {"full", "ablation", "refit"}:
            raise ValueError("Unknown personal-slope comparison.")
        self.training_people_ = tuple(sorted(set(X["participant_id"])))
        past = calibration.loc[
            calibration["participant_id"].isin(self.training_people_)
        ]
        train = pd.concat(
            [X, past.drop(columns=["target_difference"])], ignore_index=True
        )
        target = np.r_[np.asarray(y), past["target_difference"].to_numpy()]
        counts = Counter(train["participant_id"])
        weights = np.asarray([1 / counts[p] for p in train["participant_id"]])
        weights *= len(weights) / weights.sum()
        self.regressor_ = Ridge(alpha=self.population_alpha).fit(
            self._matrix(train, fit=True), target, sample_weight=weights
        )
        self.calibration_ = calibration.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def _personal(self, history):
        residuals = history["target_difference"].to_numpy() - self.regressor_.predict(
            self._matrix(history)
        )
        people = history["participant_id"].to_numpy()
        a, p = history["lag_arousal_z"].to_numpy(), history["lag_eeg_z"].to_numpy()
        coefficients = {}
        for person in sorted(set(people)):
            mask = people == person
            if self.mode == "refit":
                design = np.column_stack((np.ones(mask.sum()), p[mask]))
                penalty = np.diag([0.0, self.personal_penalty])
            else:
                design = np.column_stack((np.ones(mask.sum()), a[mask], p[mask]))
                penalty = np.diag([0.0, self.personal_penalty, self.personal_penalty])
            solution = cho_solve(
                cho_factor(design.T @ design + penalty, lower=True),
                design.T @ residuals[mask],
            )
            if self.mode == "refit":
                solution = np.asarray([solution[0], 0.0, solution[1]])
            if self.mode == "ablation":
                solution[1] = 0.0
            coefficients[person] = solution
        return coefficients

    def predict(self, X, *, calibration=None):
        """Predict every query using the population model and observed J1 only.

        Parameters
        ----------
        X : pandas.DataFrame
            Context and predecessor predictors, including a cold query grid.
        calibration : pandas.DataFrame or None, default=None
            Rebound J1 source, or the fitted J1 snapshot for standard score calls.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Unclipped predictions of a standardized measurement difference.
        """
        history = self.calibration_ if calibration is None else calibration
        self._check_history(history)
        coefficients = self._personal(history)
        base = self.regressor_.predict(self._matrix(X))
        corrections = np.asarray(
            [
                np.asarray([1.0, r.lag_arousal_z, r.lag_eeg_z])
                @ coefficients.get(r.participant_id, np.zeros(3))
                for r in X.itertuples()
            ]
        )
        return base + corrections
