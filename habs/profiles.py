"""Describe first-session physiological profiles after population stimulus control."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from habs.features import PHYSIOLOGY


class PhysiologyProfiles:
    """Learn continuous and discrete descriptions without emotional response labels.

    Parameters
    ----------
    n_clusters : int, default=3
        Number selected on development data only.
    random_state : int, default=20261108
        Deterministic clustering initialization.
    """

    def __init__(self, n_clusters=3, random_state=20261108):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, history):
        """Fit stimulus centering and representations on training-person J1.

        Parameters
        ----------
        history : pandas.DataFrame
            First-session training trials with video and physiological columns.

        Returns
        -------
        self
            Fitted representation, disabled when complete-case support is small.
        """
        complete = history.loc[np.isfinite(history[list(PHYSIOLOGY)]).all(axis=1)]
        self.enabled_ = (
            len(complete) >= 100 and complete["participant_id"].nunique() >= 10
        )
        self.training_rows_ = len(complete)
        self.training_people_ = complete["participant_id"].nunique()
        if not self.enabled_:
            return self
        self.video_means_ = complete.groupby("video")[list(PHYSIOLOGY)].mean()
        self.global_mean_ = complete[list(PHYSIOLOGY)].mean()
        residuals = self._residuals(complete)
        self.scaler_ = StandardScaler().fit(residuals)
        standardized = self.scaler_.transform(residuals)
        self.clusters_ = KMeans(
            n_clusters=self.n_clusters, n_init=20, random_state=self.random_state
        ).fit(standardized)
        self.pca_ = PCA(n_components=self.n_clusters - 1, svd_solver="full").fit(
            standardized
        )
        return self

    def _residuals(self, table):
        means = (
            self.video_means_.reindex(table["video"])
            .fillna(self.global_mean_)
            .to_numpy()
        )
        return table[list(PHYSIOLOGY)].to_numpy(dtype=float) - means

    def summarize(self, history, kind="cluster"):
        """Return per-person descriptors from a supplied session without refitting.

        Parameters
        ----------
        history : pandas.DataFrame
            Trials of the session being described. Response ratings are ignored.
        kind : {"cluster", "pca"}, default="cluster"
            Frequencies of the first K-1 clusters or mean PCA coordinates.

        Returns
        -------
        dict
            Participant identifiers mapped to descriptor arrays. At least five
            complete trials are required for a person's descriptor.
        """
        if not self.enabled_:
            return {}
        complete = history.loc[np.isfinite(history[list(PHYSIOLOGY)]).all(axis=1)]
        if len(complete) == 0:
            return {}
        standardized = self.scaler_.transform(self._residuals(complete))
        if kind == "cluster":
            labels = self.clusters_.predict(standardized)
            representation = np.column_stack(
                [labels == i for i in range(self.n_clusters - 1)]
            )
        elif kind == "pca":
            representation = self.pca_.transform(standardized)
        else:
            raise ValueError(f"Unknown representation {kind!r}.")
        frame = pd.DataFrame(representation, index=complete.index)
        frame["participant_id"] = complete["participant_id"].to_numpy()
        return {
            str(pid): group.drop(columns="participant_id").mean().to_numpy()
            for pid, group in frame.groupby("participant_id")
            if len(group) >= 5
        }
