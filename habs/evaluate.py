"""Keep population training people disjoint from later-session validation."""

import numpy as np
from sklearn.model_selection import BaseCrossValidator, GroupKFold


class SessionGroupedSplit(BaseCrossValidator):
    """Hold out participants entirely and score only their eligible J2 trials.

    Parameters
    ----------
    eligible_people : sequence of str
        Development participants eligible for later-session validation.
    n_splits : int, default=5
        Number of grouped development folds.
    random_state : int, default=20261108
        Deterministic allocation seed.
    """

    def __init__(self, eligible_people, n_splits=5, random_state=20261108):
        self.eligible_people = tuple(eligible_people)
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the configured number of participant folds.

        Parameters
        ----------
        X, y, groups : object, optional
            Unused sklearn cross-validator arguments.

        Returns
        -------
        int
            Number of folds.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yield training rows and novel later-session validation rows.

        Parameters
        ----------
        X : pandas.DataFrame
            Predict-grid metadata including participant and evaluation eligibility.
        y : object, optional
            Unused; response values never determine the split.
        groups : array-like, optional
            Participant identifiers routed from the graph's X marker.

        Yields
        ------
        tuple of numpy.ndarray
            Population training indices and later-session validation indices.

        Raises
        ------
        ValueError
            If the grouping metadata disagrees or there are too few people.
        """
        people = X["participant_id"].to_numpy()
        if groups is not None and not np.array_equal(people, np.asarray(groups)):
            raise ValueError("Group routing does not match the prediction grid.")
        future = (X["recorded_day"].to_numpy() == 2) & X[
            "evaluation_eligible"
        ].to_numpy(dtype=bool)
        active = sorted(set(people[future]) & set(self.eligible_people))
        if len(active) < self.n_splits:
            raise ValueError("Too few participants with eligible future trials.")
        active = np.asarray(active)
        splitter = GroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for _, held_indices in splitter.split(
            np.zeros((len(active), 1)), groups=active
        ):
            held_people = active[held_indices]
            held = np.isin(people, held_people)
            yield np.flatnonzero(~held), np.flatnonzero(held & future)
