"""Verify calibration isolation and prediction invariances on synthetic trials."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import FEATURES, RAW_FEATURES
from model import (
    CalibratedArousal,
    attach_calibration,
    build_learner,
    physiological_matrix,
    population,
    select_grid,
    supported_grid,
)


def synthetic_trials(people=8, videos=30, seed=2):
    """Generate known video, person and trial effects for implementation tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(people):
        offset = rng.normal(0, 0.3)
        for day in (1, 2):
            for video in range(videos):
                signal = rng.normal(size=3)
                rating = (
                    5 + np.sin(video) + offset + 0.5 * signal[1] + rng.normal(0, 0.1)
                )
                rows.append(
                    {
                        "participant_id": f"P{person:03d}",
                        "day": day,
                        "video": f"v{video}",
                        "start_marker": video,
                        "rating_arousal": float(np.clip(rating, 1, 9)),
                        "expected_emotion_label": "neutral",
                        "expected_emotion_source": "default",
                        "join_status": "matched",
                        "finite_fraction": 1.0,
                        "clamp_fraction": 0.0,
                        "flat": False,
                        "order": video / videos,
                        **dict(zip(FEATURES, signal, strict=True)),
                        **dict(zip(RAW_FEATURES, signal, strict=True)),
                    }
                )
    return pd.DataFrame(rows)


def test_prediction_does_not_read_future_ratings(tmp_path):
    """Removing held-out outcomes must leave every prediction unchanged."""
    frame = synthetic_trials()
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    people = sorted(frame.participant_id.unique())
    env = {
        "feature_file": str(path),
        "people": people[:6],
        "paired_people": people,
        "videos": sorted(frame.video.unique()),
    }
    estimator = CalibratedArousal(mode="eda", minimum_sources=1, minimum_trials=3)
    learner = build_learner(estimator, GroupKFold(5), minimum=3)
    learner.fit(env)
    test_env = {**env, "people": people[6:]}
    original = learner.predict(test_env)
    frame.loc[
        frame.participant_id.isin(people[6:]) & (frame.day == 2), "rating_arousal"
    ] = np.nan
    hidden = tmp_path / "hidden_outcomes.csv"
    frame.to_csv(hidden, index=False)
    refitted = build_learner(estimator, GroupKFold(5), minimum=3)
    refitted.fit({**env, "feature_file": str(hidden)})
    np.testing.assert_allclose(
        original, refitted.predict({**test_env, "feature_file": str(hidden)})
    )
    isolated = frame.loc[frame.participant_id.isin(people[6:])]
    other = tmp_path / "future.csv"
    isolated.to_csv(other, index=False)
    replay = learner.predict({**test_env, "feature_file": str(other)})
    assert len(original) == len(replay) == 60
    np.testing.assert_allclose(original, replay)


def test_person_only_correction_equals_calibrated_reference():
    """A constant personal signal has no within-person information to fit."""
    frame = synthetic_trials()
    grid = select_grid(
        frame, frame.participant_id.unique(), frame.video.unique(), minimum=3
    )
    X = attach_calibration(
        grid.drop(columns="rating_arousal"), frame, frame.video.unique()
    )
    y = grid.rating_arousal
    reference = CalibratedArousal(mode="affine").fit(X, y)
    control = CalibratedArousal(mode="person", minimum_sources=1, minimum_trials=3).fit(
        X, y
    )
    np.testing.assert_allclose(reference.predict(X), control.predict(X))


def test_reusing_residuals_matches_complete_refits():
    """Penalty selection must preserve every prediction after caching residuals."""
    frame = synthetic_trials()
    grid = select_grid(
        frame, frame.participant_id.unique(), frame.video.unique(), minimum=3
    )
    X = attach_calibration(
        grid.drop(columns="rating_arousal"), frame, frame.video.unique()
    )
    for mode in ("all", "video", "missing"):
        reused = CalibratedArousal(mode=mode, minimum_sources=1, minimum_trials=3).fit(
            X, grid.rating_arousal
        )
        for alpha in (0.1, 10.0, 1000.0):
            reused.refit_correction(alpha)
            fresh = CalibratedArousal(
                mode=mode,
                signal_alpha=alpha,
                minimum_sources=1,
                minimum_trials=3,
            ).fit(X, grid.rating_arousal)
            np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))


def test_shared_history_filtering_preserves_distinct_tables():
    """Caching repeated tables must not merge different histories for one ID."""
    frame = synthetic_trials()
    grid = select_grid(
        frame, frame.participant_id.unique(), frame.video.unique(), minimum=3
    )
    X = attach_calibration(
        grid.drop(columns="rating_arousal"), frame, frame.video.unique()
    )
    alternative = X.iloc[0].calibration.copy()
    alternative.loc[:, "rating_arousal"] = 2.0
    X.at[X.index[0], "calibration"] = alternative
    videos = sorted(frame.video.unique())[:15]
    source = frame.loc[frame.video.isin(videos)]
    actual = supported_grid(X, source, minimum_sources=1, minimum_trials=3)
    expected = X.loc[X.video.isin(videos)]
    pd.testing.assert_frame_equal(
        actual.drop(columns="calibration"), expected.drop(columns="calibration")
    )
    for observed, history in zip(actual.calibration, expected.calibration, strict=True):
        pd.testing.assert_frame_equal(observed, history.loc[history.video.isin(videos)])


def test_video_control_preserves_personal_calibration():
    """Identical signals must give identical scaled features in the control."""
    frame = synthetic_trials()
    grid = select_grid(
        frame, frame.participant_id.unique(), frame.video.unique(), minimum=3
    )
    X = attach_calibration(
        grid.drop(columns="rating_arousal"), frame, frame.video.unique()
    )
    source = population(X, grid.rating_arousal)
    averages = source.groupby("video")[FEATURES].mean()
    X[FEATURES] = averages.reindex(X.video).to_numpy()
    np.testing.assert_allclose(
        physiological_matrix(X, source, "all"), physiological_matrix(X, source, "video")
    )


def test_unused_session_column_cannot_change_predictions():
    """Session selection happens before fitting; its constant tag is not a feature."""
    frame = synthetic_trials()
    grid = select_grid(
        frame, frame.participant_id.unique(), frame.video.unique(), minimum=3
    )
    X = attach_calibration(
        grid.drop(columns="rating_arousal"), frame, frame.video.unique()
    )
    model = CalibratedArousal(mode="all", minimum_sources=1, minimum_trials=3).fit(
        X, grid.rating_arousal
    )
    tagged = X.assign(day=2)
    alternative = CalibratedArousal(
        mode="all", minimum_sources=1, minimum_trials=3
    ).fit(tagged, grid.rating_arousal)
    np.testing.assert_allclose(model.predict(X), alternative.predict(tagged))
