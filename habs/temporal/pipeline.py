"""Declare past-bound temporal forecasting with an early prediction-grid marker."""

import skrub

from habs.data import load_table
from habs.temporal.model import TemporalForecaster
from habs.temporal.preprocess import (
    calibration_frame,
    grid_target,
    predictor_frame,
)


def build_learner(
    *,
    splitter,
    reference_gap,
    mode="full",
    grid_preview=None,
    population_alpha=10.0,
    personal_penalty=20.0,
):
    """Declare a forecaster with raw history sources separate from the query grid.

    Parameters
    ----------
    splitter : sklearn cross-validator
        Participant-grouped split supplied by the evaluation caller.
    reference_gap : float
        Frozen J1 median inter-video gap.
    mode : str, default="full"
        Full, exact ablation or refitted restricted personal calibration.
    grid_preview : str or pathlib.Path or None, default=None
        Optional preview path; production binds every source explicitly.
    population_alpha, personal_penalty : float, default=10, 20
        Frozen population and personal regularization settings.

    Returns
    -------
    skrub.SkrubLearner
        Unfitted graph consuming grid_path, features_path and events_path.
    """
    root = (
        skrub.var("grid_path", value=str(grid_preview))
        if grid_preview is not None
        else skrub.var("grid_path")
    )
    grid = root.skb.apply_func(load_table, no_cache=True)
    raw = skrub.var("features_path").skb.apply_func(load_table, no_cache=True)
    events = skrub.var("events_path").skb.apply_func(load_table, no_cache=True)
    marked = grid.skb.mark_as_X(
        cv=splitter, split_kwargs={"groups": grid["participant_id"]}
    )
    predictors = marked.skb.apply_func(predictor_frame, raw, events, reference_gap)
    target = grid.skb.apply_func(grid_target, raw).skb.mark_as_y()
    calibration = raw.skb.apply_func(calibration_frame, events, reference_gap)
    predictions = predictors.skb.apply(
        TemporalForecaster(
            mode=mode,
            population_alpha=population_alpha,
            personal_penalty=personal_penalty,
        ),
        y=target,
        fit_kwargs={"calibration": calibration},
        predict_kwargs={"calibration": calibration},
    )
    return predictions.skb.make_learner()
