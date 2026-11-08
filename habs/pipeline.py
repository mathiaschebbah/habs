"""Declare source-bound signal materialization and calibrated prediction graphs."""

import skrub

from habs.data import load_table, target_values
from habs.features import extract_signals
from habs.model import INPUT_COLUMNS, CalibratedRegressor


def build_feature_graph():
    """Declare deterministic signal extraction from reconciled physical trials.

    Returns
    -------
    skrub.DataOp
        Graph bound by aligned_path, raw_dir and cache_dir identifiers.
    """
    trials = skrub.var("aligned_path").skb.apply_func(load_table)
    return trials.skb.apply_func(
        extract_signals, skrub.var("raw_dir"), skrub.var("cache_dir")
    )


def build_learner(data_dir_preview=None, *, splitter, **parameters):
    """Declare a predictor with a distinct first-session calibration source.

    Parameters
    ----------
    data_dir_preview : str or pathlib.Path or None, default=None
        Optional feature-table path for interactive preview; never CWD-relative.
    splitter : sklearn cross-validator
        Explicit participant/session splitter supplied by the evaluation caller.
    **parameters : dict
        Parameters of CalibratedRegressor.

    Returns
    -------
    skrub.SkrubLearner
        Unfitted learner consuming trials_path and calibration_path bindings.
    """
    source = (
        skrub.var("trials_path", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("trials_path")
    )
    data = source.skb.apply_func(load_table)
    history = skrub.var("calibration_path").skb.apply_func(load_table)
    target = parameters.get("target", "score")
    columns = list(INPUT_COLUMNS)
    if parameters.get("conditional_arousal", False):
        columns.append("observed_arousal")
    X = data.skb.select(columns).skb.mark_as_X(
        cv=splitter, split_kwargs={"groups": data["participant_id"]}
    )
    y = data.skb.apply_func(target_values, target).skb.mark_as_y()
    predictions = X.skb.apply(
        CalibratedRegressor(**parameters),
        y=y,
        fit_kwargs={"calibration": history},
        predict_kwargs={"calibration": history},
    )
    return predictions.skb.make_learner()
