"""Separate sampling uncertainty from reproducible personal carryover effects."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize

from habs.temporal.preprocess import CATEGORICAL, _numeric


@dataclass
class DayGeometry:
    """Retain the outcome-free least-squares geometry of one recorded day.

    Parameters
    ----------
    people : tuple of str
        Identical person order in both days.
    records : list of dict
        Chronologically ordered eligible transitions for this day.
    q, residual_history, influence, leverage : numpy.ndarray
        Nuisance basis, projected history, coefficient influence and leverage.
    rank, nuisance_columns, condition : int, int, float
        Identifiability diagnostics for nuisance and personal-history terms.
    """

    people: tuple
    records: list
    q: np.ndarray
    residual_history: np.ndarray
    influence: np.ndarray
    leverage: np.ndarray
    rank: int
    nuisance_columns: int
    condition: float

    def summary(self):
        """Return anonymous structural diagnostics.

        Returns
        -------
        dict
            Dimensions, rank and conditioning without empirical coefficients.
        """
        return {
            "rows": len(self.records),
            "people": len(self.people),
            "nuisance_columns": self.nuisance_columns,
            "nuisance_rank": self.q.shape[1],
            "personal_history_rank": self.rank,
            "personal_history_columns": 2 * len(self.people),
            "history_condition_number": self.condition,
            "high_leverage_rows": int(np.sum(self.leverage > 0.95)),
        }


def day_geometry(table, people, stats, reference_gap):
    """Project personal-history columns away from fixed stimulus/time effects.

    Parameters
    ----------
    table : pandas.DataFrame
        Eligible rows from one recorded day.
    people : sequence of str
        Common person order across the two days.
    stats : dict
        First-session normalization references.
    reference_gap : float
        Frozen first-session median gap.

    Returns
    -------
    DayGeometry
        Design-only linear influence matrix; current outcomes are not fitted.

    Raises
    ------
    ValueError
        If the conditional individual-history design is not identified.
    """
    people = tuple(people)
    positions = {p: i for i, p in enumerate(people)}
    records = table.sort_values(["participant_id", "event_order"]).to_dict("records")
    numeric = np.asarray([_numeric(r, stats, reference_gap) for r in records])
    blocks = [np.ones((len(records), 1)), numeric[:, 2:10]]
    for name in (*CATEGORICAL, "participant_id"):
        levels = sorted({str(r[name]) for r in records})
        encoding = {v: i for i, v in enumerate(levels)}
        block = np.zeros((len(records), len(levels)))
        for index, row in enumerate(records):
            block[index, encoding[str(row[name])]] = 1
        blocks.append(block)
    nuisance = np.column_stack(blocks)
    u, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    q = u[:, singular > singular[0] * 1e-10]
    history = np.zeros((len(records), 2 * len(people)))
    for index, row in enumerate(records):
        column = positions[row["participant_id"]]
        history[index, column] = numeric[index, 0]
        history[index, column + len(people)] = numeric[index, 1]
    residual = history - q @ (q.T @ history)
    information = residual.T @ residual
    eigenvalues = np.linalg.eigvalsh(information)
    rank = int(np.sum(eigenvalues > max(eigenvalues[-1] * 1e-10, 1e-12)))
    if rank != len(eigenvalues):
        raise ValueError(
            "Personal history is not identified after context: "
            f"rank {rank}/{len(eigenvalues)}."
        )
    inverse = np.linalg.pinv(information, rcond=1e-10)
    influence = inverse @ residual.T
    leverage = np.sum(q**2, axis=1) + np.sum(residual * influence.T, axis=1)
    return DayGeometry(
        people,
        records,
        q,
        residual,
        influence,
        leverage,
        rank,
        nuisance.shape[1],
        float(eigenvalues[-1] / eigenvalues[0]),
    )


def _score_covariance(scores, records, weights):
    result = scores @ scores.T
    by_person = {}
    for index, row in enumerate(records):
        by_person.setdefault(row["participant_id"], {})[int(row["event_order"])] = index
    for lag, weight in weights.items():
        pairs = [
            (index, group[order + lag])
            for group in by_person.values()
            for order, index in group.items()
            if order + lag in group
        ]
        if pairs:
            left, right = zip(*pairs, strict=True)
            cross = scores[:, left] @ scores[:, right].T
            result += weight * (cross + cross.T)
    result = (result + result.T) / 2
    values, vectors = np.linalg.eigh(result)
    floor = max(1e-12, values[-1] * 1e-10)
    adjusted = (vectors * np.maximum(values, floor)) @ vectors.T
    return adjusted, float(max(0, floor - values[0]))


def design_sampling_covariance(geometry, variance=2.0, serial=0.3):
    """Compute an assumed-noise covariance for outcome-free precision simulations.

    Parameters
    ----------
    geometry : DayGeometry
        Real design geometry without estimated current effects.
    variance : float, default=2
        Assumed variance of the standardized difference's measurement noise.
    serial : float, default=0.3
        Assumed event-order autocorrelation for this design scenario.

    Returns
    -------
    numpy.ndarray
        Sampling covariance under the simulated noise assumption.
    """
    scores = geometry.influence[: len(geometry.people)] * math.sqrt(variance)
    result, _ = _score_covariance(
        scores, geometry.records, {lag: serial**lag for lag in range(1, 13)}
    )
    return result


def estimate_day(geometry, *, hac_lags=2, hc3=True):
    """Estimate individual subjective/EEG slopes and their full sampling covariance.

    Parameters
    ----------
    geometry : DayGeometry
        Identified geometry and standardized current measurements for one day.
    hac_lags : int, default=2
        Maximum actual event-order distance for Bartlett covariance weights.
    hc3 : bool, default=True
        Apply leverage correction to residual score vectors.

    Returns
    -------
    dict
        Private person-ordered slopes, delta covariance and residual diagnostics.
    """
    outcomes = np.asarray([[r["arousal_z"], r["eeg_z"]] for r in geometry.records])
    coefficients = geometry.influence @ outcomes
    residual = (
        outcomes
        - geometry.q @ (geometry.q.T @ outcomes)
        - geometry.residual_history @ coefficients
    )
    difference_error = residual[:, 0] - residual[:, 1]
    if hc3:
        difference_error = difference_error / np.maximum(1 - geometry.leverage, 0.05)
    scores = geometry.influence[: len(geometry.people)] * difference_error
    covariance, adjustment = _score_covariance(
        scores,
        geometry.records,
        {lag: 1 - lag / (hac_lags + 1) for lag in range(1, hac_lags + 1)},
    )
    n = len(geometry.people)
    counts = np.asarray(
        [
            sum(r["participant_id"] == p for r in geometry.records)
            for p in geometry.people
        ]
    )
    return {
        "arousal_slopes": coefficients[:n, 0],
        "eeg_slopes": coefficients[:n, 1],
        "delta": coefficients[:n, 0] - coefficients[:n, 1],
        "sampling_covariance": covariance,
        "counts": counts,
        "covariance_eigenvalue_adjustment": adjustment,
        "residual_variance": np.var(residual, axis=0, ddof=1).tolist(),
    }


def _mean_design(counts, moderator, scale_ratio):
    n = counts.shape[1]
    result = np.zeros((2 * n, 2))
    result[:n, 0], result[n:, 1] = 1, 1
    if moderator:
        for day in (0, 1):
            values = np.log(counts[day])
            values -= np.mean(values)
            if np.std(values) > 1e-8:
                column = np.zeros((2 * n, 1))
                column[day * n : (day + 1) * n, 0] = values
                result = np.column_stack((result, column))
    if scale_ratio is not None and np.std(scale_ratio) > 1e-8:
        values = (np.asarray(scale_ratio) - np.mean(scale_ratio)) / np.std(scale_ratio)
        for day in (0, 1):
            column = np.zeros((2 * n, 1))
            column[day * n : (day + 1) * n, 0] = values
            result = np.column_stack((result, column))
    return result


def _marginal(parameters, sampling, rho_fixed):
    n = len(sampling) // 2
    sd = np.exp(parameters[:2])
    rho = float(np.tanh(parameters[2])) if rho_fixed is None else float(rho_fixed)
    covariance = float(rho * sd[0] * sd[1])
    matrix = sampling.copy()
    positions = np.arange(n)
    matrix[positions, positions] += sd[0] ** 2
    matrix[n + positions, n + positions] += sd[1] ** 2
    matrix[positions, n + positions] += covariance
    matrix[n + positions, positions] += covariance
    return matrix, sd, rho, covariance


def fit_variance_components(
    values,
    sampling,
    counts,
    *,
    rho_fixed=None,
    moderator=True,
    warm_start=None,
    scale_ratio=None,
):
    """Fit a two-day REML model with correlated sampling errors retained.

    Parameters
    ----------
    values : numpy.ndarray of shape (2 * n_people,)
        First-day delta estimates followed by second-day estimates.
    sampling : numpy.ndarray of shape (2 * n_people, 2 * n_people)
        Complete known/estimated sampling covariance, block diagonal by day.
    counts : numpy.ndarray of shape (2, n_people)
        Eligible transition counts for the length moderator.
    rho_fixed : float or None, default=None
        Fixed random-effect correlation for null or profile fitting.
    moderator : bool, default=True
        Include the prespecified day-specific log-count terms.
    warm_start : numpy.ndarray or None, default=None
        Optimization start from the same specified model in bootstrap/profile work.
    scale_ratio : numpy.ndarray of shape (n_people,) or None, default=None
        Known J1 arousal/EEG scale ratio, controlled separately in both days.

    Returns
    -------
    dict
        REML parameters, marginal covariance and convergence diagnostics.

    Raises
    ------
    RuntimeError
        If no finite converged optimization is obtained.
    """
    values, sampling, counts = (
        np.asarray(values),
        np.asarray(sampling),
        np.asarray(counts),
    )
    h = _mean_design(counts, moderator, scale_ratio)
    n = len(values) // 2

    def evaluate(parameters, details=False):
        matrix, sd, rho, covariance = _marginal(parameters, sampling, rho_fixed)
        try:
            factor = cho_factor(matrix, lower=True, check_finite=False)
            weighted_h = cho_solve(factor, h, check_finite=False)
            information = h.T @ weighted_h
            mean_factor = cho_factor(information, lower=True, check_finite=False)
            mean = cho_solve(mean_factor, weighted_h.T @ values, check_finite=False)
            residual = values - h @ mean
            objective = float(
                np.sum(np.log(np.diag(factor[0])))
                + np.sum(np.log(np.diag(mean_factor[0])))
                + 0.5 * residual @ cho_solve(factor, residual, check_finite=False)
            )
            if details:
                return objective, mean, matrix, sd, rho, covariance
            return objective
        except np.linalg.LinAlgError:
            return 1e30

    residual = values - h @ (np.linalg.pinv(h) @ values)
    moment = np.asarray(
        [
            math.sqrt(
                max(
                    1e-4,
                    float(
                        np.var(residual[d * n : (d + 1) * n], ddof=1)
                        - np.mean(np.diag(sampling)[d * n : (d + 1) * n])
                    ),
                )
            )
            for d in (0, 1)
        ]
    )
    length = 3 if rho_fixed is None else 2
    starts = [
        np.r_[np.log(moment), 0.0][:length],
        np.r_[np.log([0.1, 0.1]), 0.3][:length],
    ]
    if warm_start is not None:
        warm = np.asarray(warm_start)
        if len(warm) < length:
            warm = np.r_[warm, 0.0]
        starts = [warm[:length], *starts]
    bounds = [(-12, 2), (-12, 2)] + ([(-3, 3)] if length == 3 else [])
    solutions = [
        minimize(
            evaluate,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250, "ftol": 1e-10},
        )
        for start in starts
    ]
    valid = [s for s in solutions if s.success and np.isfinite(s.fun) and s.fun < 1e20]
    if not valid:
        fallback = minimize(
            evaluate,
            starts[0],
            method="Powell",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        if fallback.success and np.isfinite(fallback.fun):
            valid = [fallback]
    if not valid:
        raise RuntimeError("REML variance components did not converge.")
    best = min(valid, key=lambda s: s.fun)
    objective, mean, matrix, sd, rho, covariance = evaluate(best.x, details=True)
    return {
        "reml_objective": objective,
        "parameters": best.x,
        "mean_parameters": mean,
        "mean_design": h,
        "marginal_covariance": matrix,
        "sd": sd,
        "rho": rho,
        "covariance": covariance,
        "near_zero_variance": bool(min(sd) < 1e-3),
        "optimizer_success": bool(best.success),
        "optimizer_iterations": int(best.nit),
    }


def covariance_profile(values, sampling, counts, *, moderator=True, scale_ratio=None):
    """Profile the cross-day correlation without interpreting variance boundaries.

    Parameters
    ----------
    values, sampling, counts : numpy.ndarray
        Inputs to the two-day variance-component model.
    moderator : bool, default=True
        Include day-specific count effects.
    scale_ratio : numpy.ndarray or None, default=None
        First-session scale-ratio moderator.

    Returns
    -------
    dict
        Fitted model, null model and correlation likelihood profile.
    """
    null = fit_variance_components(
        values,
        sampling,
        counts,
        rho_fixed=0,
        moderator=moderator,
        scale_ratio=scale_ratio,
    )
    fitted = fit_variance_components(
        values,
        sampling,
        counts,
        moderator=moderator,
        scale_ratio=scale_ratio,
        warm_start=np.r_[null["parameters"], 0.0],
    )
    grid = sorted(
        set([-0.995, 0.995, 0.0, fitted["rho"], *[-0.99 + 0.05 * i for i in range(40)]])
    )
    profile = []
    for rho in grid:
        model = fit_variance_components(
            values,
            sampling,
            counts,
            rho_fixed=rho,
            moderator=moderator,
            warm_start=fitted["parameters"],
            scale_ratio=scale_ratio,
        )
        profile.append(
            (
                float(rho),
                max(0.0, 2 * (model["reml_objective"] - fitted["reml_objective"])),
            )
        )
    accepted = [rho for rho, distance in profile if distance <= 3.841458820694124]
    return {
        "model": fitted,
        "null": null,
        "profile": profile,
        "rho_profile95_grid": [min(accepted), max(accepted)],
        "correlation_weakly_identified": bool(
            fitted["near_zero_variance"] or max(accepted) - min(accepted) >= 1.5
        ),
        "likelihood_ratio": max(
            0.0, 2 * (null["reml_objective"] - fitted["reml_objective"])
        ),
    }


def bootstrap_components(
    values, sampling, counts, *, draws=999, seed=20261109, scale_ratio=None
):
    """Quantify covariance uncertainty and a boundary-aware model-based null.

    Parameters
    ----------
    values, sampling, counts : numpy.ndarray
        Empirical two-session model inputs.
    draws : int, default=999
        Draws from both fitted and zero-covariance models.
    seed : int, default=20261109
        Reproducible bootstrap seed.
    scale_ratio : numpy.ndarray or None, default=None
        Known first-session scale-ratio moderator.

    Returns
    -------
    dict
        Fitted/profile estimates, bootstrap interval and null tail probability.
    """
    profile = covariance_profile(values, sampling, counts, scale_ratio=scale_ratio)
    model, null = profile["model"], profile["null"]
    rng = np.random.default_rng(seed)
    root = np.linalg.cholesky(model["marginal_covariance"])
    null_root = np.linalg.cholesky(null["marginal_covariance"])
    means = model["mean_design"] @ model["mean_parameters"]
    null_means = null["mean_design"] @ null["mean_parameters"]
    covariances, null_statistics = [], []
    for index in range(draws):
        synthetic = means + root @ rng.normal(size=len(values))
        alt = fit_variance_components(
            synthetic,
            sampling,
            counts,
            warm_start=model["parameters"],
            scale_ratio=scale_ratio,
        )
        covariances.append(alt["covariance"])
        synthetic_null = null_means + null_root @ rng.normal(size=len(values))
        fit_null = fit_variance_components(
            synthetic_null,
            sampling,
            counts,
            rho_fixed=0,
            warm_start=null["parameters"],
            scale_ratio=scale_ratio,
        )
        fit_alt = fit_variance_components(
            synthetic_null,
            sampling,
            counts,
            warm_start=np.r_[fit_null["parameters"], 0.0],
            scale_ratio=scale_ratio,
        )
        null_statistics.append(
            max(0.0, 2 * (fit_null["reml_objective"] - fit_alt["reml_objective"]))
        )
        if (index + 1) % 100 == 0:
            print(f"Variance-component bootstrap {index + 1}/{draws}", flush=True)
    return {
        "covariance": model["covariance"],
        "covariance_ci95": np.quantile(covariances, [0.025, 0.975]).tolist(),
        "random_effect_sd": model["sd"].tolist(),
        "rho_point": model["rho"],
        "rho_profile95_grid": profile["rho_profile95_grid"],
        "rho_profile": profile["profile"],
        "correlation_weakly_identified": profile["correlation_weakly_identified"],
        "near_zero_variance": model["near_zero_variance"],
        "likelihood_ratio": profile["likelihood_ratio"],
        "null_bootstrap_p_two_sided": float(
            (1 + sum(v >= profile["likelihood_ratio"] for v in null_statistics))
            / (draws + 1)
        ),
        "bootstrap_draws": draws,
        "reml_objective": model["reml_objective"],
        "mean_parameters": model["mean_parameters"].tolist(),
        "conditioning": (
            "Normal random effects and fixed estimated sampling covariance, "
            "conditional on the video catalogue."
        ),
    }


def precision_simulation(geometries, *, draws=100, seed=20261109, scale_ratio=None):
    """Assess design precision using simulated slopes, never real current effects.

    Parameters
    ----------
    geometries : sequence of DayGeometry
        Outcome-free J1/J2 geometries in common person order.
    draws : int, default=100
        Replicates per prespecified between-person scenario.
    seed : int, default=20261109
        Simulation seed.
    scale_ratio : numpy.ndarray or None, default=None
        Known J1 scale ratio controlled in each fitted model.

    Returns
    -------
    dict
        Clearly labelled simulated precision and boundary diagnostics.
    """
    n = len(geometries[0].people)
    sampling = block_diag(*(design_sampling_covariance(g) for g in geometries))
    counts = np.asarray(
        [
            [sum(r["participant_id"] == p for r in g.records) for p in g.people]
            for g in geometries
        ]
    )
    rng = np.random.default_rng(seed)
    scenarios = []
    for sd, rho, kind in (
        (0.0, 0.0, "random_effect"),
        (0.15, 0.0, "random_effect"),
        (0.15, 0.6, "random_effect"),
        (0.3, 0.0, "random_effect"),
        (0.3, 0.6, "random_effect"),
        (0.5, 0.0, "random_effect"),
        (0.5, 0.6, "random_effect"),
        (0.0, 0.0, "shared_scale_only"),
    ):
        model_cov = sampling.copy()
        indices = np.arange(n)
        model_cov[indices, indices] += sd**2
        model_cov[n + indices, n + indices] += sd**2
        model_cov[indices, n + indices] += rho * sd**2
        model_cov[n + indices, indices] += rho * sd**2
        root = np.linalg.cholesky(model_cov)
        estimates, observed_correlations = [], []
        mean = np.full(2 * n, 0.2)
        if (
            kind == "shared_scale_only"
            and scale_ratio is not None
            and np.std(scale_ratio) > 1e-8
        ):
            scale = (scale_ratio - np.mean(scale_ratio)) / np.std(scale_ratio)
            mean += np.r_[0.3 * scale, 0.3 * scale]
        for _ in range(draws):
            synthetic = mean + root @ rng.normal(size=2 * n)
            fit = fit_variance_components(
                synthetic, sampling, counts, scale_ratio=scale_ratio
            )
            estimates.append((fit["covariance"], fit["rho"], fit["near_zero_variance"]))
            observed_correlations.append(
                float(np.corrcoef(synthetic[:n], synthetic[n:])[0, 1])
            )
        values = np.asarray(estimates)
        scenarios.append(
            {
                "scenario": kind,
                "true_random_sd": sd,
                "true_correlation": rho if sd else None,
                "true_covariance": rho * sd**2,
                "estimated_covariance_quantiles": np.quantile(
                    values[:, 0], [0.025, 0.5, 0.975]
                ).tolist(),
                "estimated_correlation_quantiles": np.quantile(
                    values[:, 1], [0.025, 0.5, 0.975]
                ).tolist(),
                "near_zero_variance_fraction": float(np.mean(values[:, 2])),
                "raw_correlation_quantiles": np.quantile(
                    observed_correlations, [0.025, 0.5, 0.975]
                ).tolist(),
                "replicates": draws,
            }
        )
        print(f"Synthetic precision: SD={sd}, correlation={rho}, n={n}", flush=True)
    return {
        "status": "simulated_design_precision_complete",
        "empirical_current_effects_fitted": False,
        "people": n,
        "geometry": [g.summary() for g in geometries],
        "assumed_difference_noise_variance": 2.0,
        "assumed_serial_correlation": 0.3,
        "sampling_standard_error_quantiles": [
            np.quantile(
                np.sqrt(np.diag(sampling)[d * n : (d + 1) * n]), [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            for d in (0, 1)
        ],
        "scenarios": scenarios,
        "warning": (
            "Synthetic design diagnostic, not empirical evidence "
            "or a calibrated power guarantee."
        ),
    }
