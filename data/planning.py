"""Define the individual-variation score and its planning sensitivity."""

import argparse
import csv
import math
from pathlib import Path
from statistics import NormalDist, mean


def centered_mse(observed, predicted):
    """Score prediction fluctuations without rewarding a constant offset.

    Parameters
    ----------
    observed : sequence of float
        Held-out ratings for one participant.
    predicted : sequence of float
        Frozen predictions for the same trials, in the same order.

    Returns
    -------
    float
        Mean squared error after centering the errors within this participant.
    """
    if len(observed) != len(predicted) or len(observed) < 2:
        raise ValueError("Provide at least two paired observations.")
    if not all(math.isfinite(x) for x in [*observed, *predicted]):
        raise ValueError("Scores require finite observations and predictions.")
    errors = [y - predicted[index] for index, y in enumerate(observed)]
    offset = mean(errors)
    return mean((error - offset) ** 2 for error in errors)


def precision_table(destination):
    """Write normal-approximation scenarios, not estimated HABS power.

    Parameters
    ----------
    destination : pathlib.Path
        CSV destination for assumed between-participant gain dispersions.
    """
    normal = NormalDist()
    critical = normal.inv_cdf(0.975)
    rows = []
    for participants in (20, 25, 30, 50, 75):
        for sigma in (0.10, 0.25, 0.50, 1.00):
            se = sigma / math.sqrt(participants)
            for delta in (0.0, 0.05, 0.10, 0.20):
                shift = delta / se
                power = normal.cdf(-critical - shift) + 1 - normal.cdf(critical - shift)
                rows.append(
                    {
                        "test_participants": participants,
                        "assumed_sd_individual_gain": sigma,
                        "assumed_mean_gain": delta,
                        "normal_95_half_width": critical * se,
                        "normal_80pct_detectable_gain": (critical + normal.inv_cdf(0.8))
                        * se,
                        "normal_two_sided_power": power,
                    }
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    precision_table(arguments.destination)
