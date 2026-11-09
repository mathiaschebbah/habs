"""Assess identifiability and simulated precision before fitting real effects."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from habs.data import load_table
from habs.temporal.data import verify_inputs
from habs.temporal.inference import day_geometry, precision_simulation
from habs.temporal.preprocess import prepare_data


def main():
    """Write a design-only freeze and clearly labelled synthetic diagnostics.

    Returns
    -------
    None
        Publication-safe structure and simulated precision JSON files are exported.
    """
    out = Path(os.environ.get("HABS_ANALYSIS_OUT", "/out"))
    source = Path(os.environ.get("PRIVATE_INPUT_DEST", out / "data"))
    public = out / "public"
    public.mkdir(parents=True, exist_ok=True)
    verify_inputs(source)
    features, events = (
        load_table(source / "features.csv.gz"),
        load_table(source / "recorded_events.csv.gz"),
    )
    table, stats, metadata = prepare_data(features, events)
    people = sorted(set(table["participant_id"]))
    geometries = [
        day_geometry(
            table.loc[table["recorded_day"] == d],
            people,
            stats,
            metadata["reference_gap_s"],
        )
        for d in (1, 2)
    ]
    scale_ratio = np.asarray([stats[p]["sd_a"] / stats[p]["sd_p"] for p in people])
    report = precision_simulation(geometries, draws=100, scale_ratio=scale_ratio)
    package = Path(__file__).resolve().parents[1] / "habs/temporal"
    names = ("data.py", "preprocess.py", "inference.py")
    code_hash = hashlib.sha256(
        b"".join(n.encode() + (package / n).read_bytes() for n in names)
    ).hexdigest()
    metadata.update(
        {
            "empirical_current_effects_fitted": False,
            "person_selection_sha256": hashlib.sha256(
                "\n".join(people).encode()
            ).hexdigest(),
            "pair_grid_sha256": hashlib.sha256(
                "\n".join(table["trial_id"]).encode()
            ).hexdigest(),
            "geometry_inference_code_sha256": code_hash,
            "geometry": [g.summary() for g in geometries],
            "input_features_sha256": hashlib.sha256(
                (source / "features.csv.gz").read_bytes()
            ).hexdigest(),
        }
    )
    (public / "temporal_structure.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    (public / "temporal_precision.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    print(
        json.dumps(
            {"structure": metadata, "precision_scenarios": report["scenarios"]},
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
