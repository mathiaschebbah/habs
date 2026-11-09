"""Count consecutive rated CURATED-EEG trials before fitting sequential effects."""

import json
import os
from pathlib import Path

import skrub

from habs.temporal.data import (
    build_transition_graph,
    feasibility_summary,
    verify_inputs,
)


def main():
    """Export structural support while keeping individual observations private.

    Returns
    -------
    None
        Private transitions and publication-safe feasibility counts are written.
    """
    out = Path(os.environ.get("HABS_ANALYSIS_OUT", "/out"))
    source = Path(os.environ.get("PRIVATE_INPUT_DEST", out / "data"))
    public = out / "public"
    public.mkdir(parents=True, exist_ok=True)
    verify_inputs(source)
    table = build_transition_graph().skb.eval(
        {
            "features_path": source / "features.csv.gz",
            "events_path": source / "recorded_events.csv.gz",
        }
    )
    table.to_csv(source / "temporal_transitions.csv.gz", index=False)
    report = skrub.TableReport(
        table,
        title="Consecutive HABS video trials",
        verbose=0,
        compute_associations=False,
    )
    report.write_html(source / "eda_temporal_private.html")
    facts = json.loads(report.json())
    summary = feasibility_summary(table, source)
    summary["overview"] = {
        "rows": facts.get("n_rows"),
        "columns": [
            {k: c.get(k) for k in ("name", "dtype", "null_proportion", "n_unique")}
            for c in facts.get("columns", [])
        ],
    }
    (public / "temporal_feasibility.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
