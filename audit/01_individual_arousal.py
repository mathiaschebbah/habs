# %%
"""Read the stored primary report and the participant-level study contrasts."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if not (ROOT / "model.py").exists():
    raise RuntimeError("Run the audit from the repository root")
sys.path.insert(0, str(ROOT))

from skore._plugins.local.project import read_report

# %%
result_dir = Path(os.environ.get("HABS_RESULTS", ROOT / "out/study"))
if not result_dir.is_absolute():
    result_dir = ROOT / result_dir
workspace = result_dir / "reports"
if not (workspace / ".SKORE_WORKSPACE").exists():
    raise FileNotFoundError(f"No persisted study report in {workspace}")
report_paths = sorted(
    (workspace / "projects/habs-arousal/reports").glob("*__id_*__01_individual_arousal")
)
if not report_paths:
    raise LookupError("The primary arousal report is absent")
report_path = report_paths[-1]
metadata = json.loads((report_path / "metadata.json").read_text())
{key: metadata[key] for key in ("report_id", "name", "creation-date")}

# %%
# Skore 0.29.1 initializes storage in Project(); its report reader avoids writes.
report = read_report(report_path)
metrics = report.metrics.summarize().frame()
metrics

# %%
# Expensive alternative-model diagnostics are not used to select new models.
checks = report.checks.summarize(fast_mode=True).frame()
checks

# %%
outcomes = json.loads((result_dir / "results.json").read_text())
primary = outcomes["cohorts"]["2"]
primary["effects_vs_affine"]["all"]

# %%
# Skore metrics pool trials; the main study gives equal weight to each person.
audit = {
    "pooled_metrics": json.loads(metrics.to_json()),
    "checks": json.loads(checks.to_json(orient="records")),
    "participant_weighted_metrics": primary["models"]["all"],
    "primary_gain": primary["effects_vs_affine"]["all"],
    "control_comparisons": primary["individual_signal_controls"],
    "frozen_plan_sha256": primary["frozen_plan_sha256"],
}
if __name__ == "__main__":
    print("HABS_AUDIT=" + json.dumps(audit, separators=(",", ":")))
audit
