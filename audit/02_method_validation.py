# %%
"""Check the completed synthetic validation table against its declared design."""

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
folder = Path(os.environ.get("HABS_SIM_RESULTS", ROOT / "results"))
if not folder.is_absolute():
    folder = ROOT / folder

# %%
config = json.loads((folder / "simulation_config.json").read_text())
records = pd.read_csv(folder / "simulation_replicates.csv")
if records.duplicated(["scenario", "repetition"]).any():
    raise ValueError("Duplicate simulation replications")
counts = records.groupby("scenario").size()
counts

# %%
for scenario in config["scenarios"]:
    if counts.get(scenario, 0) != config["target_per_scenario"]:
        raise ValueError(f"Incomplete validation for {scenario}")
records.groupby("scenario")[["gain_affine", "gain_video", "gain_missing"]].mean()

# %%
flags = (records.low_affine > 0) & (records.low_video > 0) & (records.low_missing > 0)
if not flags.equals(records.positive_after_controls):
    raise ValueError("Recorded control decision differs from the stored intervals")
{"replications": len(records), "scenarios": counts.to_dict(), "complete": True}
