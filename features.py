"""Extract short-trial electrodermal features and audit their measurement support."""

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

from starting_notebooks.bdf_loader import uV_to_uS

FEATURES = ["eda_level", "eda_change", "eda_spread"]
RAW_FEATURES = ["voltage_level", "voltage_change", "voltage_spread"]


def describe_window(values, rate):
    """Return level, end-minus-start change and interquartile spread."""
    early, late = values[: round(3 * rate)], values[round(7 * rate) :]
    return [
        float(np.median(values)),
        float(np.median(late) - np.median(early)),
        float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    ]


def extract_session(task):
    """Read one GSR channel once, retaining one row per supplied annotation."""
    data_dir, person, day, rows = task
    folder = data_dir / "CURATED" / person
    path = folder / f"{person}_day-{day}_metadata.json"
    result = [{**r, "join_status": "missing_metadata"} for r in rows]
    if not path.exists():
        return result
    metadata = json.loads(path.read_text())
    segment = metadata["dataset_export"].get("segments", {}).get("emotion-video")
    if not segment or not (folder / segment["file"]).exists():
        for row in result:
            row["join_status"] = "missing_recording"
        return result
    events = defaultdict(list)
    videos = [e for e in metadata["events"] if e.get("trial_type") == "emotion_video"]
    videos.sort(key=lambda e: e["onset_s"])
    for order, event in enumerate(videos):
        events[int(event["marker_id"])].append((order, event))
    with pyedflib.EdfReader(str(folder / segment["file"])) as reader:
        labels = reader.getSignalLabels()
        if "GSR" not in labels:
            for row in result:
                row["join_status"] = "missing_gsr"
            return result
        index = labels.index("GSR")
        rate = reader.getSampleFrequency(index)
        unit = reader.getPhysicalDimension(index)
        physical = reader.readSignal(index)
    if unit not in ("uV", "µV"):
        raise ValueError(f"Unrecognized GSR header unit: {unit!r}")
    voltage = physical / 1e6
    conductance = uV_to_uS(physical)
    for row in result:
        matches = events.get(int(row["start_marker"]), [])
        row["join_status"] = "missing_event" if not matches else "ambiguous_event"
        if len(matches) != 1:
            continue
        order, event = matches[0]
        if Path(row["video"]).name != Path(event.get("media_file") or "").name:
            row["join_status"] = "video_mismatch"
            continue
        onset = event["onset_s"] - segment["metadata_time_origin_s"]
        start, stop = round(onset * rate), round((onset + 10) * rate)
        if (
            onset < 0
            or onset + 10 > segment["duration_s"] + 0.01
            or start < 0
            or stop > len(conductance)
            or abs(event["duration_s"] - 10) > 0.01
        ):
            row["join_status"] = "outside_segment"
            continue
        values, volts = conductance[start:stop], voltage[start:stop]
        row.update(
            join_status="matched",
            order=order / max(1, len(videos) - 1),
            finite_fraction=float(np.isfinite(values).mean()),
            clamp_fraction=float(((values <= 1) | (values >= 40)).mean()),
            raw_min=float(np.min(volts)),
            raw_max=float(np.max(volts)),
            flat=bool(np.max(values) == np.min(values)),
            raw_flat=bool(np.max(volts) == np.min(volts)),
        )
        if row["finite_fraction"] < 0.99:
            continue
        # Interpolate only isolated non-finite samples within an otherwise valid trial.
        finite = np.isfinite(values)
        positions = np.arange(len(values))
        values = np.interp(positions, positions[finite], values[finite])
        volts = np.interp(positions, positions[finite], volts[finite])
        row.update(dict(zip(FEATURES, describe_window(values, rate), strict=True)))
        row.update(dict(zip(RAW_FEATURES, describe_window(volts, rate), strict=True)))
    return result


def load_trials(data_dir):
    """Read released annotations without replacing or reconstructing labels."""
    return pd.read_csv(Path(data_dir) / "labels_train.csv", na_values=["TESTSUBJECT"])


def extract(data_dir, out_dir, workers=4):
    """Save cohort-2 trial features privately and print aggregate quality counts.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only HABS dataset root.
    out_dir : pathlib.Path
        Private result directory, retained by the HABS platform.
    workers : int
        Maximum concurrent recording reads.

    Returns
    -------
    pandas.DataFrame
        Trial rows with labels, deterministic features and quality indicators.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_trials(data_dir)
    subset = labels.loc[(labels.cohort == 2) & labels.day.isin([1, 2])].copy()
    columns = [
        "participant_id",
        "day",
        "video",
        "start_marker",
        "rating_arousal",
        "expected_emotion_label",
        "expected_emotion_source",
    ]
    subset = subset[columns]
    if subset.duplicated(["participant_id", "day", "start_marker"]).any():
        raise ValueError("Non-unique annotation keys")
    tasks = [
        (data_dir, str(person), int(day), group.to_dict("records"))
        for (person, day), group in subset.groupby(["participant_id", "day"])
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for position, batch in enumerate(pool.map(extract_session, tasks), 1):
            rows.extend(batch)
            if position % 25 == 0:
                print(f"Recordings processed: {position}/{len(tasks)}", flush=True)
    frame = pd.DataFrame(rows)
    for name in [
        *FEATURES,
        *RAW_FEATURES,
        "finite_fraction",
        "clamp_fraction",
        "order",
    ]:
        if name not in frame:
            frame[name] = np.nan
    frame["flat"] = frame.get("flat", pd.Series(True, index=frame.index)).fillna(True)
    frame["raw_flat"] = frame.get(
        "raw_flat", pd.Series(True, index=frame.index)
    ).fillna(True)
    frame.to_csv(out_dir / "trials.csv.gz", index=False, compression="gzip")
    summary = {
        "cohort": 2,
        "rows": len(frame),
        "recordings": len(tasks),
        "joins": dict(Counter(frame.join_status)),
        "labels_sha256": hashlib.sha256(
            (data_dir / "labels_train.csv").read_bytes()
        ).hexdigest(),
        "quality": {},
        "measurement": {},
    }
    matched = frame.loc[frame.join_status == "matched"]
    for threshold in (0.01, 0.05, 0.10, 1.01):
        valid = matched.loc[
            (matched.finite_fraction >= 0.99)
            & ~matched.flat
            & (matched.clamp_fraction < threshold)
            & matched.rating_arousal.between(1, 9)
        ]
        counts = (
            valid.groupby(["participant_id", "day"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[1, 2], fill_value=0)
        )
        paired = counts.loc[(counts.get(1, 0) >= 20) & (counts.get(2, 0) >= 20)]
        summary["quality"][str(threshold)] = {
            "valid_trials": len(valid),
            "paired_people": len(paired),
            "paired_trials": int(paired.to_numpy().sum()),
            "day1": int(paired.get(1, pd.Series(dtype=int)).sum()),
            "day2": int(paired.get(2, pd.Series(dtype=int)).sum()),
        }
    for name in ["clamp_fraction", "raw_min", "raw_max", *FEATURES]:
        summary["measurement"][name] = np.quantile(
            matched[name].dropna(), [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
    summary["measurement"]["flat_trials"] = int(matched.flat.sum())
    summary["measurement"]["raw_flat_trials"] = int(matched.raw_flat.sum())
    (out_dir / "quality.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("HABS_QUALITY=" + json.dumps(summary, separators=(",", ":")), flush=True)
    return frame
