"""Reconcile annotations with unique recorded events without using rating agreement."""

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def video_name(value):
    """Normalize path separators and Unicode without changing the video identity."""
    return unicodedata.normalize(
        "NFC", str(value).replace("\\", "/").split("/")[-1]
    ).lower()


def event_index(data_dir):
    """Index all recorded video events, coalescing raw and preprocessed copies."""
    events = {}
    for source in ("CURATED", "PREPROCESSED"):
        for path in sorted((Path(data_dir) / source).glob("P*/*metadata.json")):
            metadata = json.loads(path.read_text())
            export = metadata["dataset_export"]
            person = export["participant_id"]
            day = int(str(export["day"]).replace("day", ""))
            segment = export.get("segments", {}).get("emotion-video")
            if not segment or not (path.parent / segment["file"]).exists():
                continue
            videos = sorted(
                (
                    e
                    for e in metadata["events"]
                    if e.get("trial_type") == "emotion_video"
                ),
                key=lambda e: e["onset_s"],
            )
            for order, event in enumerate(videos):
                name = video_name(event.get("media_file") or "")
                onset = float(event["onset_s"])
                key = (person, day, int(event["marker_id"]), name, round(onset, 6))
                identity = "|".join(map(str, key))
                record = events.setdefault(
                    key,
                    {
                        "participant_id": person,
                        "day": day,
                        "start_marker": int(event["marker_id"]),
                        "video": name,
                        "event_uid": hashlib.sha256(identity.encode()).hexdigest()[:20],
                        "order": order / max(1, len(videos) - 1),
                        "order_index": order,
                        "duration": float(event["duration_s"]),
                        "variants": {},
                    },
                )
                relative = onset - float(segment["metadata_time_origin_s"])
                record["variants"][source] = {
                    "path": str((path.parent / segment["file"]).relative_to(data_dir)),
                    "onset": relative,
                    "segment_duration": float(segment["duration_s"]),
                    "bounded": relative >= -0.001
                    and relative + 10 <= segment["duration_s"] + 0.0011,
                }
    index = defaultdict(list)
    for record in events.values():
        index[
            (record["participant_id"], record["start_marker"], record["video"])
        ].append(record)
    return index


def reconcile(data_dir, out_dir):
    """Match notes to actual sessions using only participant, marker and video.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only release root.
    out_dir : pathlib.Path
        Private output directory for the complete reconciled table.

    Returns
    -------
    pandas.DataFrame
        Every numeric arousal annotation with explicit matching provenance.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data_dir / "labels_train.csv", na_values=["TESTSUBJECT"])
    labels = labels.loc[labels.rating_arousal.between(1, 9)].copy()
    index = event_index(data_dir)
    rows = []
    for row in labels.to_dict("records"):
        original_day = row["day"]
        key = (
            row["participant_id"],
            int(row["start_marker"]),
            video_name(row["video"]),
        )
        candidates = index.get(key, [])
        exact = [event for event in candidates if event["day"] == original_day]
        event = (
            exact[0]
            if len(exact) == 1
            else candidates[0]
            if not exact and len(candidates) == 1
            else None
        )
        row["label_day"] = original_day
        row["event_uid"] = None
        row["signal_source"] = None
        row["alignment"] = "unmatched" if not candidates else "ambiguous"
        if event is not None:
            variants = [
                (source, info)
                for source, info in event["variants"].items()
                if info["bounded"]
            ]
            row["alignment"] = "outside_segment"
            if variants and abs(event["duration"] - 10) <= 0.001:
                source, info = variants[0]
                row.update(
                    day=event["day"],
                    video=event["video"],
                    event_uid=event["event_uid"],
                    order=event["order"],
                    order_index=event["order_index"],
                    recording=info["path"],
                    onset=max(0.0, info["onset"]),
                    signal_source=source,
                    variants=json.dumps(event["variants"]),
                    alignment="exact"
                    if exact
                    else "day_recovered"
                    if pd.isna(original_day)
                    else "day_corrected",
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    linked = frame.event_uid.notna()
    collision = linked & frame.event_uid.duplicated(keep=False)
    frame.loc[collision, "alignment"] = "duplicate_annotation"
    frame.loc[collision, "event_uid"] = None
    frame.to_csv(out_dir / "aligned_trials.csv.gz", index=False, compression="gzip")
    accepted = frame.loc[
        frame.alignment.isin(["exact", "day_corrected", "day_recovered"])
    ]
    summary = {
        "rows": len(frame),
        "people": int(frame.participant_id.nunique()),
        "alignment": dict(Counter(frame.alignment)),
        "cohorts": {},
    }
    for cohort, original in frame.groupby("cohort"):
        retained = accepted.loc[accepted.cohort == cohort]
        sizes = retained.groupby(["participant_id", "day"]).size()
        days = sizes.loc[sizes >= 20].groupby("participant_id").size()
        summary["cohorts"][str(int(cohort))] = {
            "people_with_notes": int(original.participant_id.nunique()),
            "alignment": dict(Counter(original.alignment)),
            "matched_people": int(retained.participant_id.nunique()),
            "matched_trials": len(retained),
            "at_least_two_days_20_trials": int((days >= 2).sum()),
            "at_least_one_day_40_trials": int(
                sizes.loc[sizes >= 40].index.get_level_values(0).nunique()
            ),
            "label_day_vs_recorded_day": {
                f"{label}->{actual}": int(n)
                for (label, actual), n in retained.groupby(["label_day", "day"])
                .size()
                .items()
            },
            "sources": dict(Counter(retained.signal_source)),
        }
    (out_dir / "alignment.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("HABS_ALIGNMENT=" + json.dumps(summary, separators=(",", ":")), flush=True)
    return frame
