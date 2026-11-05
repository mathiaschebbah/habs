"""Measure annotation coverage and trial alignment for the HABS study."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def number(value):
    """Return a finite number, or None for a missing or masked value."""
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def counts(values):
    """Count values without disclosing individual annotations."""
    return dict(sorted(Counter(str(v) for v in values).items()))


def extent(values):
    """Summarize numeric support, retaining missingness separately."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": values[0],
        "max": values[-1],
        "median": values[len(values) // 2],
        "mean": sum(values) / len(values),
    }


def inventory(data_dir, out_dir):
    """Write aggregate coverage and metadata alignment without fitting models.

    Parameters
    ----------
    data_dir : pathlib.Path
        Read-only dataset root containing CSV tables and session metadata.
    out_dir : pathlib.Path
        Destination for the aggregate JSON report.

    Returns
    -------
    dict
        Annotation coverage, matched trial counts and file provenance.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "inputs": {}, "cohorts": {}}
    tables = {}
    for name in ("labels_train.csv", "participants_train.csv", "markers.csv"):
        path = data_dir / name
        if path.exists():
            payload = path.read_bytes()
            with path.open(encoding="utf-8-sig", newline="") as stream:
                tables[name] = list(csv.DictReader(stream))
            report["inputs"][name] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "rows": len(tables[name]),
                "columns": list(tables[name][0]) if tables[name] else [],
            }
    labels = tables.get("labels_train.csv", [])
    report["sample_only"] = not bool(labels)
    report["all_rows"] = len(labels)
    report["masked_rows"] = sum(
        any(v == "TESTSUBJECT" for v in r.values()) for r in labels
    )
    valid = [r for r in labels if number(r.get("rating_arousal")) is not None]
    report["arousal_rows"] = len(valid)
    report["arousal_participant_ids"] = len({r["participant_id"] for r in valid})
    report["flags_all"] = counts(r.get("flagged_outliers") for r in labels)
    report["final_label_missing_with_arousal"] = sum(
        r.get("final_emotion_label") in (None, "", "TESTSUBJECT") for r in valid
    )
    keys = [(r["participant_id"], r["day"], r["start_marker"]) for r in labels]
    report["duplicate_trial_keys"] = sum(n - 1 for n in Counter(keys).values())
    events = defaultdict(list)
    available = set()
    meta_totals = Counter()
    for path in sorted((data_dir / "CURATED").glob("P*/*metadata.json")):
        obj = json.loads(path.read_text())
        export = obj["dataset_export"]
        participant = export["participant_id"]
        day = number(str(export["day"]).replace("day", ""))
        segment = export.get("segments", {}).get("emotion-video")
        meta_totals["sessions"] += 1
        if not segment:
            continue
        exists = (path.parent / segment["file"]).is_file()
        origin = number(segment.get("metadata_time_origin_s"))
        duration = number(segment.get("duration_s"))
        if exists:
            available.add((participant, day))
        for event in obj.get("events", []):
            if event.get("trial_type") != "emotion_video":
                continue
            meta_totals["video_events"] += 1
            onset, length = (
                number(event.get("onset_s")),
                number(event.get("duration_s")),
            )
            bounded = all(v is not None for v in (onset, length, origin, duration))
            bounded = bounded and onset >= origin - 0.01
            bounded = bounded and onset + length <= origin + duration + 0.01
            events[(participant, day, number(event.get("marker_id")))].append(
                {
                    "media": Path(event.get("media_file") or "").name,
                    "bounded": bounded,
                    "bdf_exists": exists,
                    "duration": length,
                    "arousal": number(event.get("arousal_score")),
                    "rating_tool": event.get("rating_tool"),
                }
            )
    report["metadata"] = dict(meta_totals)
    report["available_emotion_sessions"] = len(available)
    joins, joined = Counter(), []
    durations, metadata_tools, rating_deltas = [], [], []
    for row in valid:
        key = (row["participant_id"], number(row["day"]), number(row["start_marker"]))
        candidates = events.get(key, [])
        if len(candidates) != 1:
            joins["no_event" if not candidates else "ambiguous_event"] += 1
            continue
        event = candidates[0]
        if Path(row["video"]).name != event["media"]:
            joins["video_mismatch"] += 1
            continue
        if not event["bdf_exists"] or not event["bounded"]:
            joins["missing_bdf_or_outside_segment"] += 1
            continue
        joins["matched"] += 1
        joined.append(row)
        durations.append(event["duration"])
        metadata_tools.append(event["rating_tool"])
        if event["arousal"] is not None:
            rating_deltas.append(abs(event["arousal"] - number(row["rating_arousal"])))
    report["joins"] = dict(joins)
    report["video_duration_s"] = extent(durations)
    report["metadata_rating_tools_matched"] = counts(metadata_tools)
    report["metadata_csv_arousal_abs_difference"] = extent(rating_deltas)
    for cohort in sorted({r["cohort"] for r in labels}):
        rows = [r for r in labels if r["cohort"] == cohort]
        rated = [r for r in valid if r["cohort"] == cohort]
        matched = [r for r in joined if r["cohort"] == cohort]
        by_person = defaultdict(lambda: defaultdict(list))
        video_people = defaultdict(set)
        for row in matched:
            by_person[row["participant_id"]][number(row["day"])].append(row)
            video_people[row["video"]].add(row["participant_id"])
        both = {p: days for p, days in by_person.items() if 1 in days and 2 in days}
        eligible = {
            p: days
            for p, days in both.items()
            if len(days[1]) >= 20 and len(days[2]) >= 20
        }
        repeated = [
            len({r["video"] for r in d[1]} & {r["video"] for r in d[2]})
            for d in both.values()
        ]
        report["cohorts"][cohort] = {
            "rows": len(rows),
            "rated_rows": len(rated),
            "rated_participant_ids": len({r["participant_id"] for r in rated}),
            "rated_by_day": counts(r["day"] for r in rated),
            "matched_rows": len(matched),
            "matched_people": len(by_person),
            "both_days_people": len(both),
            "at_least_20_each_day_people": len(eligible),
            "eligible_day1_rows": sum(len(d[1]) for d in eligible.values()),
            "eligible_day2_rows": sum(len(d[2]) for d in eligible.values()),
            "matched_videos": len(video_people),
            "people_per_video": extent([len(s) for s in video_people.values()]),
            "videos_at_least_10_people": sum(
                len(s) >= 10 for s in video_people.values()
            ),
            "repeated_videos_between_days": extent(repeated),
            "expected_source": counts(
                r.get("expected_emotion_source") for r in matched
            ),
            "arousal": extent([number(r["rating_arousal"]) for r in rated]),
            "integer_arousal_fraction": sum(
                number(r["rating_arousal"]).is_integer() for r in rated
            )
            / max(1, len(rated)),
        }
    report["raw_signals_read"] = False
    (out_dir / "inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("HABS_INVENTORY=" + json.dumps(report, separators=(",", ":")))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    arguments = parser.parse_args()
    inventory(arguments.data_dir, arguments.out_dir)
