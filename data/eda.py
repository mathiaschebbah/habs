# %% [markdown]
# # Released HABS trials and recording coverage
#
# Audit annotations, event clocks and native sensor units before selecting models.
# Response distributions exclude participants reserved for final evaluation.

# %%
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyedflib
import skrub

from habs import PROJECT_ROOT

RAW_DIR = Path(os.environ.get("HABS_DATA_ROOT", PROJECT_ROOT / "sample"))
EDA_DIR = Path(os.environ.get("HABS_ANALYSIS_OUT", PROJECT_ROOT)) / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20261108


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _video(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


def _day(value):
    match = re.search(r"\d+", str(value))
    return int(match[0]) if match else None


def _target(row):
    valence = _number(row.get("rating_valence"))
    arousal = _number(row.get("rating_arousal"))
    if (
        valence is None
        or arousal is None
        or not (1 <= valence <= 9 and 1 <= arousal <= 9)
    ):
        return None
    return max(0.0, (5 - valence) / 4) * (arousal - 1) / 8


def _overview(frame, name):
    report = skrub.TableReport(frame, title=name, verbose=0)
    report.write_html(EDA_DIR / f"eda_{name}.html")
    facts = json.loads(report.json())
    return {
        "n_rows": facts.get("n_rows"),
        "n_columns": facts.get("n_columns"),
        "columns": [
            {
                k: c.get(k)
                for k in (
                    "name",
                    "dtype",
                    "null_proportion",
                    "n_unique",
                    "value_is_constant",
                )
            }
            for c in facts.get("columns", [])
        ],
    }


# %% [markdown]
# ## Recordings and full-session event clocks
#
# CURATED is preferred to PREPROCESSED. A video must fit the exported segment
# and its onset must agree with a BDF annotation bearing the same marker.

# %%
recordings, events, header_errors = [], [], []
seen_sessions = set()
for source in ("CURATED", "PREPROCESSED"):
    for metadata_path in sorted((RAW_DIR / source).glob("*/*metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        export = metadata.get("dataset_export", {})
        participant = export.get("participant_id", metadata_path.parent.name)
        day = _day(export.get("day", metadata_path.name.split("_day-")[-1]))
        session = (participant, day)
        segment = export.get("segments", {}).get("emotion-video", {})
        bdf = metadata_path.parent / str(segment.get("file", "not-present"))
        if session in seen_sessions or not bdf.is_file():
            continue
        try:
            with pyedflib.EdfReader(str(bdf)) as reader:
                names = reader.getSignalLabels()
                duration = min(
                    float(reader.file_duration), float(segment["duration_s"])
                )
                annotation_onsets, _, annotation_codes = reader.readAnnotations()
                annotation_times = defaultdict(list)
                for time, code in zip(annotation_onsets, annotation_codes, strict=True):
                    number = _number(code)
                    if number is not None:
                        annotation_times[int(number)].append(float(time))
                header = {
                    "participant_id": participant,
                    "recorded_day": day,
                    "source": source,
                    "recording": str(bdf.relative_to(RAW_DIR)),
                    "duration_s": duration,
                    "n_channels": len(names),
                    "eeg_selected_count": len(
                        set(names) & {"F3", "F4", "Fz", "P3", "P4", "O1", "O2"}
                    ),
                }
                for modality, name in (
                    ("ecg", "ECG"),
                    ("eda", "GSR"),
                    ("resp", "Resp"),
                ):
                    if name in names:
                        index = names.index(name)
                        header[f"{modality}_unit"] = reader.getPhysicalDimension(index)
                        header[f"{modality}_rate_hz"] = float(
                            reader.getSampleFrequency(index)
                        )
                        header[f"{modality}_physical_min"] = reader.getPhysicalMinimum(
                            index
                        )
                        header[f"{modality}_physical_max"] = reader.getPhysicalMaximum(
                            index
                        )
                recordings.append(header)
                seen_sessions.add(session)
        except (OSError, ValueError, KeyError) as error:
            header_errors.append(
                {"recording": str(bdf.relative_to(RAW_DIR)), "error": str(error)}
            )
            continue
        origin = float(segment["metadata_time_origin_s"])
        selected = [
            e
            for e in metadata.get("events", [])
            if e.get("trial_type") == "emotion_video"
        ]
        selected.sort(key=lambda e: float(e["onset_s"]))
        for order, event in enumerate(selected):
            marker = _number(event.get("marker_id"))
            if marker is None:
                continue
            onset = float(event["onset_s"]) - origin
            event_duration = _number(event.get("duration_s"))
            inside = onset >= -0.002 and onset + 10 <= duration + 0.002
            annotation_error = min(
                (abs(t - onset) for t in annotation_times.get(int(marker), [])),
                default=None,
            )
            events.append(
                {
                    **header,
                    "video": _video(event.get("media_file")),
                    "start_marker": int(marker),
                    "event_order": order,
                    "onset_s": onset,
                    "event_duration_s": event_duration,
                    "inside_segment": inside
                    and event_duration is not None
                    and event_duration >= 9.95,
                    "annotation_error_s": annotation_error,
                }
            )
{
    "recordings": len(recordings),
    "video_events": len(events),
    "header_errors": len(header_errors),
}

# %% [markdown]
# ## Label reconciliation
#
# Require participant, normalized video filename and start marker. A unique match
# on another recorded day repairs a label-day error. Exclude ambiguous matches
# and duplicate labels mapping to one physical event.

# %%
label_path = RAW_DIR / "labels_train.csv"
RAW = (
    pd.read_csv(label_path, na_values=["TESTSUBJECT"])
    if label_path.is_file()
    else pd.DataFrame()
)
label_records = RAW.to_dict("records")
index = defaultdict(list)
for event in events:
    index[(event["participant_id"], event["video"], event["start_marker"])].append(
        event
    )
aligned, exclusions = [], Counter()
for label in label_records:
    score = _target(label)
    if score is None:
        exclusions["missing_or_invalid_valence_arousal"] += 1
        continue
    participant = str(label.get("participant_id", "")).strip()
    marker = _number(label.get("start_marker"))
    key = (
        (participant, _video(label.get("video")), int(marker))
        if marker is not None
        else None
    )
    candidates = index.get(key, [])
    label_day = _day(label.get("day"))
    same_day = [e for e in candidates if e["recorded_day"] == label_day]
    matches = same_day if len(same_day) == 1 else candidates
    if len(matches) != 1:
        exclusions["ambiguous_event" if matches else "no_matching_recorded_event"] += 1
        continue
    event = matches[0]
    if not event["inside_segment"]:
        exclusions["outside_or_short_segment"] += 1
        continue
    if event["annotation_error_s"] is None or event["annotation_error_s"] > 0.025:
        exclusions["bdf_marker_disagreement"] += 1
        continue
    aligned.append(
        {
            **label,
            **event,
            "label_day": label_day,
            "aversive_score": score,
            "day_corrected": label_day != event["recorded_day"],
        }
    )
event_keys = Counter(
    (r["participant_id"], r["recorded_day"], r["start_marker"], r["onset_s"])
    for r in aligned
)
unique = []
for row in aligned:
    key = (
        row["participant_id"],
        row["recorded_day"],
        row["start_marker"],
        row["onset_s"],
    )
    if event_keys[key] == 1:
        unique.append(row)
    else:
        exclusions["duplicate_label_for_physical_event"] += 1
aligned = unique
{"label_rows": len(RAW), "matched_trials": len(aligned), "exclusions": dict(exclusions)}

# %% [markdown]
# ## Structural allocation before response distributions
#
# Cohort-2 eligibility requires twenty rated trials in each recorded session.
# One third of eligible identifiers are reserved by deterministic hash. All
# response distributions for those people remain outside this exploration.

# %%
cohort_by_person = {
    str(r.get("participant_id")): _number(r.get("cohort")) for r in label_records
}
session_counts = Counter((r["participant_id"], r["recorded_day"]) for r in aligned)
paired = sorted(
    {
        pid
        for pid, day in session_counts
        if cohort_by_person.get(pid) == 2
        and session_counts[(pid, 1)] >= 20
        and session_counts[(pid, 2)] >= 20
    }
)
ranked = sorted(
    paired, key=lambda pid: hashlib.sha256(f"{SEED}:{pid}".encode()).hexdigest()
)
heldout = set(ranked[: math.ceil(len(ranked) / 3)])
allocation = [
    {
        "participant_id": pid,
        "role": "final_test" if pid in heldout else "development",
        "eligible_two_sessions": pid in paired,
    }
    for pid in sorted(cohort_by_person)
]
development = [r for r in aligned if r["participant_id"] not in heldout]
primary_development = [r for r in development if _number(r.get("cohort")) == 2]
allocation_hash = hashlib.sha256(
    json.dumps(allocation, sort_keys=True).encode()
).hexdigest()
{
    "cohort2_paired_people": len(paired),
    "heldout_people": len(heldout),
    "development_paired_people": len(paired) - len(heldout),
    "allocation_sha256": allocation_hash,
}

# %% [markdown]
# ## Coverage and composite support
#
# TableReport supplies per-column missingness and cardinality. Domain counts
# describe stimulus repetitions and responses at comparable arousal. The derived
# final emotion label is not an independent measure of the selected target.

# %%
DEV = pd.DataFrame(development)
overview = (
    _overview(DEV, "development_trials") if len(DEV) else {"n_rows": 0, "columns": []}
)
participant_path = RAW_DIR / "participants_train.csv"
participant_overview = None
if participant_path.is_file():
    PEOPLE = pd.read_csv(participant_path, na_values=["TESTSUBJECT"])
    PEOPLE = PEOPLE.loc[~PEOPLE["participant_id"].isin(heldout)]
    participant_overview = _overview(PEOPLE, "development_participants")
overview

# %%
distributions, emotion_counts, arousal_valence = (
    defaultdict(Counter),
    defaultdict(Counter),
    Counter(),
)
for row in primary_development:
    day = str(row["recorded_day"])
    score, valence, arousal = (
        row["aversive_score"],
        float(row["rating_valence"]),
        float(row["rating_arousal"]),
    )
    distributions[day][min(19, int(score * 20))] += 1
    emotion = str(row.get("self_reported_emotion_1", "missing"))
    emotion_counts[emotion]["trials"] += 1
    emotion_counts[emotion]["positive_proxy"] += int(score > 0)
    if arousal >= 6:
        arousal_valence[
            "negative" if valence <= 4 else "positive" if valence >= 6 else "middle"
        ] += 1
cohort_counts = Counter(
    str(int(cohort_by_person.get(r["participant_id"]) or 0)) for r in aligned
)
cohort_people = {
    c: len(
        {
            r["participant_id"]
            for r in aligned
            if str(int(cohort_by_person.get(r["participant_id"]) or 0)) == c
        }
    )
    for c in sorted(cohort_counts)
}
repeated = defaultdict(set)
for row in aligned:
    repeated[(row["participant_id"], row["video"])].add(row["recorded_day"])
unit_counts = {
    m: dict(Counter(str(r.get(f"{m}_unit", "missing")) for r in recordings))
    for m in ("ecg", "eda", "resp")
}
SUMMARY = {
    "status": "full_annotation_audit"
    if label_path.is_file()
    else "sample_without_ratings",
    "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest()
    if label_path.is_file()
    else None,
    "seed": SEED,
    "label_rows": len(RAW),
    "matched_trials": len(aligned),
    "matched_people": len({r["participant_id"] for r in aligned}),
    "recordings": len(recordings),
    "header_errors": header_errors,
    "exclusions": dict(exclusions),
    "cohort_trials": dict(cohort_counts),
    "cohort_people": cohort_people,
    "day_corrections": sum(r["day_corrected"] for r in aligned),
    "cross_session_person_video_repetitions": sum(
        len(d) > 1 for d in repeated.values()
    ),
    "cohort2_paired_people": len(paired),
    "heldout_people": len(heldout),
    "allocation_sha256": allocation_hash,
    "native_unit_counts": unit_counts,
    "development_overview": overview,
    "participant_overview": participant_overview,
    "cohort2_development_proxy_histogram": {
        d: dict(c) for d, c in distributions.items()
    },
    "cohort2_development_emotions": {n: dict(c) for n, c in emotion_counts.items()},
    "cohort2_development_high_arousal_valence": dict(arousal_valence),
    "warning": (
        "Held-out response distributions and final model scores "
        "have not been inspected."
    ),
}
pd.DataFrame(aligned).to_csv(EDA_DIR / "aligned_trials.csv.gz", index=False)
pd.DataFrame(allocation).to_csv(EDA_DIR / "allocation.csv", index=False)
pd.DataFrame(recordings).to_csv(EDA_DIR / "recordings.csv", index=False)
pd.DataFrame(events).to_csv(EDA_DIR / "recorded_events.csv.gz", index=False)
(EDA_DIR / "eda_summary.json").write_text(
    json.dumps(SUMMARY, indent=2, allow_nan=False)
)
SUMMARY
