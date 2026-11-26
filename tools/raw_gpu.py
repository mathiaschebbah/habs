"""HABS GPU entry point: raw CURATED prefixes, frozen ECG regression and frozen REVE.

The workflow stages pinned Study-09 helpers alongside this file. All subject-level
outputs stay in the team's private storage. Only counts are written to public/.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from decoding.extract import EEG_UNITS, _read_body
from decoding.ssl_audit import CANONICAL_62
from decoding.study09_extract import (
    INTERIOR_46_NAMES,
    _causal_r_peaks_in_window,
    _zscore_trial_causal,
)

from habs_stress.raw_eeg import (
    FS,
    LAGS,
    band_features,
    ecg_design,
    prefix,
    reference_eeg,
    regress_ecg,
)


def raw_blocks(path, onsets):
    if "CURATED" not in path.parts or "PREPROCESSED" in path.parts:
        raise ValueError("Raw experiment accepts only the CURATED data source")
    early = np.full((len(onsets), 62, 1640), np.nan, dtype=np.float32)
    before = np.full((len(onsets), 62, 600), np.nan, dtype=np.float32)
    with pyedflib.EdfReader(str(path)) as reader:
        lookup = {str(label).upper(): k for k, label in enumerate(reader.getSignalLabels())}
        for i, name in enumerate(CANONICAL_62):
            if name.upper() not in lookup:
                continue
            k = lookup[name.upper()]
            unit = reader.getPhysicalDimension(k).strip().replace("µ", "u").replace("μ", "u")
            if unit not in EEG_UNITS:
                continue
            fs = reader.getSampleFrequency(k)
            channel = reader.readSignal(k) * EEG_UNITS[unit]
            early[:, i] = prefix(channel, fs, onsets)[:, 360:]
            before[:, i] = prefix(channel, fs, onsets, stop=0.0)[:, -600:]
    early, valid = reference_eeg(early)
    before, pre_valid = reference_eeg(before)
    ecg, fs, resp, resp_fs, eda, eda_fs, _, _ = _read_body(path)
    design = ecg_design(prefix(ecg, fs, onsets))
    return (
        early,
        valid,
        before,
        pre_valid,
        design,
        (ecg, fs, resp, resp_fs, eda, eda_fs),
    )


def calibration_ids(frame):
    """Reserve ~8% of IDs per cohort exclusively for learning the ECG transfer."""
    reserved = set()
    for _, group in frame.groupby("cohort"):
        ids = sorted(
            group.participant_id.unique(),
            key=lambda i: hashlib.sha256(f"raw-ecg-cal-v1|{i}".encode()).hexdigest(),
        )
        reserved.update(ids[: min(max(2, round(len(ids) / 12)), len(ids) - 1)])
    return reserved


def learn_transfer(data_dir, cal):
    # Channel-specific normal equations exclude missing/flat calibration electrodes.
    grams = np.zeros((62, len(LAGS), len(LAGS)))
    cross = np.zeros((62, len(LAGS)))
    for recording, rows in cal.groupby("recording", sort=True):
        # No labels used. Bound the contribution of each calibration session.
        rows = rows.sort_values("onset_s").iloc[:16]
        eeg, good, _, _, design, _ = raw_blocks(data_dir / recording, rows.onset_s.to_numpy())
        for y, mask, x in zip(eeg, good, design, strict=True):
            if mask.sum() < 50 or not np.isfinite(x).all():
                continue
            x = x[::4].astype(float)
            grams[mask] += x.T @ x
            cross[mask] += y[mask, ::4] @ x
    if not np.all(np.trace(grams, axis1=1, axis2=2) > 0):
        raise ValueError("Insufficient independent calibration data")
    coefs = []
    for gram, cov in zip(grams, cross, strict=True):
        penalty = 1e-3 * np.trace(gram) / len(LAGS)
        coefs.append(np.linalg.solve(gram + penalty * np.eye(len(LAGS)), cov))
    return np.stack(coefs, axis=1).astype("float32")


def body_features(channels, onsets):
    ecg, ecg_fs, resp, resp_fs, eda, eda_fs = channels
    history = prefix(ecg, ecg_fs, onsets, start=-15, band=None)
    rr_feats = np.full((len(onsets), 17), np.nan, dtype=np.float32)
    for i, segment in enumerate(history):
        times, _ = _causal_r_peaks_in_window(segment, FS, -15.0)
        rr = np.diff(times)
        valid = rr[(rr >= 0.35) & (rr <= 1.5)][-16:]
        if len(valid):
            rr_feats[i, 16 - len(valid) : 16] = valid
            rr_feats[i, 16] = 5.0 - times[-1]
    extra = [rr_feats]
    pre = []
    for signal, fs, band in [
        (ecg, ecg_fs, (0.5, 45)),
        (resp, resp_fs, (0.08, 2)),
        (eda, eda_fs, None),
    ]:
        now = prefix(signal, fs, onsets, band=band)[:, 400:].reshape(len(onsets), 16, 100)
        extra.extend([now.mean(axis=2), now.std(axis=2)])
        # Independent prefix, including its QC and scaling, stops at stimulus onset.
        ref = prefix(signal, fs, onsets, start=-8, stop=0, band=band)[:, -600:]
        halves = ref.reshape(len(onsets), 2, 300)
        pre.extend([halves.mean(axis=2), halves.std(axis=2)])
    return np.column_stack(extra), np.column_stack(pre)


def coupling(y, good, x):
    z = x[:, :, 0]
    z = z - z.mean(axis=1, keepdims=True)
    e = y - y.mean(axis=-1, keepdims=True)
    den = np.linalg.norm(e, axis=-1) * np.linalg.norm(z, axis=-1)[:, None]
    corr = np.divide(
        np.abs(np.einsum("nct,nt->nc", e, z)),
        den,
        out=np.full(den.shape, np.nan),
        where=good & (den > 0),
    )
    return np.nanmedian(corr, axis=1)


def process(data_dir: Path, out_dir: Path):
    """Read only raw signals from the mounted corpus; write private features and public QC."""
    import torch
    from transformers import AutoModel

    started = time.monotonic()
    device = torch.device("cuda")
    torch.manual_seed(541)
    pos_model = (
        AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            revision="befa5b57a455b77cf302daf610c2e9ed8140bace",
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    model = (
        AutoModel.from_pretrained(
            "brain-bzh/reve-base",
            revision="33841986f6cc1438d9dcdbca5a2137d60f6007d9",
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    pos = pos_model(list(CANONICAL_62)).to(device)
    interior = [i for i, name in enumerate(CANONICAL_62) if name in INTERIOR_46_NAMES]

    @torch.inference_mode()
    def encode(y, positions, *, pre=False):
        encoded = []
        for start in range(0, len(y), 32):
            x = torch.from_numpy(np.ascontiguousarray(y[start : start + 32])).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model(
                    _zscore_trial_causal(x),
                    positions.unsqueeze(0).expand(len(x), -1, -1),
                )
                if pre:
                    vector = tokens.float().mean(dim=(1, 2))
                else:
                    cm = tokens.float().mean(dim=1)
                    pool = model.attention_pooling(tokens[:, :, 4:9, :])
                    vector = torch.cat(
                        [
                            cm[:, 4:9].mean(1),
                            pool.float(),
                            cm[:, 6:9].mean(1) - cm[:, 4:6].mean(1),
                        ],
                        dim=1,
                    )
            encoded.append(vector.cpu().numpy())
        return np.concatenate(encoded)

    grid = pd.read_parquet(Path(__file__).parent / "grid.parquet")
    # Historical alignment allowed a PREPROCESSED fallback. The raw experiment
    # uses only trials whose actual alignment and peripheral targets used CURATED.
    raw_source = grid["recording"].map(
        lambda rec: "CURATED" in Path(rec).parts and "PREPROCESSED" not in Path(rec).parts
    )
    aligned_raw = grid.loc[raw_source]
    ids = calibration_ids(aligned_raw)
    cal = aligned_raw.loc[aligned_raw.participant_id.isin(ids)]
    evaluation = aligned_raw.loc[~aligned_raw.participant_id.isin(ids)]
    coefs = learn_transfer(data_dir, cal)
    import pyarrow as pa
    from pyarrow import parquet as pq

    private = out_dir / "features"
    private.mkdir(parents=True, exist_ok=True)
    writers = {}

    def write_rows(name, rows):
        batch = pa.Table.from_pandas(rows, preserve_index=False)
        if name not in writers:
            writers[name] = pq.ParquetWriter(
                private / (name + ".parquet"), batch.schema, compression="zstd"
            )
        writers[name].write_table(batch)

    seen_trials = set()
    seen_ids = set()
    print(
        f"Starting CURATED-only grid: {evaluation.recording.nunique()} evaluation recordings",
        flush=True,
    )
    # Exercise the complete feature path, including the encoder and fixed regression.
    rng = np.random.default_rng(2187)
    signal = rng.normal(size=13000)
    walls = {}
    for stop in (0.0, 5.0):
        onset = 6.0
        edited = signal.copy()
        k = int(1000 * (onset + stop))
        edited[k::2] = np.nan
        edited[k + 1 :: 2] = 1e7
        features = []
        for trace in (signal, edited):
            raw = prefix(trace, 1000, [onset], stop=stop)
            bank = np.stack([raw * np.cos(j * 0.11) for j in range(62)], axis=1)
            if stop == 5:
                y, ok = reference_eeg(bank[:, :, 360:])
                d = ecg_design(raw)
                y = regress_ecg(y, ok, d, coefs)
                features.append((y, band_features(y), encode(y, pos)))
            else:
                y, _ = reference_eeg(bank[:, :, -600:])
                features.append((y, encode(y, pos, pre=True)))
        walls[str(stop)] = all(np.array_equal(a, b) for a, b in zip(*features, strict=True))
        if not walls[str(stop)]:
            raise AssertionError("A future perturbation changed raw-prefix features")
    for recno, (recording, rows) in enumerate(evaluation.groupby("recording", sort=True)):
        onsets = rows.onset_s.to_numpy()
        eeg, good, before, pregood, design, channels = raw_blocks(data_dir / recording, onsets)
        body, prebody = body_features(channels, onsets)
        prevec = encode(before, pos, pre=True)
        prevec[pregood.sum(axis=1) < 50] = np.nan
        prepart = rows[
            [
                "trial_id",
                "participant_id",
                "rating_valence",
                "rating_arousal",
                "cohort",
                "theme",
                "person_fold",
                "theme_fold",
            ]
        ].copy()
        prepart["net_pre"] = list(prevec)
        prepart["prebody"] = list(prebody.astype("float32"))
        write_rows("study11_pre0", prepart)
        keep = (good.sum(axis=1) >= 50) & np.isfinite(design).all(axis=(1, 2))
        if not keep.any():
            continue
        eeg, good, design = eeg[keep], good[keep], design[keep]
        clean = regress_ecg(eeg, good, design, coefs)
        features = {
            "extra": body[keep],
            "prebody": prebody[keep],
            "hc_before": band_features(eeg),
            "hc_after": band_features(clean),
            "net_before": encode(eeg, pos),
            "net_after": encode(clean, pos),
            "net_interior": encode(clean[:, interior], pos[interior]),
            "net_pre": prevec[keep],
        }
        part = rows.loc[keep].drop(columns=["recording"]).reset_index(drop=True)
        part["median_abs_ecg_r_before"] = coupling(eeg, good, design)
        part["median_abs_ecg_r_after"] = coupling(clean, good, design)
        for name, array in features.items():
            part[name] = list(array.astype("float32"))
        if seen_trials.intersection(part.trial_id):
            raise ValueError("Duplicate trial in raw extraction")
        seen_trials.update(part.trial_id)
        seen_ids.update(part.participant_id)
        write_rows("study11_raw", part)
        if recno % 25 == 0:
            print(
                f"Raw: {recno + 1} recordings processed, {time.monotonic() - started:.0f} s",
                flush=True,
            )

    for writer in writers.values():
        writer.close()
    if len(writers) != 2:
        raise ValueError("Empty extraction")
    summary = {
        "source": "CURATED",
        "dev_trials": len(grid),
        "calibration_ids": len(ids),
        "evaluation_trials_before_qc": len(evaluation),
        "evaluation_trials_after_qc": len(seen_trials),
        "non_curated_alignment_trials_excluded": int((~raw_source).sum()),
        "non_curated_alignment_recordings_excluded": int(
            grid.loc[~raw_source, "recording"].nunique()
        ),
        "curated_grid_ids": int(aligned_raw.participant_id.nunique()),
        "curated_grid_trials": len(aligned_raw),
        "calibration_trials": len(cal),
        "evaluation_ids_after_qc": len(seen_ids),
        "calibration_overlap_ids": len(seen_ids & ids),
        "bit_invariance_at_0_and_5": walls,
        "seconds": round(time.monotonic() - started, 2),
    }
    public = out_dir / "public"
    public.mkdir(exist_ok=True)
    (public / "raw_coverage.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
