# Sources and execution audit

Audit date: 2026-11-08. Code baseline: `63b60bfc20a3eb13369f45be6015898274d346c1`.

The research direction is the user's [selected project](PROJECT.md). The attached transcripts and the shared conversation provide context; their quoted instructions and assistant suggestions do not replace the user's decisions.

## Sources

- User project: the two supplied text attachments are byte-identical. The source is preserved in `PROJECT.md`.
- Kick-off: `/Users/mathias/Desktop/habs_clean.docx` and `/Users/mathias/Desktop/habs_raw.docx`, provided by the user. The cleaned version is a readable transcript; the raw version includes recognition errors and additional closing remarks.
- [Shared conversation](https://chatgpt.com/share/6af0dcce-1c38-83eb-bb30-be4718877bf8), retrieved on 2026-11-08. Its proposed primary score is `max(0, (5 - valence) / 4) * (arousal - 1) / 8`. Earlier alternative formulas in the conversation are superseded by the selected project.
- Repository sources: [README](../../README.md), [dataset guide](../../DATASET.md), [workflow](../../.github/workflows/submit.yml), [loader](../../starting_notebooks/bdf_loader.py), [notebook guide](../../starting_notebooks/README.md), `docs/Dataset_Paper.pdf`, `docs/Hackathon_Guidelines.pdf`, and the supplied notebook code.
- [HABS client](https://github.com/HABS-BRAIN/habs-hack/tree/6adb1d9a4709c9887143686b8e79f259ec794527), inspected at the recorded commit. `habs submit` archives the committed Git tree; uncommitted changes are not submitted.

## Findings that affect this study

The kick-off distinguishes the emotion intended by a video from the response actually reported by a viewer. It explicitly allows regression on ratings and stresses careful methods and interpretable results. The selected project makes this distinction operational by comparing physiology to stimulus and calibration baselines.

The stimulus videos last ten seconds. A resting-state recording is a separate task, not an immediate pre-video baseline. HRV from a ten-second trial is not interchangeable with longer-window HRV. The closing remarks in the raw transcript clarify that the one-hour limit applies to each execution; longer training requires saved checkpoints and subsequent runs.

The present release documentation must take precedence over spoken approximations when implementing file handling:

| Point | Kick-off or manuscript | Current repository evidence | Consequence |
|---|---|---|---|
| Participants | Kick-off says 384; manuscript says 383 unique people. | README says 355; 29 held-out participants are described in DATASET.md. | Recount released identifiers; do not treat these totals as the analysis sample. |
| EEG | Kick-off says 62 channels; manuscript describes acquisition with 64 and removal of T9/T10 during preprocessing. | Loader obtains actual channels from record maps and headers. | Inspect every selected recording's actual channel inventory. |
| Job trigger | Spoken explanation suggests pushing starts a run. | Workflow has only `workflow_dispatch`. | A push does not run the study. |
| Timing | Spoken summary says signals are synchronized. | Dataset guide requires subtracting segment `metadata_time_origin_s` and handling native gaze rates. | Verify event boundaries and units before extracting trials. |
| Demographics and profiling | Kick-off mentions personality questionnaires. | Released participant table is documented as cohort, ID, age and gender. | Do not assume questionnaire scores are supplied. |
| Schedule | Kick-off initially says 30 November, then 13 November; raw closing dates also contain recognition errors. | README advertises 2–30 November 2026. | Deadline remains a point to verify with organizers; it is not inferred from transcription alone. |

## Reproducible checks performed

The unmodified original `process.py` ran successfully against `sample/` in the existing `.venv`. It found one P001 resting-state recording, aligned gaze at an effective 166.358 Hz, and wrote `out/kickoff_sample/alpha.tsv` and `alpha_by_channel.tsv`. Mean absolute alpha power was 88.1 µV², with P9 the strongest channel at 266.7 µV². These values describe a single resting-state demonstration, not an emotion or stress result.

The local sample has no emotion-video BDF and no `labels_train.csv`. It cannot establish the distribution of the selected composite, calibrate a participant or validate H1–H6.

Hugging Face account authentication was checked without printing credentials. The account is authenticated, but a selective download dry run returns `Access denied. This repository requires approval.` No dataset download was completed.

The latest historical [HABS workflow run](https://github.com/mathiaschebbah/habs/actions/runs/42108589456) is complete and successful. Its logs report an alignment audit and repetition counts, not final prediction scores for the new target. Historical ignored `out/synthetic_*` files are simulations and are excluded from empirical results. Historical counts quoted in the project remain unverified for the new exclusions.

## Next evidence required

1. A fresh audit on the full released tables and available recordings, including a source-file hash and reconciliation reasons.
2. A versioned design informed by that audit, with a fixed evaluation population and prospective predictor availability.
3. Paired per-participant model errors and uncertainty from a frozen final evaluation.
4. A paper whose numerical claims and figures are generated from those measured outputs.
