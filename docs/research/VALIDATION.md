# Validation evidence

## Before reserved scoring

- Annotation audit: [run 42522381425](https://github.com/mathiaschebbah/habs/actions/runs/42522381425), source `1b53f01`.
- Signal preparation and real-data structural smoke: [run 42526673694](https://github.com/mathiaschebbah/habs/actions/runs/42526673694), source `0f4c410`. The smoke predicts a separate real J2 grid without current response columns or an in-grid prehistory buffer, with J1 supplied as the declared upstream calibration source.
- Python 3.12 CI validated the dependency pins, editable package, lint and unit tests. Local Python 3.13 tests additionally check future-response exclusion, invalid calibration rejection, grouped skore routing, both nonlinear/profile paths, participant weighting and saved reports whose original raw-input files are unavailable.
- An end-to-end unit integration, explicitly not an empirical experiment, exercises development selection, the settings freeze, reserved-report creation, profile stability, the extreme-score diagnostic, and all seven figure exports under pytest's temporary directory.
- Structured review command: `/Users/mathias/.codex/skills/autoreview/scripts/autoreview --mode local --prompt-file scratch/review_study_context.md --output /tmp/habs-kickoff/study-review.md --json-output /tmp/habs-kickoff/study-review.json --stream-engine-output`.
- Review outcome: exit 0, no accepted/actionable findings. The reviewer independently inspected source routing, J1 calibration, five-second eligibility, identical comparison grids, participant-weighted errors and paired bootstrap intervals. No review-triggered source fixes were required.

These checks preceded the real reserved-outcome evaluation. The scientific source
`29379d8` subsequently passed its real-data structural smoke (1 passed in 2.26s),
froze settings at 23:15:36 UTC on 8 November, exported seven figures and finished
successfully. The backend job `20261108T230726-28480a` succeeded at 23:19:24 UTC.

## Completed report audit and artifact verification

[Audit recovery run 42535323126](https://github.com/mathiaschebbah/habs/actions/runs/42535323126)
successfully retrieved 11 primary reports and their exact dataset dependencies:
160 files, 3.22 MB, within the published 25 MB transfer limit. The full job output
is 71 MB and was not downloaded as an unrestricted bundle. HNS directory metadata
is filtered; the original AzureCLI folder-as-file failure did not rerun models.

The [executed audit](results/audit.md) confirms source and frozen/data hashes,
997 finite predictions for every primary model, agreement of registered
participant-macro MAE with the exported evidence, and independent reconstruction
of the paired 10,000-resample interval. No `evaluate`, `fit` or `put` is called
by the audit. Generic skore checks are mostly not applicable to the prefit custom
wrapper; `default_score` lacks calibration routing, while registered macro-MAE,
cached predictions and built-in R²/RMSE are readable. Those limitations are not
presented as successful automated leakage or overfitting tests.

The manuscript is four A4 pages and the supplement two. Both were rendered with
Poppler and every final page inspected. All seven empirical figures are included
across the two PDFs, with no clipped text, overlapping elements or missing glyphs.

## Reproduction checks

Skore 0.29.1 report workspaces contain absolute references to saved dataset files.
Cached prediction evidence survives unavailable original input CSVs, but moving
the whole workspace requires restoring its original absolute location. The
post-run CPU audit therefore restores the team's private output tree at `/out`
before opening `/out/reports`. It never refits a missing report or changes mode.

```sh
.venv/bin/ruff check process.py habs experiments tests
.venv/bin/python -m pytest tests -q
```

The real-data smoke is intentionally skipped on a machine lacking the restricted
labelled feature table. It is required and executed inside HABS, where
`HABS_REAL_FEATURES` is bound to the prepared table. Unit fixtures and historical
ignored simulation files are never used as evidence of predictive performance.
