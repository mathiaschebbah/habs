# Validation evidence

## Before reserved scoring

- Annotation audit: [run 42522381425](https://github.com/mathiaschebbah/habs/actions/runs/42522381425), source `1b53f01`.
- Signal preparation and real-data structural smoke: [run 42526673694](https://github.com/mathiaschebbah/habs/actions/runs/42526673694), source `0f4c410`. The smoke predicts a separate real J2 grid without current response columns or an in-grid prehistory buffer, with J1 supplied as the declared upstream calibration source.
- Python 3.12 CI validated the dependency pins, editable package, lint and unit tests. Local Python 3.13 tests additionally check future-response exclusion, invalid calibration rejection, grouped skore routing, both nonlinear/profile paths, participant weighting and portable cached reports.
- An end-to-end unit integration, explicitly not an empirical experiment, exercises development selection, the settings freeze, reserved-report creation, profile stability, the extreme-score diagnostic, and all seven figure exports under pytest's temporary directory.
- Structured review command: `/Users/mathias/.codex/skills/autoreview/scripts/autoreview --mode local --prompt-file scratch/review_study_context.md --output /tmp/habs-kickoff/study-review.md --json-output /tmp/habs-kickoff/study-review.json --stream-engine-output`.
- Review outcome: exit 0, no accepted/actionable findings. The reviewer independently inspected source routing, J1 calibration, five-second eligibility, identical comparison grids, participant-weighted errors and paired bootstrap intervals. No review-triggered source fixes were required.

All these checks precede the real reserved-outcome evaluation. The experiment
must additionally pass its real-data smoke on the current source and produce
persisted reports, a read-only audit and verified figures before being marked done.

## Reproduction checks

```sh
.venv/bin/ruff check process.py habs experiments tests
.venv/bin/python -m pytest tests -q
```

The real-data smoke is intentionally skipped on a machine lacking the restricted
labelled feature table. It is required and executed inside HABS, where
`HABS_REAL_FEATURES` is bound to the prepared table. Unit fixtures and historical
ignored simulation files are never used as evidence of predictive performance.
