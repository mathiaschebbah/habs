# Carryover study validation

## Before real scientific effects

- User approved autonomous feasibility, execution, audit and paper on 2026-11-09;
  [design note](../../../journal/02_emotional_carryover.md).
- Immutable native feature inputs are verified against their existing manifest
  and the read-only feasibility fingerprints. Original study-01 native extraction,
  models and frozen settings are unchanged; study-02 code is under `habs/temporal/`.
- [Real feasibility](https://github.com/mathiaschebbah/habs/actions/runs/42594747849)
  and [pre-effect simulated precision](https://github.com/mathiaschebbah/habs/actions/runs/42598671314)
  establish 53 paired IDs / 4,554 novel-current-video pairs and full conditional
  personal-history rank. Precision is explicitly synthetic, not a result of the
  scientific hypothesis. A final exact-source refresh is required by the run guard.
- Existing environment choices remain pandas, pip/venv, local skore and matplotlib.
  Installed versions were inspected: NumPy 2.4.6, pandas 3.0.1, SciPy 1.17.1,
  sklearn 1.9.1, skrub 0.11.0 and skore 0.29.1.
- Source-bound prediction marks the query grid before any predecessor join.
  Population transforms fit only training IDs. Held-person J1 calibration is
  allowed; current responses are excluded from predictors. Tests mutate current
  responses and held-person calibration separately to check those boundaries.
- A real-data cold-grid test is present for each study. Both must execute inside
  HABS before the scientific study continues; local restricted-data skips are
  not presented as passing real-data tests.
- Skore 0.29.1 does not forward groups with an explicit CV object. The producer
  computes genuine GroupKFold indices with groups and passes those to evaluate.
  The same five grouped J2 folds are used for all three forecast comparisons.
- Skore CV restoration needs original bindings. A unit reproduces the failure
  without them, restores exact files, forbids estimator fitting and verifies
  cached forecasts and metrics. Recovery restores those private files at `/out`;
  only public aggregates, figures and audit text leave the private tree.
- Covariance inference was checked on simulated effects with known covariance,
  scale confounding and zero-variance boundaries; projection is invariant to fixed
  level shifts. An end-to-end synthetic test exercises statistics, all forecasts,
  saved reports and five figures. Synthetic tests are never empirical evidence.

## Code review and local checks

```sh
/Users/mathias/.codex/skills/autoreview/scripts/autoreview --mode local --engine codex --prompt-file scratch/review_carryover_context.md --parallel-tests 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q' --output /tmp/habs-kickoff/carryover-review.md --json-output /tmp/habs-kickoff/carryover-review.json --stream-engine-output
```

The helper exited 0 with no actionable findings, confidence 0.84. Static review
covered numerical inference, forecast routing, restoration, dispatch and bounded
retrieval. No review-triggered fixes were needed. Its independent parallel test
process passed 37 tests, with the two restricted-data smoke tests skipped locally.
Lint passes on `process.py habs experiments audit tests tools`.

## Remaining execution evidence

Final exact-source precision, frozen real-data fits, cached-report audit and PDF
visual verification will be recorded here after their actual completion.
