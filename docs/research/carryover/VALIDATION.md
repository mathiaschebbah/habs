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

## PDF authoring environment

The PDF builder uses a separate publication environment, not the HABS numerical
virtual environment. In this desktop session it runs with the bundled Python
runtime, where ReportLab 4.4.9 was inspected and the audit-binding rejection
checks passed. A portable setup from the repository root is:

```sh
python3 -m venv .venv-publication
.venv-publication/bin/pip install -r requirements-publication.txt
.venv-publication/bin/python docs/research/carryover/build_paper.py --font-dir /path/to/liberation-fonts
```

The font directory must contain LiberationSans-Regular.ttf, LiberationSans-Bold.ttf
and LiberationSans-Italic.ttf. The builder also detects the bundled desktop fonts
and common Linux Liberation font locations. It refuses to publish result/freeze
files whose exact hashes are not covered by the successful executed audit.

## Execution closeout

The [exact-source precision refresh](https://github.com/mathiaschebbah/habs/actions/runs/42617738546)
has completed on source `0e9bd01`. The downloaded geometry fingerprint matches
the reviewed implementation, with 53 people / 4,554 transitions and full rank.
The [Code checks for that source](https://github.com/mathiaschebbah/habs/actions/runs/42617710024)
also pass on Python 3.12.

[Scientific run 42618363214](https://github.com/mathiaschebbah/habs/actions/runs/42618363214)
succeeded on source `5b079a836da54652240c00c43555a1624400ae0e`, backend
`20261109T123726-87e44e`. The freeze was written at 12:37:36 UTC before fitting
the scientific effects. Both real-data smoke tests passed in 15.59 s. All
prespecified statistical and forecast comparisons were completed without tuning
their settings on the new outcomes.

[Read-only audit 42657666606](https://github.com/mathiaschebbah/habs/actions/runs/42657666606)
passed source/input/freeze hashes, saved-likelihood reconstruction, three cached
cross-validation reports and fifteen folds, finite predictions, registered
participant errors and both paired intervals. Its structured completion record
binds the exact result and freeze hashes to four completed verification stages.

The first recovery selection exceeded 25 MB before downloading. Metadata-only
inspection found `latest__` directory mirrors and duplicate `estimator_.pickle`
files unused by the installed reader. Their omission recovers 309 necessary
files, 21.62 MB, within the published limit. Canonical learners, original private
bindings and dataset dependencies remain intact. The focused restore-without-fit
test passes after removing the same duplicates.

Generic skore checks are unavailable here: constant-input warnings and an
array-dimensionality error arise on the query metadata. They are not described
as passed leakage/overfitting tests. Cached metrics and targeted source/grid
invariants are verified separately. Default score and R² are readable; MAPE is
not interpreted for the signed difference target.

The delivery review accepted fixes for exact audit/input binding, a separate
publication dependency declaration and structured executed completion rather
than an echoed-source marker. Matching completion is accepted; stale results,
stale freeze, echoed source and incomplete stages are rejected by focused checks.
Final helper run exited 0 with no actionable findings, confidence 0.88:

```sh
/Users/mathias/.codex/skills/autoreview/scripts/autoreview --mode local --engine codex --prompt-file scratch/review_carryover_delivery_context.md --output /tmp/habs-kickoff/carryover-delivery-review.md --json-output /tmp/habs-kickoff/carryover-delivery-review.json --stream-engine-output
```

## Final artifact inspection

The final manuscript has five A4 pages, the supplement three. Both PDFs were
rendered with Poppler at 125 dpi and all eight final pages inspected. A figure
annotation was moved clear of its reference line, and figures were regenerated
only from saved audited numbers. No model or statistical fit was rerun for layout.
Final pages have no clipped text, overlapping annotations, missing glyphs or
broken tables. The five figures distinguish actual observations from simulated
precision. [Results](RESULTS.md) give the interpretation and limitations in French.
