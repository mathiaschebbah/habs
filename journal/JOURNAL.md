# HABS research journal

## Status

- **Project / dataset:** HABS multimodal emotional video recordings.
- **Goal:** Study individual emotional responses and subjective–EEG temporal repeatability with stimulus-aware controls, then deliver audited results and research papers.
- **Research direction:** the user-selected [project](../docs/research/PROJECT.md), supplied and confirmed on 2026-11-08. The historical arousal-only study is not the experiment being evaluated.
- **Last experiment:** 02_emotional_carryover, done; measured feasibility, exact-source precision, frozen scientific fits, saved-report audit and paper completed.
- **Last result:** 53 paired IDs / 4,554 transitions; cross-day contrast covariance +0.003496, 95% model-based CI [−0.010277, +0.015055], p=0.253. Correlation weakly identified. Personal-slope J2 MAE gain −0.005539 [−0.012251, +0.000938]; no demonstrated increment. [Results](../docs/research/carryover/RESULTS.md), [paper](../docs/research/carryover/manuscript.pdf), [supplement](../docs/research/carryover/supplement.pdf).
- **Literature follow-up, 2026-11-09:** [systematized scoping review](../docs/research/literature/2026-11-09/REVIEW.md), 21 Exa searches and five complete Europe PMC queries; 28 core publications plus 29 contextual publications extracted. Its [question](../docs/research/literature/2026-11-09/QUESTION.md) was promoted and completed as study 02. Novelty remains conditional on the searched corpus.
- **Workspace decisions** (user answers, 2026-11-08):
  - tabular library: pandas
  - env manager: pip+venv
  - package name: habs; root package layout, importable from the HABS source archive
  - skore mode: local
  - skore hub workspace: n/a
  - skore mlflow tracking uri: n/a
  - plotting: matplotlib
  - EDA: run before model design
  - agent feature: installed in .venv-agent and .venv-lsp; IPython and pyright verified by the bundled installer, 2026-11-09
  - CV splitter family: five grouped participant folds with J2-only validation and J1-only personal calibration; recorded after EDA under autonomous authority
  - optional features: none selected
  - study-02 validation: five GroupKFold person folds on J2 queries, explicit grouped index tuples for skore 0.29.1, with J1-only personal calibration
  - publication environment: separate bundled ReportLab 4.4.9; portable pins in requirements-publication.txt
- **Execution authority:** on 2026-11-08 the user explicitly requested autonomous completion without further questions. Remaining implementation, dependency, design and run decisions will be made and recorded under that authorization.

## Data understanding (EDA)

- **Status:** annotation/coverage and native signal audit done, 2026-11-08; real-data structural smoke passed.
- **Observed access:** local sample has resting-state and artefact recordings only, without trial ratings. Hugging Face access is denied; the HABS/Azure route is verified by a successful full-data audit.
- **Source audit:** [kick-off and repository](../docs/research/KICKOFF.md).
- **Report:** [data/eda.md](../data/eda.md), [aggregate evidence](../data/eda_summary.json), [successful cloud run](https://github.com/mathiaschebbah/habs/actions/runs/42522381425). Response concentration near zero supports continuous analysis; both positive and negative responses at high arousal exist in development.
- **Signal evidence:** [feature summary](../data/feature_summary.json), [successful preparation run](https://github.com/mathiaschebbah/habs/actions/runs/42526673694). Cohort 2 has 7,795 EEG-usable trials and 3,672 complete multimodal observations. Out-of-calibration EDA is marked missing in 2,742 trials. Final eligibility retains 22 of the original 26 reserved identifiers without reassignment.

## History

| Stem | Intent | Status | Headline result | Design note |
|---|---|---|---|---|
| 01_aversive_reactivity | Test incremental physiology beyond stimulus and J1 calibration. | done | Gain −0.000383 [−0.002877, +0.001822], 22 reserved IDs; reports audited and four-page paper delivered. | [design](01_aversive_reactivity.md) |
| 02_emotional_carryover | Test cross-session repeatability of the subjective–EEG history-effect contrast. | done | Covariance +0.003496 [−0.010277, +0.015055], p=0.253; correlation weakly identified; J2 gain −0.005539 [−0.012251, +0.000938]. Audited, five-page paper delivered. | [design](02_emotional_carryover.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1–B5 | Resolved by 01_aversive_reactivity: audit, fixed comparison, specificity/dose controls, profiles, reserved evaluation and paper. | user project |
| B6 | Obtain independent state-stress and timestamped onset criteria for clinical/onset claims. | limits of 01_aversive_reactivity |
| B7 | Validate ECG detections and test richer representations on a newly reserved sample; do not tune on the completed test. | limits of 01_aversive_reactivity |
| B8 | Study entirely population-new stimuli, respiratory forecasting or probabilistic profile families in separately designed experiments. | optional extensions in the selected project |
| B9 | Resolved in 02_emotional_carryover: actual transition support, identifiability, precision, adjusted repeatability and grouped predictive checks completed. | user request 2026-11-09 |
| B10 | Obtain independent, more precise repeated measurements before claiming a stable personal subjective–EEG history contrast; account for reference-scale noise and the limited reliability of short-session slopes. | measured limits of 02_emotional_carryover |
