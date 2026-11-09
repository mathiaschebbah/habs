# HABS research journal

## Status

- **Project / dataset:** HABS multimodal emotional video recordings.
- **Goal:** Test whether physiology predicts individual aversive emotional reactivity beyond stimulus information and first-session calibration, then deliver verified results and a short research paper.
- **Research direction:** the user-selected [project](../docs/research/PROJECT.md), supplied and confirmed on 2026-11-08. The historical arousal-only study is not the experiment being evaluated.
- **Last experiment:** 01_aversive_reactivity, done; actual HABS evaluation and read-only audit completed.
- **Last result:** 22 reserved identifiers / 997 trials; macro-MAE M1=0.112443, M4=0.112826; gain −0.000383, 95% CI [−0.002877, +0.001822]. No demonstrated increment for the tested physiology. [Results](../docs/research/RESULTS.md), [paper](../docs/research/manuscript.pdf), [supplement](../docs/research/supplement.pdf).
- **Literature follow-up, 2026-11-09:** [systematized scoping review](../docs/research/literature/2026-11-09/REVIEW.md), 21 Exa searches and five complete Europe PMC queries; 28 core publications plus 29 contextual publications extracted. [Proposed question](../docs/research/literature/2026-11-09/QUESTION.md): cross-session reproducibility of the difference between subjective and EEG sequential effects. No new association or model has been run; novelty is conditional on the searched corpus, and usable transition support remains to be audited.
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

## Backlog

| # | Item | Source |
|---|---|---|
| B1–B5 | Resolved by 01_aversive_reactivity: audit, fixed comparison, specificity/dose controls, profiles, reserved evaluation and paper. | user project |
| B6 | Obtain independent state-stress and timestamped onset criteria for clinical/onset claims. | limits of 01_aversive_reactivity |
| B7 | Validate ECG detections and test richer representations on a newly reserved sample; do not tune on the completed test. | limits of 01_aversive_reactivity |
| B8 | Study entirely population-new stimuli, respiratory forecasting or probabilistic profile families in separately designed experiments. | optional extensions in the selected project |
| B9 | Audit consecutive-trial support and identifiability, then assess whether the subjective–EEG effect of recent emotional history is individually reproducible across J1/J2. Treat the opened data as exploratory; do not reuse the old reserved sample as a pristine confirmation. | [literature review and question](../docs/research/literature/2026-11-09/QUESTION.md), user request 2026-11-09 |
