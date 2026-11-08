# HABS research journal

## Status

- **Project / dataset:** HABS multimodal emotional video recordings.
- **Goal:** Test whether physiology predicts individual aversive emotional reactivity beyond stimulus information and first-session calibration, then deliver verified results and a short research paper.
- **Research direction:** the user-selected [project](../docs/research/PROJECT.md), supplied and confirmed on 2026-11-08. The historical arousal-only study is not the experiment being evaluated.
- **Last experiment:** 01_aversive_reactivity, ready for frozen development selection and reserved evaluation after successful real-data signal preparation.
- **Last result:** full data audit: 22,974 aligned rated trials, 314 identifiers; cohort 2 has 76 eligible paired people, with 26 reserved before inspecting their response distributions. No predictive result yet.
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
- **Signal evidence:** [feature summary](../data/feature_summary.json), [successful preparation run](https://github.com/mathiaschebbah/habs/actions/runs/42526673694). Cohort 2 has 7,795 EEG-usable trials and 3,672 complete multimodal observations. Out-of-calibration EDA is marked missing in 2,742 trials. No predictive result yet.

## History

| Stem | Intent | Status | Headline result | Design note |
|---|---|---|---|---|
| 01_aversive_reactivity | Test incremental physiology beyond stimulus and J1 calibration. | approved | Implementing after measured EDA. | [design](01_aversive_reactivity.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Reconcile video events and ratings, audit the composite distribution and usable two-session sample. | user project, phases 1–2 |
| B2 | Freeze the participant split and compare M0–M4 on identical target trials using participant-macro MAE. | user project, H1 |
| B3 | Evaluate specificity at comparable arousal and linear versus nonlinear dose-response on development data. | user project, H2–H3 |
| B4 | Test continuous versus clustered physiological profiles, their inter-session stability and predictive utility if data quality permits. | user project, H4–H6 |
| B5 | Run the frozen final evaluation and write a 2–5 page paper with at least two figures. | user request; hackathon guidelines |
