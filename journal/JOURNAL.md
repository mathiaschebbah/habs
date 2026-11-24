# HABS research journal

## Status

- **Project / dataset:** HABS multimodal emotional video recordings.
- **Goal:** Study individual emotional responses and subjective–EEG temporal repeatability with stimulus-aware controls, then deliver audited results and research papers.
- **Research direction:** the user-selected [project](../docs/research/PROJECT.md), supplied and confirmed on 2026-11-08. The historical arousal-only study is not the experiment being evaluated.
- **Last experiment:** 08_ssl_multimodal_jepa, done (2026-11-24). Full French report: [docs/research/ssl/RESULTS.md](../docs/research/ssl/RESULTS.md), bilingual figures: [docs/research/ssl/figures](../docs/research/ssl/figures). Previous submission papers (studies 04–07): [docs/paper/paper.pdf](../docs/paper/paper.pdf), [docs/paper/paper_fr.pdf](../docs/paper/paper_fr.pdf), and 2-page French synthesis [docs/paper/synthese_fr.pdf](../docs/paper/synthese_fr.pdf).
- **Last result:** Double dissociation on fresh sealed split `habs-ssl-2026-11-24-b` (2.30 H100 GPU-h across 5 pretrained EEG foundation models, EEGNet/EEGConformer/Scratch, and Multimodal `PhysioJEPA` + full ablations): (1) strong monotonic SSL scaling on objective cross-modal `EEG [0,5)s -> Body [5,10)s` forecasting in 86 unseen people (`REVE-base` 60k h: HR +0.255, SC +0.136, Resp +0.213 vs handcrafted +0.168 / +0.051 / +0.066), while `PhysioJEPA` beats matched `Scratch` on affect (`test ρ 0.096 vs 0.055`), reaches top PE-free temporal dynamics (`pre/vid 0.922`, `early/late 0.680`, `AoT 0.802`), and collapses to chance under temporal shuffling (`0.556 -> 0.484`); (2) **zero generalization gain** over Study 07 0-GPU linear baselines on double-unseen subjective stress (`S07 sequence+enriched 0.622`, `enriched_full 0.596` vs best FM `LaBraM 0.608` [gain +0.020 null], `EEGConformer 0.583`, `EEGNet 0.578`, `PhysioJEPA 0.559–0.560` / `[0,10)s` abl. `0.591`, all fusions null `−0.036` to `+0.008`). At ~53 h scale, latent JEPA does not beat raw-signal MAE (`0.558 / 0.573`), and adding 9.5 h dev resting-state windows is marginal (`+0.002 / +0.005`).
- **Study 03 result:** sealed within-person ρ 0.139 [0.083, 0.196] vs 0.083; gain +0.056 [+0.012, +0.101]. [Results](../docs/research/affect/RESULTS.md).
- **Study 02 result:** 53 paired IDs / 4,554 transitions; cross-day contrast covariance +0.003496 [−0.010277, +0.015055], p=0.253. [Results](../docs/research/carryover/RESULTS.md).
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
| 03_affect_decoding | Step-by-step model search for rating prediction on new people and new video themes. | done | 72 experiments, dev 0.030 → 0.105; sealed test 0.139 [0.083, 0.196] vs 0.083, paired gain +0.056 [+0.012, +0.101]. Four-page paper delivered. | [design](03_affect_decoding.md) |
| 04_stress_detection | Detect self-reported stress reactions from physiology in new people and new videos; separate stress from arousal. | done | Sealed AUC 0.605 [0.544, 0.668] vs 0.634 for the starting model; search gain did not transfer. Stress slows the heart and raises skin responses. Five-page paper. | [design](04_stress_detection.md) |
| 05_stress_timecourse | Time course, detection latency, emotion split, habituation and ultra-short HRV of stress reactions. | done | Brain first (0–1 s), heart and breathing 1–2 s, skin 4 s; detectable from 1–4 s in new people; heart slows for every negative emotion; 10-s RMSSD unreliable during videos. | [design](05_stress_timecourse.md) |
| 06_person_or_video | Split each rating and response into the part shared with other viewers of the same video and the person's own part; heartbeat-evoked EEG during stress. | done | Heart and breathing follow only the shared part (own valence 0.004 [−0.015, 0.021]); skin own arousal 0.023 vs shared 0.106; same-video AUC 0.517 vs 0.917 for the other viewers' ratings. Heartbeat-evoked EEG unchanged (d = 0.039 [−0.002, 0.078]). | [design](06_person_or_video.md) |
| 07_comprehensive_revision | Audit & purify stress criterion, test direct emotion & unsupervised models, delineate beat-by-beat ECG P-QRS-T and rich 4-phase EEG, test raw CURATED HEP without ICA, run non-adjacent fixed-video & 2×2 appraisal tests, map sensitivity EEG topography, and benchmark sequence architectures. | done | Criterion 89.2% neg rank 1 (97.2% ranks 1–3), opt 3/7; PR strictly null (−0.05 ms) & QTc Bazett artifact null after ΔRR; raw HEP = ICA (+0.038 vs +0.039); same-video AUC vs calm jumps 0.525 → 0.564 (2×2 paired HR −2.97 bpm, only α automatic); trial sensitivity in frontal β/γ & centroparietal δ (93% intact after ocular proxy); 4-phase enriched + Ridge V&A / 1D conv lift new-both sealed test 0.526 → 0.589 / 0.586 (dev 0.576, vs relaxed 0.656). | [design](07_comprehensive_revision.md) |
| 08_ssl_multimodal_jepa | Test whether pretrained EEG foundation models (Signal-JEPA, CBraMod, LaBraM, EEGPT, REVE-base) or a HABS-native Multimodal JEPA World Model (`PhysioJEPA`, EEG+ECG+EDA+Resp) learn transferable temporal neurophysiological representations beating Study 07 linear baselines. | done | Double dissociation on fresh sealed split `habs-ssl-2026-11-24-b` (2.30 H100 GPU-h): strong monotonic SSL scaling on unseen-person `EEG [0,5)s -> Body [5,10)s` forecasting (`REVE-base` HR +0.255, SC +0.136, Resp +0.213 vs S07 +0.168/+0.051/+0.066) and top PE-free temporal dynamics in `PhysioJEPA` (`pre/vid 0.922`, `early/late 0.680`, `AoT 0.802`, shuffle collapse `0.556→0.484`, affect ρ +73% vs matched Scratch), YET **no generalization gain** on double-unseen subjective stress over 0-GPU Study 07 linear baselines (`0.596–0.622` vs all FMs/neural/JEPA/fusions, paired gains `−0.036` to `+0.008`, null). MAE ties/edges JEPA at 53 h scale; raw latents encode 26%–53% subject-ID variance (cross-day ID up to 45.4%). | [design](08_ssl_multimodal_jepa.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1–B5 | Resolved by 01_aversive_reactivity: audit, fixed comparison, specificity/dose controls, profiles, reserved evaluation and paper. | user project |
| B6 | Obtain independent state-stress and timestamped onset criteria for clinical/onset claims. | limits of 01_aversive_reactivity |
| B7 | Validate ECG detections and test richer representations on a newly reserved sample; do not tune on the completed test. | limits of 01_aversive_reactivity |
| B8 | Study entirely population-new stimuli, respiratory forecasting or probabilistic profile families in separately designed experiments. | optional extensions in the selected project |
| B9 | Resolved in 02_emotional_carryover: actual transition support, identifiability, precision, adjusted repeatability and grouped predictive checks completed. | user request 2026-11-09 |
| B10 | Obtain independent, more precise repeated measurements before claiming a stable personal subjective–EEG history contrast; account for reference-scale noise and the limited reliability of short-session slopes. | measured limits of 02_emotional_carryover |
| B11 | Test time-resolved or differently filtered features in a new search with a fresh sealed split; never reuse the study-03 test for selection. | limits of 03_affect_decoding |
