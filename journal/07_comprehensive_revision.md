# 07_comprehensive_revision: criterion validation, emotion-specific models, enriched cardiac/EEG dynamics, raw brain–heart control, and fixed-video appraisal tests

Written 2026-11-23, before running any new extraction or analysis. Directly
addresses all six methodological and conceptual critiques raised during team
review of studies 04–06.

## Motivation and reframing

1. **Broad neuroscientific & translational framing (replacing the surgical
   storytelling).** Rather than framing around operating-room stress detection,
   the study addresses two general questions in affective neuroscience and
   physiological monitoring: (a) descriptive physiological characterization of
   acute stress reactions (what varies, in what temporal order, with what
   cortical spatialization and homogeneity, and under which autonomic/central
   branches), and (b) rigorous validation of acute-stress detection setups,
   relevant from neurofeedback and experimental psychology to non-communicating
   clinical populations (pre-verbal pediatric pain assessment, locked-in
   syndrome, coma, and general anesthesia).

2. **Sanity check and purification of the stress criterion.** Our previous
   criterion (`rating_valence <= 4` and `rating_arousal >= 6`) was defined from
   continuous/discrete scales without verifying the rank-ordered distribution of
   participants' own first-chosen emotion (`self_reported_emotion_1`, cohorts
   2–5) inside the included stress set, nor testing how stricter thresholds
   purify the sample or alter statistical power and effect sizes.

3. **Direct modeling on top reported emotions & unsupervised structure.**
   Pooling heterogeneous negative emotions into a binary quadrant may mask
   category-specific trends (e.g., fear vs disgust vs anger vs sadness). We
   therefore systematically evaluate: (a) one-vs-rest and one-vs-non-negative
   binary classifiers per frequent `self_reported_emotion_1`, (b) a unified
   $(K+1)$-class multiclass classifier (`fear`, `anger`, `disgust`, `sadness`,
   `other`), (c) per-emotion shared vs individual-deviation decomposition, and
   (d) unsupervised clustering (PCA + k-means / GMM) of person-centred
   physiology tested for spontaneous alignment with emotions, stress quadrants,
   and stimuli.

4. **Richer features, 4-phase temporal resolution, ECG P-QRS-T delineation, EEG
   spatial homogeneity & power variation, and sequence-aware models.**
   - Four physiological epochs: **Reference** `[-3, 0)` s, **Early** `[0, 5)` s,
     **Late** `[5, 10)` s, **Post-exposure** `[10, 12)` s, with explicit
     transition deltas $t \to t+1$ (0 at reference; `early - ref`,
     `late - early`, `post - late`) alongside baseline-referenced changes.
   - Single-lead ECG (bilateral infraclavicular Lead I, 1,000 Hz; single lead,
     so multi-lead spatial comparison is unavailable) delineated beat by beat
     after automatic polarity alignment and QRS template quality filtering:
     **PR peak** (`P_peak -> R_peak`), **PR interval** (`P_onset -> QRS_onset`),
     **PR segment** (`P_offset -> QRS_onset`, directly indexing AV-node / NAV
     adrenergic vs vagal dromotropic modulation), **QRS duration**, **QT / QTc
     (Bazett & Fridericia)**, **RT peak**, and normalized **T-wave amplitude**.
   - EEG spatial homogeneity across the 62 scalp channels per band and phase
     (inter-channel SD, IQR, normalized Shannon entropy, inter-regional SD),
     anterior-posterior gradient, frontal and temporal hemispheric asymmetries,
     and within-phase temporal power variation (1-s SD, linear slope, max
     second-to-second step).
   - Sequence-aware classifiers evaluated on a **fresh person × theme split**
     (`SALT = "habs-revision-2026-11-23"`): linear baseline, 4-phase + explicit
     $\Delta(t\to t+1)$ linear model, non-linear GBDT (`HistGradientBoosting`),
     and sequence models (multi-scale temporal convolutional / recurrent
     dynamics over the multivariate sequence) — scored by both within-person
     AUC and same-video AUC.

5. **Brain–heart (HEP) empirical preprocessing control (`CURATED` vs
   `PREPROCESSED`).** HABS `PREPROCESSED/` applies Picard ICA + ICLabel
   (rejecting eye $\ge 0.75$ and muscle/line/channel $\ge 0.90$; heart is not
   explicitly rejected, yet ICA can still remove cardiac-covariant cortical
   activity). We extract R-locked EEG directly from unprocessed `CURATED/`
   (identical 62-channel montage excluding T9/T10, common-average referenced,
   0.5–30 Hz, **no ICA**) on the exact same R peaks as `hep.parquet`, and test:
   (a) raw `CURATED` HEP stress effect with and without ECG amplitude + RR
   covariates (at both 100 µV and 200 µV rejection thresholds), and (b) the
   exact ICA-removed signal (`CURATED - PREPROCESSED`) locked to R peaks.

6. **Reconceptualizing "person vs situation": individual stress sensitivity,
   all-or-nothing (on/off) vs graded responses, fixed-video distribution &
   non-adjacent contrasts, and EEG spatial correlates of sensitivity.**
   - Rename the two variance parts accurately: **normative / consensus
     response** (stereotyped inter-individual reaction mixing sensory input and
     shared human threat orienting) vs **idiosyncratic deviation / differential
     stress sensitivity** (propensity to appraise a stimulus as more or less
     stressful than peers viewing the same stimulus).
   - Inspect per-video stress-rate distribution across all 252 videos (skew /
     class imbalance) and restrict fixed-video tests to: (a) **polarizing /
     balanced videos** ($0.20 \le p_v \le 0.80$, both pair-weighted and
     unweighted per-video mean), and (b) **sharply separated non-adjacent
     contrasts inside the same video** — stressed (`V <= 4, A >= 6`) vs clearly
     calm/unbothered ("oklm": `V >= 5.5, A <= 4.5` and `V >= 5, A <= 5`), plus
     axis-isolated valence and arousal contrasts at fixed video!
   - **2 × 2 Stimulus Consensus × Viewer State table:** directly compare
     physiological changes of clearly calm ("oklm") viewers vs stressed viewers
     on consensus-stressful videos versus consensus-neutral videos. If calm
     viewers on consensus-stressful videos still decelerate/respond like
     stressed viewers, the response is automatic/stimulus-locked; if only
     viewers who cross the subjective stress threshold respond, it is an
     appraisal-gated on/off switch.
   - **Graded vs On/Off dose-response curves:** mean physiological responses
     across fine bins of valence (1–9), arousal (1–9), and consensus stress
     rate, comparing linear vs threshold/step vs quadratic fits.
   - **EEG spatial correlates of stress sensitivity** (anxiety neuroscience
     perspective): map regional (6 regions × 5 bands) and hemispheric
     asymmetry (`FAA`, temporal asymmetry) correlates of (a) trial-level
     idiosyncratic sensitivity (`beta_own`), and (b) participant-level trait
     stress sensitivity (overall individual stress-rating propensity vs
     baseline `ref` EEG topography and reactivity).

## Extraction and execution plan

- **Extraction job (`decoding/rich_extract.py`, `stage = "rich_features"`):**
  resumes audited inputs from `20261108T205355-ade0ef` and writes
  `features/rich.parquet` plus public coverage `public/rich.json`.
- **Analysis module (`decoding/revision.py`):**
  - Track A (runs immediately on existing tables): `--split criterion`,
    `--split emotion_models`, `--split fixed_video`, `--split sensitivity`.
  - Track B (runs once `rich.parquet` completes): `--split ecg_morphology`,
    `--split hep_raw`, `--split sequence_models`, `--split criterion_full`
    (adding `self_reported_emotion_2/3` and `flagged_outliers`).
- All statistics use 2,000-draw person or video bootstraps (95% percentile CIs).
  Only aggregate JSON files leave HABS/Azure storage.
