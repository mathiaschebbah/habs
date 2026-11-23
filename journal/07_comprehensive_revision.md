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


## Outcomes and Provenance (2026-11-23)

### Backend extraction and GitHub Actions runs
- **Enriched feature extraction (`submit.yml` run `45090695249`, commit `ea8b707`):** completed in 13m 3s; Azure suffix `20261123T004720-2ea585`, producing `features/rich.parquet` (22,974 rows × 650 columns) and `docs/research/revision/results/rich_coverage.json`.
  - Hardware & data audits: single bipolar bilateral infraclavicular ECG lead (no multi-lead spatial ECG comparison possible); `flagged_outliers == 0.0` on 100% of the 22,974 audited trials; **96.57% inverted ECG polarity** auto-aligned by template-guided P/T detection (99.2% median clean QRS beats, 94.7–98.7% morphological coverage).
- **Study 07 analysis splits (`decoding.yml`, commit `b909c69`, 5-job `SOURCE_JOB` ending in `20261123T004720-2ea585`):**
  - `criterion` (balanced): run `45092371645` → `docs/research/revision/results/criterion.json`
  - `criterion_full` (ranks 1–3, `flagged_outliers`): run `45092366951` → `docs/research/revision/results/criterion_full.json`
  - `emotion_models` (balanced + unsupervised): run `45092374849` → `docs/research/revision/results/emotion_models.json`
  - `fixed_video` (2×2 matrix, non-adjacent contrasts, dose-response): run `45091493959` → `docs/research/revision/results/fixed_video.json`
  - `sensitivity` (6×5 topography, 62-ch, trait sensitivity): run `45091497148` → `docs/research/revision/results/sensitivity.json`
  - `ecg_morphology` (P-QRS-T, ΔRR controls, rich EEG, pre-ICA ocular proxy): run `45092356704` → `docs/research/revision/results/ecg_morphology.json`
  - `hep_raw` (CURATED raw CAR no-ICA vs PREPROCESSED ICA + exact ICA-removed diff): run `45092360238` → `docs/research/revision/results/hep_raw.json`
  - `sequence_models` (12-model benchmark on fresh split `habs-revision-2026-11-23`): run `45092363738` → `docs/research/revision/results/sequence_models.json`

### Key verified findings
1. **Criterion sanity check & purification:** inside `V <= 4 & A >= 6` ($n = 3,938$ cohorts 2–5), **89.2%** report a negative emotion at rank 1 (**97.2%** across ranks 1–3; 72.5% core acute threat: anger 25.7%, fear 23.9%, disgust 22.9%; sadness 14.9%; only 5.3% incoherent). Sweeping thresholds from `<5/>5` to `<=2/>=8` raises negative purity from 83.8% to 95.3% (incoherent 10.1% → 1.4%) and more than doubles skin conductance ($d = +0.143 \to +0.342$ at `2.5/7.5`) and delta ($+0.126 \to +0.302$), while halving cardiac deceleration ($-0.106 \to -0.041$) due to sympathetic cardiac recruitment at extreme arousal; `V <= 3 & A >= 7` maximizes detection (WP-AUC 0.549 [0.535, 0.563], SV-AUC 0.527 [0.510, 0.543]).
2. **Direct emotion models & unsupervised clustering:** fear shows full autonomic reactivity (HR $-0.20$, SC $+0.25$, resp $-0.11$, alpha $-0.12$); disgust is vagal/delta (HR $-0.20$, delta $+0.15$, SC null); sadness is physiologically flat across sensors. Balanced binary OvR AUCs range from 0.486 (anger) to 0.557 (relaxation) (`core_threat` 0.545; `all_negative` SV-AUC 0.532 [0.515, 0.547] with non-zero heart `beta_own = -0.032`); 5-class balanced accuracy 22.3% vs 20.0% chance. Unsupervised k-means/GMM fails to recover stress (`ARI < 0.009`, `NMI <= 0.0006`) or emotions (`NMI <= 0.0015`). Methodological note: class balancing (`class_weight="balanced"`) was critical to prevent theme-fold base-rate shift from producing spurious sub-0.50 concatenated AUCs.
3. **ECG P-QRS-T morphology & rich EEG dynamics:** RR lengthens ($+5.06$ ms, $d = +0.116$ [+0.071, +0.163]), whereas PR peak ($-0.05$ ms, $d = -0.002$), clinical PR ($+0.06$ ms, $+0.010$), pure PR segment ($-0.04$ ms, $-0.005$), QRS ($+0.007$), and T/R ($+0.025$) are strictly null before and after ΔRR adjustment — establishing fast vagal sinus braking without measurable adrenergic AV-node acceleration on 10-s clips. Apparent Bazett QTc shortening ($d = -0.092$, $-1.00$ ms) vanishes completely after ΔRR control ($-0.007$ [-0.035, +0.020]). New EEG descriptors show strong early 1-s temporal volatility (delta max jump $d = +0.151$ [+0.107, +0.193]; temporal SD $+0.091$ to $+0.101$), anterior-posterior late shift ($\delta +0.075$, $\alpha +0.070$), leftward temporal beta asymmetry ($-0.063$), and biphasic spatial reorganization (early homogenization `sp_iqr` $\theta -0.064$, reversing to post-clip focalization `sp_ent` $\alpha -0.076$).
4. **Raw CURATED vs ICA PREPROCESSED HEP:** exact match on 287 people (`CURATED` no-ICA $d = +0.0382$ [-0.0008, +0.0774] vs `PREPROCESSED` ICA $+0.0386$ [-0.0021, +0.0783]; ICA-removed diff $+0.0117$ [-0.0276, +0.0528], ECG field corr $-0.0044$). Both show the same tiny extreme `stress vs relaxed` shift (`+0.076*` raw vs `+0.072*` ICA). ICA did not destroy a hidden HEP signal.
5. **Fixed-video non-adjacent contrasts, 2×2 matrix & nonlinear dose-response:** on high-consensus stressful clips ($p_{v,-i} \ge 50\%$), 33.6% of viewers are borderline and only 2.5% ($n = 124$; 63 strict) are truly calm ("oklm"), so contrasting `stressed vs all else` diluted appraisal 93% of the time. Contrasting `stressed vs very calm` within the *same* video lifts same-video AUC from 0.525 to **0.564 [0.528, 0.602]** (heart 0.553, delta 0.564, skin $d = +0.197$). In the 2×2 matrix, calm viewers on stressful clips show zero bradycardia ($-0.49$ bpm [-3.16, +2.25]) and zero SCR rise ($+0.014\,\mu$S, identical to neutral clips), whereas paired within-person appraisal reaches $-2.97$ bpm ($d = -0.41*$) and delta $d = +0.45*$; only alpha desynchronizes automatically in calm viewers ($d = -0.33*$ to $-0.41*$). Dose-response curves confirm a step function in HR at valence = 5, a U-curve in HR on arousal (rebound at $A = 8$–$9$), and high-threshold alarms ($A \ge 7$) for skin and delta.
6. **EEG sensitivity topography & ocular control:** trial-level `beta_own` (negative valence) peaks in prefrontal fast rhythms (frontal gamma $-0.039*$, frontal beta $-0.035*$, Fp2/Fp1/AF7/F4) and centroparietal delta (central $-0.037*$, FC3/POz/Pz/C3); 93% of delta's `beta_own` survives adjusting for the pre-ICA Fp1/Fp2 blink proxy ($+0.0286 \to +0.0267*$). Trait sensitivity ($N = 314$, cohort-adjusted) correlates with resting HR ($r = +0.140*$), resting frontal/scalp gamma ($r = +0.165*$, $\rho = +0.185$), lower temporal theta ($r = -0.149*$), and female sex ($+0.172*$), without larger phasic autonomic responses.
7. **4-phase & sequence model benchmark (`habs-revision-2026-11-23`):** switching from 1-s bins ($p = 340$) to 4 phases ($p = 114$) boosts dev WP-AUC from 0.546 to 0.560 and double-unseen sealed test (`test_new_both`) from 0.526 [0.481, 0.570] to 0.570 [0.519, 0.618]. Adding EEG temporal volatility ($+30$ feats) lifts dev to 0.567 (SV 0.539) and EEG spatial homogeneity lifts `new_both` to 0.582 (`new_video` SV 0.542), whereas ECG P-QRS-T adds zero gain. Full enriched linear reaches **0.589 [0.540, 0.635]** on `new_both` ($+0.063$ over 1-s baseline); **continuous V&A Ridge (`enriched_full_ridge_va`)** is the best overall model (dev **0.576**, `new_both` **0.586**, `new_video` **0.558**, `new_person` **0.607**, vs relaxed **0.656**). Multi-scale 1D conv sequence reaches 0.569 dev and best SV-AUC on dev (0.541) and `new_both` (0.527); nonlinear MLP/GBDT heads do not outperform regularized linear models on unseen themes.

### Deliverables
- Upgraded French paper: `docs/paper/paper_fr.pdf` (11 pages, 7 figures, 4 tables)
- Upgraded English paper: `docs/paper/paper.pdf` (11 pages, 7 figures, 4 tables)
- Upgraded French synthesis: `docs/paper/synthese_fr.pdf` (2 pages, 3 figures)
- French results note & figures: `docs/research/revision/RESULTS.md`, `docs/research/revision/figures/`
