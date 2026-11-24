# 08_ssl_multimodal_jepa: can self-supervised latent world models learn transferable neurophysiological representations across individuals and stimuli?

Written 2026-11-24, after the H100 feasibility & inventory audit (run
`45292917269`, job `20261123T235620-00e2c8`, counts and hardware only) and
**before** extracting any foundation-model embeddings, training any neural or
self-supervised model, or inspecting any development or sealed-test scores.

## 1. Central scientific question and hypotheses

Studies 04–07 established that acute self-reported stress reactions
($V \le 4 \cap A \ge 6$) during 10-s videos produce a reproducible three-branch
physiological pattern (automatic occipital alpha desynchronization,
appraisal-gated vagal sinus deceleration without AV-node PR shift, and
high-arousal sudomotor/centroparietal-delta activation), yet regularized linear
models over 4-phase enriched summary features reach only modest double-unseen
generalization (Study 07: within-person AUC 0.589 linear / 0.586 Ridge V&A on
`habs-revision-2026-11-23`; nonlinear MLP/GBDT heads did not beat linear
models on unseen themes).

Study 08 tests the core computational-neuroscience hypothesis of **self-supervised
representation learning (SSL)** and **predictive latent world models (JEPA)** on
multimodal physiology:

> **Central question:** Can self-supervised learning — via both externally
> pretrained EEG foundation models and a HABS-native Multimodal Physiological
> JEPA World Model (`PhysioJEPA`) over raw EEG + ECG + EDA + Respiration —
> learn latent neurophysiological representations that transfer across unseen
> individuals and unseen video themes, capturing directed **temporal
> physiological dynamics** rather than merely static **participant identity**?

**Explicit constraints and epistemic discipline:**
1. **No video content:** video pixels and audio are not distributed in HABS
   (`DATASET.md`). Every model operates strictly on label-free physiology; the
   Multimodal JEPA is an autonomous physiological world model with no stimulus
   conditioning.
2. **Prediction vs correlation vs causality:** downstream ROC AUC and Spearman
   $\rho$ quantify out-of-sample predictive association; representation probes
   quantify correlational geometry (subject identity vs temporal phase/order vs
   cross-modal predictability). No causal neurophysiological mechanism is
   claimed from observational film viewing alone.
3. **Null-result criterion:** if neither pretrained EEG foundation models nor
   `PhysioJEPA` deliver a statistically reliable paired gain (95% bootstrap CI
   excluding zero) over Study 07's regularized linear physiological baselines on
   the fresh sealed test, that negative result is reported as the primary
   finding alongside GPU cost, statistical power limits, and representation
   diagnostics.

## 2. Feasibility & environment audit (run `45292917269`, 2026-11-24)

Verified prior to freezing this protocol (`docs/research/ssl/results/ssl_audit.json`):
- **Hardware:** NVIDIA H100 NVL (93.1 GB VRAM, CUDA 12.6, cuDNN 9.10, bf16),
  40 CPUs, 314.7 GB RAM, PyTorch 2.14.1+cu126. Outbound Hugging Face Hub
  connectivity verified (`HTTP 200`, 142 ms).
- **Channel integrity:** 506 / 508 audited `emotion-video` recordings use
  `PREPROCESSED` ICA-cleaned CAR EEG with the **exact same 62 canonical
  channels in identical order** (`Fp1..POz`, T9/T10 excluded, 0 missing
  channels); the remaining 2 fall back to `CURATED` CAR 62-ch.
- **Unlabeled SSL corpus (strictly dev participants only):** 416 `resting-state`
  `PREPROCESSED` recordings (100% of dev participants, 416/416 exact 62-ch,
  11.09 hours) plus dev-only `emotion-video` trials (52.9 hours across 15-s
  trial windows `[-3, +12) s`). Test participants' emotion-video AND
  resting-state recordings are **strictly excluded** from all SSL pretraining.

## 3. Frozen fresh split & zero-leakage rules

- **Fresh salt:** `STUDY08_SALT = "habs-ssl-2026-11-24-b"`, chosen solely on
  aggregate split-size balance from the counts-only audit (never touching
  scores or features) via `decoding.common.assign_splits`:
  - `dev`: **12,693 trials**, 228 people (208 stress-eligible, 124 multi-day),
    66 video themes (3×3 person-fold × theme-fold cross-fitting: person folds
    62 / 80 / 86; theme folds 25 / 17 / 24).
  - `test_new_both` (**primary sealed test**, unseen people × unseen themes):
    **1,481 trials**, 86 people (30 stress-eligible, 52 multi-day), 18 themes.
  - `test_new_person` (unseen people × seen themes): **4,934 trials**, 86 people
    (78 stress-eligible), 66 themes.
  - `test_new_video` (seen people × unseen themes): **3,866 trials**, 228 people
    (71 stress-eligible), 18 themes.
- **Zero leakage across subjects, videos, and sessions:**
  1. All recording days (`recorded_day` 1, 2, 3) of a participant share the same
     `participant_id` bucket — zero session leakage.
  2. All clips sharing a `video_theme` share the same theme bucket — zero
     stimulus-theme leakage.
  3. **Zero transductive SSL leakage:** in 3×3 dev CV, SSL pretraining is
     **fold-isolated** (each dev person-fold $p \in \{0, 1, 2\}$ is evaluated
     with an encoder pretrained strictly on the other two dev person-folds!).
     For the one-time sealed test, SSL pretraining uses **strictly `dev`
     participants and `dev` trials** (plus `dev`-only resting-state windows for
     the `+rest` ablation) — zero test participants and zero test themes ever
     enter pretraining or scaler/PCA fitting.
  4. **Sealed test discipline:** all architecture, normalization, masking,
     window, and fusion choices are selected exclusively on `dev` 3×3 CV. The
     sealed test is evaluated once on the frozen comparison table.

## 4. Audit of genuinely accessible pretrained EEG foundation models

All five models are publicly downloadable without gating on Hugging Face Hub
and audited against HABS's exact 62-channel 10-10 CAR montage:

| Model | Venue / Source | Params | Pretraining objective & corpus | Native rate & window | Input scaling | HABS 62-ch match |
|---|---|---:|---|---|---|---|
| **Signal-JEPA** (`braindecode/signal-jepa`) | Guetschel et al., 2024 | 3.4 M | Spatial-block JEPA (Lee2019, 62 ch) | 128 Hz, 1-s tokens | Raw $\mu\text{V}$ | **47 / 62** (`pretrain_aligned`) |
| **CBraMod** (`braindecode/cbramod-pretrained`) | Wang et al., ICLR 2025 | 4.9 M | Criss-cross masked EEG recon (>9,000 h TUEG) | 200 Hz, 1-s patches | $\mu\text{V} / 100$ | **62 / 62** (montage-agnostic) |
| **LaBraM** (`braindecode/labram-pretrained`) | Jiang et al., ICLR 2024 | 5.8 M | VQ Neural Spectrum Prediction (~2,500 h incl. SEED/Emobrain) | 200 Hz, 1-s patches | $\mu\text{V} / 100$ | **62 / 62** |
| **EEGPT** (`braindecode/eegpt-pretrained`) | Wang et al., NeurIPS 2024 | 25.3 M | Dual MAE + EMA latent alignment (mixed incl. SEED) | 256 Hz, 4-s windows (patch 64) | CAR, mV ($\mu\text{V}\times 10^{-3}$) | **54 / 62** pretrained (58 in `CHANNEL_DICT`; AF7/AF8/PO5/PO6 untrained; F9/F10/P9/P10 absent) |
| **REVE-base** (`brain-bzh/reve-base`) | El Ouahidi et al., NeurIPS 2025 | 69.0 M | 4D Fourier-pos MAE (>60,000 h, 92 datasets, 25k subj.) | 200 Hz, 1-s patches (step 0.9 s) | Per-rec z-score, clip $\pm 15\sigma$ | **62 / 62** exact case in `reve-positions` |

For every pretrained model we extract phase-resolved frozen embeddings
`(z_ref [-3, 0), z_early [0, 5), z_late [5, 10))` across all 22,974 trials,
enabling BOTH raw (`z_vid`) and baseline-referenced (`z_early - z_ref`,
`z_late - z_ref`, `z_late - z_early`) probing, representation geometry
diagnostics, and multimodal fusion with Study 07 peripheral / enriched features.

## 5. Model families and ablations

1. **Study 07 linear & enriched physiological baselines** (re-evaluated on the
   exact Study 08 split `habs-ssl-2026-11-24-b`): `baseline_1s_logreg` ($p=340$),
   `phase4_logreg` ($p=114$), `enriched_full_logreg` ($p=448$),
   `enriched_full_ridge_va` ($p=448$), `sequence_plus_enriched_logreg` ($p=1068$),
   plus EEG-only and Body-only Study 07 linear baselines.
2. **Pretrained EEG foundation models** (frozen embeddings → standardized
   linear LogReg & V&A Ridge probes, raw vs baseline-delta, plus multimodal
   fusion with Study 07 peripheral features): `Signal-JEPA`, `CBraMod`,
   `LaBraM`, `EEGPT`, `REVE-base`.
3. **End-to-end supervised neural baselines** (3 seeds `0, 1, 2`, strict 3×3 dev
   CV + sealed test):
   - `EEGNet` (Lawhern et al., 2018; 2.2k params) on 62-ch 128 Hz EEG.
   - `EEGConformer` (Song et al., 2023; 237k params) on 62-ch 128 Hz EEG.
   - `MultimodalConformer_Scratch` (exact `PhysioJEPA` encoder backbone, 0.90M
     params, trained supervised from scratch with zero SSL pretraining —
     matched-capacity control!).
4. **HABS-Native Multimodal Physiological JEPA World Model (`PhysioJEPA`)**:
   - Compact 0.90M-param encoder (1.53M total trainable with 2-layer cross-modal
     latent predictor) operating on 1-second multi-rate tokens across 4
     modalities: 62-ch EEG (128 Hz, dual temporal-conv + differentiable log-FFT
     patch tokenizer), 2-ch polarity-aligned ECG + QRS envelope (128 Hz), 2-ch
     EDA conductance + phasic derivative (16 Hz), 2-ch respiration + Hilbert
     envelope (16 Hz), with explicit `[MISSING]` token masking (zero artificial
     EDA imputation).
   - EMA target encoder ($\tau: 0.996 \to 1.0$) + Smooth-L1 latent prediction +
     VICReg variance/covariance anti-collapse regularization.
   - Evaluated via **frozen linear probes** (LogReg + V&A Ridge) AND
     **lightweight fine-tuning** across **3 seeds (`0, 1, 2`)** on `dev`
     (fold-isolated!) and sealed test, plus fusion with Study 07 features.
5. **Systematic ablations (all on dev 3×3 CV; key variants also reported on
   sealed test after freezing):**
   - **Modality ablation:** `all (4 mod)` vs `eeg_only`, `body_only`,
     `eeg+ecg`, `eeg+eda`, `eeg+resp`.
   - **Temporal window ablation:** `[0, 5) s` (early), `[0, 10) s` (video),
     `[-3, 10) s` (pre+video), `[-3, 12) s` (full 15 s).
   - **Pretraining strategy ablation:** `jepa_crossmodal` (temporal + brain↔body
     masking) vs `jepa_future` (causal world-model rollout) vs `jepa_masked`
     (temporal block masking) vs `mae_recon` (raw signal reconstruction) vs
     `jepa_raw_norm` (fixed EEG scaling testing the Identity Shortcut) vs
     `jepa_plus_rest` (adding 11.1 h dev-only unlabeled resting-state windows)
     vs `scratch` (no SSL).
   - **Data-size scaling ablation:** 10%, 25%, 50%, 100% of dev participants
     (subsampled strictly by participant ID).

## 6. Evaluation metrics and representation probes

- **Downstream affective transfer:**
  - Acute stress (`V <= 4 & A >= 6`): within-person ROC AUC + 95% person
    bootstrap CI (2,000 draws), same-video ROC AUC (all viewers & vs calm
    `V >= 5 & A <= 5`) + 95% video bootstrap CI, specificity controls
    (`vs_excitement`, `vs_calm_negative`, `vs_relaxed`, `strict 3/7`), and
    paired gain vs `enriched_full_ridge_va` (95% paired bootstrap CI).
  - Valence & arousal: within-person Spearman $\rho$ + 95% CI, within-person
    binary AUC (`<=4` vs `>=6`) + 95% CI, pooled `>5` AUC.
  - Seed variability: mean ± SD across 3 seeds (`0, 1, 2`).
- **Representation dynamics vs participant-identity diagnostics:**
  1. Subject identity fingerprint: fraction of variance explained by person ID
     ($R^2_{\text{subject}}$), within-session chronological subject-ID accuracy
     (top-1 / top-5), strict cross-day (Day 1 → Day 2) subject-ID accuracy, and
     within- vs between-person cosine similarity — contrasted between raw
     embeddings (`z_vid`) and baseline-referenced delta embeddings (`z_vid - z_ref`).
  2. Directed temporal dynamics (trained on `dev` people, evaluated on strictly
     unseen test people): `pre_vs_video` AUC, `early_vs_late` AUC,
     `arrow_of_time` forward-vs-reversed `(z_early, z_late)` AUC, and
     **temporal-shuffle degradation test** (shuffling the 1-s time axis at test
     time to verify genuine reliance on temporal ordering).
  3. Cross-modal predictive coupling: does early `[0, 5) s` EEG latent state
     linearly predict late `[5, 10) s` cardiac deceleration, SCR rise, and
     breathing depth on unseen people?
  4. Anti-collapse health: Roy-Vetterli effective rank of `z_vid` and `z_delta`.
- **Compute & GPU cost accounting:** exact wall-clock seconds, GPU-seconds,
  peak VRAM (GB), parameter counts, and estimated H100 cost per model family.

## 7. Execution plan

- **GPU Job 1 (`decoding/ssl_fm_extract.py`, `stage = "ssl_fm_extract"`):**
  updates `requirements.txt` once with `braindecode==2.1.0`, `transformers`,
  `safetensors`, `huggingface-hub`, `einops`; extracts frozen phase-resolved
  embeddings for all 22,974 trials across all 5 pretrained EEG foundation models
  (`Signal-JEPA`, `CBraMod`, `LaBraM`, `EEGPT`, `REVE-base`) in one pass over
  the 508 recordings; saves private `/out/features/fm_embeddings.parquet`
  (< 250 MB) and public runtime/coverage telemetry `/out/public/fm_extract.json`.
- **GPU Job 2 (`decoding/ssl_train.py`, `stage = "ssl_jepa_suite"`):**
  loads the 22,974 multimodal 15-s trials + dev-only resting-state windows once
  into H100 node RAM (~5.5 GB), runs the complete controlled GPU benchmark
  (`EEGNet`, `EEGConformer`, `MultimodalConformer_Scratch`, `PhysioJEPA`
  fold-isolated 3×3 dev CV + 3-seed sealed test + all modality/window/strategy/
  data-scale/shuffle ablations + representation diagnostics), writes private
  `/out/features/jepa_embeddings.parquet` and public complete aggregate metrics
  `/out/public/ssl_jepa_suite.json`.
- **CPU Analysis Job (`decoding/ssl_study.py` via `decoding.yml`):**
  joins Study 07's existing tables (`trials.parquet`, `timecourse.parquet`,
  `rich.parquet`) with Job 1's `fm_embeddings.parquet` and Job 2's
  `jepa_embeddings.parquet`; evaluates ALL Study 07 baselines, ALL 5 pretrained
  foundation models (raw, delta, person-centered, and fused with Study 07
  peripheral/enriched features), and `PhysioJEPA` fusion + cross-modal
  forecasting + complete paired bootstrap gains against Study 07 on the exact
  frozen split `habs-ssl-2026-11-24-b`.
