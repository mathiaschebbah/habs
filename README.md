# Multimodal neurophysiology of acute distress (Team 12)

**Université Paris-Saclay x HABS Neuro-AI Hackathon 2026 (Multimodal AI track)**
Authors: Mathias Chebbah, Aurélien Saint-Grégoire, Ashley Russo (Mentor: Andrew Jeyathasan).

## Research question

How do cortical activity (EEG) and peripheral autonomic responses (ECG, EDA, respiration) unfold during 10-second acute distress episodes, how do leakage controls and cross-day drift constrain multimodal decoding, and does early EEG add prospective information about future autonomic reactivity beyond past body dynamics?

We analyze 314 healthy adult participant IDs (of 317 labeled; 22,974 aligned trials, 508 recordings, 252 clips, 84 themes, 5 cohorts, 169 multi-day IDs) from the HABS dataset (`HABSteam/multimodal-eeg-emotion`), combining 62-channel EEG (excluding T9/T10), ECG, EDA, and respiration. Eye-tracking and pupillometry streams are excluded per the HABS organizer advisory documenting temporal misalignment. All post-audit modeling stays strictly inside the development split (`N = 228` IDs, 12,693 trials, 66 themes) under nested ID-by-theme cross-validation, fold-isolated scaling, and a post-release `t = 5.000 s` boundary for exploratory forecasting.

## Headline results

| Domain | Key finding | Verified metric |
| --- | --- | --- |
| Acute stress ECG morphology | Lead I polarity aligned by recording (96.6% of trial rows flagged); distress shows cardiac deceleration without detected AV-node or RR-independent QT shifts | RR `d = +0.116`, PR segment `d = -0.005`, Bazett QTc standardized `beta = -0.088 -> -0.007` after `dRR` adjustment; raw CURATED (`+0.038 [-0.001, +0.077]`) and ICA HEP (`+0.039 [-0.002, +0.078]`) both span zero |
| Exploratory prospective EEG-to-heart (`t = 5.000 s` post-release cutoff) | Trial-matched early EEG adds a small increment to future heart rate change (`HR[5,10) - HR[-3.5,0)`) over past body baseline (`M0 r = 0.685`, RMSE `5.540 bpm`) | `M2_resid +0.009` (`-0.077 bpm` RMSE), `M2_joint +0.011` (`-0.086 bpm` RMSE, 71.4% with `dr > 0`), internal 5-cohort LOCO (no theme mask) `+0.011` (positive in 5/5 cohorts; 3/5 CIs > 0); degrades EDA (`-0.010` resid, `-0.030` joint); no detectable change on respiration (`+0.001`). Upstream zero-phase FIR and whole-recording ICA leave strict end-to-end causality open |
| Model benchmark & generalization (`R1 -> R4`) | After removing global centering (`+0.014`) and bidirectional seen-theme leaks, single-trial ROC-AUC stays modest across architectures | Cold-Start R4 within-ID AUC: 4-phase Ridge (`[-3,12) s`) `0.569 [0.553, 0.584]`, `PhysioJEPA` `0.557`; EEG-only spans `0.526` (`S-JEPA`) to `0.572` (`EEGNet`, `[0,10) s`, 3-seed ens.) with `REVE-base` `[0,10) s` at `0.561`; across-viewer fixed-video R4 weighted AUC is `0.525`; masking themes represents 70-79% of the R1-R4 drop (`0.589 -> 0.569`, with training volume varying) |
| Longitudinal drift & `k`-shot adaptation | Across 1 week, peripheral-only decoding degrades; naive personal Ridge VA retraining lags Cold-Start | `EDA_only` drops `-0.039 [-0.073, -0.009]` (`N = 103` paired; unpaired `0.560 -> 0.523`), `Resp_only` drops to `0.501`, with no detected drop for `EEG_only` (`+0.025 [-0.006, +0.055]`); naive personal retraining lags through `k = 40` (`-0.053 [-0.087, -0.021]`), while `EB-MAP` stays flat (`-0.001`) |
| Varimax PCA, coherence & offline known-class averaging | Varimax-rotated PCA separates 3 orthogonal axes (53.3% variance); individual coherence stability is inconclusive; known-class averaging tracks Gaussian noise reduction | Neuro-autonomic coherence median `rho = +0.111` (21.0% `<= 0`; 1-week `r = 0.066 [-0.087, +0.211], p = 0.39`). Offline averaging of `K = 10` known-class clips yields AUC `0.680` (standard; per-ID oracle sensitivity at 90% spec `21.0 -> 41.8%`), `0.692` (threat; `27.0 -> 45.5%`), and `0.698` (strict 3/7 vs. relaxed; `31.5 -> 52.8%`) |

## Key deliverables

- **Submission paper (5 pages, 4 figures, 2 tables):** `paper/submission_paper.pdf` (English) and `paper/submission_paper_fr.pdf` (French)
- **Interactive results walkthrough:** `notebooks/results_walkthrough.ipynb`
- **Complete team synthesis & 10-minute presentation script (French):** `docs/SYNTHESE_EQUIPE12.md`
- **Verified aggregate outputs (zero individual data):** `results/` (5 JSON files; full exploratory history across Studies 01-10 is preserved in git commit `501d24b`)

## Minimal codebase map

The repository is refactored to six focused modules (~1,825 lines of package code total) so reviewers and teammates can audit every formula directly:

```text
src/habs_stress/
  signals.py          # GSR calibration, Lead I polarity alignment, P-QRS-T, forward-only t = 5.000 s boundary
  protocol.py         # ID x theme splits, nested CV, Cold-Start vs. causal online EB, bootstrap, BH-FDR
  physiology.py       # Distress contrasts, within-ID d/AUC, RR-adjusted ECG, fixed-video appraisal
  forecasting.py      # Exploratory M0/M1/M2 forecasting, 9-artefact residualization, permuted same-video control
  generalization.py   # CORAL, chronological k-shot EB-MAP, Varimax PCA/coherence, offline K-episode averaging
  figures.py          # Deterministic generation of Figures 1-4 from results/*.json
scripts/reproduce.py  # One-command reproduction of paper tables and figures
process.py            # Minimal HABS platform entrypoint tested against sample/
tests/                # 14 fast unit and contract tests (< 10 s)
```

## Quickstart and reproduction

Python 3.12+ is required. No GPU or 385 GB dataset download is needed to reproduce all tables, figures, and tests locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .

# 1. Reproduce all paper summary tables and Figures 1-4
python scripts/reproduce.py

# 2. Run linting and all 14 unit + contract tests (including local sample/ audit)
ruff check src scripts tests process.py
pytest -q
```

To run the HABS platform audit entrypoint directly against the included sample participant:

```bash
python -c "from pathlib import Path; from process import process; process(Path('sample'), Path('out'))"
```
