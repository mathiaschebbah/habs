# Multimodal neurophysiology of acute distress (Team 12)

**Université Paris-Saclay x HABS Neuro-AI Hackathon 2026 (Multimodal AI track)**
Authors: Mathias Chebbah, Aurélien Saint-Grégoire, Ashley Russo (Mentor: Andrew Jeyathasan).

## Research question

How do cortical activity (EEG) and peripheral autonomic responses (ECG, EDA, respiration) unfold during 10-second acute distress episodes, does early EEG prospectively forecast future autonomic reactivity beyond past body dynamics, and what limits passive multimodal stress monitoring across unseen people, unseen video themes, days, and cohorts?

We analyze all 314 healthy adults and 22,974 aligned trials (508 recordings, 252 clips, 84 themes, 5 cohorts, 169 multi-day participants) from the HABS dataset under nested person-by-theme cross-validation, fold-isolated scaling, and forward-only post-release filtering at `t = 5.000 s`. Eye-tracking and pupillometry streams are excluded from primary modeling per the HABS organizer advisory documenting temporal misalignment with other modalities.

## Headline results

| Domain | Key finding | Verified metric |
| --- | --- | --- |
| Acute stress ECG morphology | 96.6% of Lead I ECG recordings are inverted; after polarity correction, distress is consistent with vagally mediated cardiac deceleration | RR `d = +0.116`, PR segment `d = -0.005`, QTc Bazett `-0.088 -> -0.007` after RR control; raw and ICA HEP both null (`+0.038` vs. `+0.039`) |
| Prospective brain-to-heart (`t = 5.000 s` boundary) | Early EEG (`REVE-base`) improves future heart rate (`[5, 10) s`) beyond past body dynamics (`M0 r = 0.6849`) | `M2_joint +0.0110` (71.4% improved, `q < 1e-6`), survives 9 artefacts (`+0.0085`), same-video other-viewer control (`+0.0242`), and 5/5 cohorts under LOCO-CV (`+0.0107`); no incremental benefit detected for EDA (`-0.0097`) or respiration (`+0.0008`) |
| Model benchmark & generalization (`R1 -> R4`) | Zero-GPU 4-phase Ridge matches or outperforms 5 EEG foundation models, `PhysioJEPA`, and LoRA fine-tuning | Audited Cold-Start dev AUC `0.5709` (`[0.5565, 0.5849]`); 70% of random-split optimism (`0.589 -> 0.569`) comes from seen themes |
| Longitudinal drift & `k`-shot personalization | One-week transfer degrades peripheral-only models; naive personal retraining collapses | `EDA_only` drops `-0.039` AUC across 1 week while EEG holds steady; naive `k`-shot drops `-0.081` at `k = 5`, while `EB-MAP` prevents collapse |
| Divergence & simulated passive monitoring | Varimax separates EDA, automatic EEG-cardio, and subjective ratings (21% discordant); aggregating discrete episodes lifts sensitivity | Integrating Cold-Start scores across `K = 1..10` discrete clips (`10-100 s` cumulative; disjoint check verified for `K <= 5`) raises AUC to `0.680-0.698` and sensitivity at 90% specificity from `21%` to `42-53%` |

## Key deliverables

- **Submission paper (5 pages, 4 figures, 2 tables):** `paper/submission_paper.pdf` (`paper/submission_paper.tex`)
- **Interactive results walkthrough:** `notebooks/results_walkthrough.ipynb`
- **Complete team synthesis & 10-minute presentation script (French):** `docs/SYNTHESE_EQUIPE12.md`
- **Verified aggregate outputs (zero individual data):** `results/` (5 JSON files; full exploratory history across Studies 01-10 is preserved in git commit `501d24b`)

## Minimal codebase map

The repository is refactored to six focused modules (1,825 lines of package code total) so reviewers and teammates can audit every formula directly:

```text
src/habs_stress/
  signals.py          # GSR calibration, Lead I polarity auto-fix, P-QRS-T, forward-only t = 5.000 s boundary
  protocol.py         # Person x theme splits, nested CV, Cold-Start vs. causal online EB, bootstrap, BH-FDR
  physiology.py       # Distress contrasts, within-person d/AUC, RR-adjusted ECG, fixed-video appraisal
  forecasting.py      # Prospective M0/M1/M2 forecasting, 9-artefact residualization, same-video control
  generalization.py   # CORAL, chronological k-shot EB-MAP, Varimax coherence, simulated 10 s -> 100 s monitoring
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
