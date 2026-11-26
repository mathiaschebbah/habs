# HABS Team 12 · Distress decoding and raw cardiac forecasting

Mathias Chebbah, Aurélien Saint-Grégoire and Ashley Russo. Université Paris-Saclay × HABS Neuro-AI Hackathon, Multimodal AI track. Mentor: Andrew Jeyathasan.

The publication revision studies 22,974 emotional-video trials from 314 participant IDs. It separates ranking self-reported distress from forecasting heart-rate change. A single person could receive several IDs across cohorts.

| Manuscript | English | Français |
| --- | --- | --- |
| Main article | [Paper](paper/submission_paper.pdf) | [Article](paper/submission_paper_fr.pdf) |
| Secondary analyses | [Supplement](paper/supplement.pdf) | [Annexe](paper/supplement_fr.pdf) |

## Findings

- For multimodal Ridge, the paired R4 minus R1 within-ID AUC contrast was −0.0072 [−0.0138, −0.0003] at equal training size and −0.0153 [−0.0226, −0.0087] with full training pools on the same test cells. REVE logistic regression changed by +0.0258 [+0.0137, +0.0382] under the matched design, but +0.0004 [−0.0088, +0.0104] when both trials in an AUC pair had to share the same fitted model. On that same sensitivity measure the multimodal Ridge contrast was −0.0129 [−0.0214, −0.0047].
- Removing the 164 explicit post-clip features changed the multimodal Ridge AUC from 0.5694 to 0.5655. The paired gain from retaining these features was +0.0039 [−0.0042, +0.0129]. A linear session-index predictor scored 0.5009 [0.4822, 0.5203].
- On 10,031 common trials from 205 IDs (203 evaluable for within-ID metrics), the raw EEG model after ECG regression reduced RMSE by 0.1085 bpm [0.0476, 0.1971] versus enriched linear Ridge, but by only 0.0139 bpm [−0.0084, 0.0370] versus the best body-only model, boosted trees. The linear enrichment itself worsened baseline RMSE. Held-out median absolute scalp ECG–EEG coupling did not decrease (0.1276 → 0.1311). A cortical origin is unresolved.

## Reproduce the artifacts

The source table for each finding, its population, its code revision and workflow are documented in the [Study 11 protocol](docs/study11_protocol.md) and [provenance manifest](results/study11/provenance.json). Publication-safe aggregates are versioned; the repository does not contain private trial matrices.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
python scripts/study11_tables.py
python scripts/study11_figures.py
pytest -q
```

Use Python 3.12+ and a LaTeX installation with Latin Modern, microtype, booktabs, graphicx and hyperref. In `paper/`, compile each of `submission_paper.tex`, `submission_paper_fr.tex`, `supplement.tex`, and `supplement_fr.tex` twice with `pdflatex`.

The experimental workflows use the team's authorized HABS GitHub identity and private data storage. The GPU job reads CURATED EEG. Its private outputs stay in team storage; the scoring job exports aggregate results. Reproduction from the original human recordings requires access granted by HABS. [Dataset and access conditions](https://huggingface.co/datasets/HABSteam/multimodal-eeg-emotion)

## Historical material

The [team synthesis](docs/SYNTHESE_EQUIPE12.md) and [notebook](notebooks/results_walkthrough.ipynb) document studies 01–10. `python scripts/reproduce.py` regenerates their four original graphics and summaries. The production analysis sources for those studies are preserved at commit `501d24b`. The pre-Study-11 paper is available at [abd9d69](https://github.com/mathiaschebbah/habs/tree/abd9d6970ed79825d72eec780d349b180cd3b398/paper). The local sample contains resting-state and artefact recordings; `process.py` remains the sample signal-audit entry point.
