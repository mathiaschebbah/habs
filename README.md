# HABS Neuro-AI Hackathon

**Build advanced AI models on real EEG data and contribute to the next generation of neurotechnology.**

Université Paris-Saclay · 2–30 November 2026 · [hackathon.habs.ai](https://hackathon.habs.ai/)

## Research study

**Final consolidated scientific article (Studies 04–10 + 39-paper literature synthesis):** [**Neurophysiologie, prévision cerveau–cœur, généralisation et monitoring passif du stress aigu chez 314 personnes** (FR PDF)](docs/paper/final_paper_fr.pdf) — integrates our complete empirical arc across 314 participants / 22,974 trials together with Study 10 ([protocol & execution log](journal/10_final_generalization_coherence_clinical.md), [French results](docs/research/study10/RESULTS.md), [aggregates](docs/research/study10/results), [figures](docs/research/study10/figures)). Run in `0.00 H100-h` (`35.8 min` CPU, run `45604993423`), Study 10 establishes: (1) `R1 → R4` generalization optimism (`0.5887 → 0.5688`, 70% from seen themes, `+0.041` pooled biometric leak on `REVE`) and R4 sensor ablation showing **EEG (`−0.028*`) and EDA (`−0.018*`)** as complementary pillars (`ECG_only = 0.508`); (2) cross-day `J1 → J2` transfer where **`EDA_only` drops `−0.039*`** (`0.560 → 0.523`) while EEG & multimodal hold steady, alongside `SCR` habituation (`−0.032*`) vs persistent vagal `RR`; (3) chronological `k`-shot (`k = 0..40`) & `J1 → J2` calibration proving naive supervised personalization collapses (`−0.081` to `−0.053`, `−0.040* / −0.069*` across days), eliminated by `EB-MAP` shrinkage, while **Study 09's causal cardiac `M2 − M0` gain replicates in 5/5 held-out cohorts under LOCO-CV (`+0.0107 [+0.0071, +0.0145]`, 68.8% improved)**; (4) three orthogonal Varimax autonomic–subjective dimensions on $N = 314 / n = 16,276$ (`F1 EDA`, `F2 Neuro-Auto` orthogonal to ratings, `F3 Subj-Cardio`) and individual coherence heterogeneity (`79% > 0`, `21%` discordant, intensity gain `+0.101*`); (5) passive 10 s → 100 s multi-episode aggregation lifting Cold-Start AUC to `0.680–0.698` (`Sens@Spec90%` doubled up to `53%`, verified without replacement); and (6) within-clip `Fear − Disgust` early `δ` vs late `EDA/α/γ` temporal dissociation.

**Earlier standalone papers (Studies 04–07 & Study 09):** [Studies 04–07 paper (EN)](docs/paper/paper.pdf), [in French](docs/paper/paper_fr.pdf), [2-page executive synthesis](docs/paper/synthese_fr.pdf); [Study 09 paper (FR)](docs/paper/study09_fr.pdf).

**Study 09, causal early-EEG predictive physiology of acute stress (`M2 − M0`), Studies 07–08 audit, REVE LoRA, Trait/State, oculometry & stacking:** [scientific article (FR PDF)](docs/paper/study09_fr.pdf), [pre-registered protocol & execution log](journal/09_eeg_predictive_physiology.md), [French results](docs/research/study09/RESULTS.md), [aggregates](docs/research/study09/results), [figures](docs/research/study09/figures).
Audited and fixed transductive normalization (`person_center_matrix`, inflating `dev` LogReg by `+0.0141`) and Study 08's bidirectional 13-s `REVE-base` future-EEG leak + 77% seen-theme shortcut (inflating `M1` by `+0.059` HR, `+0.042` SC, `+0.069` Resp). Exclusively inside `dev` (`habs-ssl-2026-11-24-b`, nested blocked `person × theme` CV, `test_new_both` treated as exposed, `1.40 H100-h` + `3.41 CPU-h`, bit-verified hard causal wall at `t = 5.000 s`): **(1) Primary `M2 − M0` (Cold-Start)**: early causal EEG adds a real, modest, idiosyncratic, artefact-resistant predictive gain on future cardiac dynamics (`ECG HR [5,10) s`: `M0 = 0.6849 → M2_reve_resid +0.0089 [+0.0054, +0.0125]`, `M2_reve_joint +0.0110 [+0.0065, +0.0156]` [71.4% of 224 subjects improved, $q_{\mathrm{FDR}} < 10^{-6}$], `5×5 +0.0120`, `M2_gbdt − M0_gbdt +0.0063 [+0.0021, +0.0102]`, pure increment `ΔHR +0.0182`, persists at 2-s `[7,10)` `+0.0089` and 5-s `[10,12)` `+0.0038` guard gaps, survives 9 artefacts `+0.0085`, 46 interior ch `+0.0070`, clean trials `+0.0110`, oculo `+0.0088`, prev trial `+0.0073`, null in `NC1 −0.0003` & `NC3 −0.0010`, destroyed by 1-s shuffle `NC4`, strictly idiosyncratic vs same-video other viewer `+0.0242` and within-video `+0.0065`); **strictly refuted on EDA (`−0.0097` / GBDT `−0.0155`) and respiration (`+0.0008`; nonlinear `M0_gbdt = 0.7036`, `M2_gbdt − M0_gbdt = −0.0008`)**. **(2) Secondary**: lightweight LoRA yields zero gain over frozen `REVE-base` (`−0.0069` at 21.7× GPU cost), while intra-subject ranking loss is decisive (`+0.0406` vs `norank`); EB shrinkage state denoising outperforms blind reference subtraction (`0.5605` vs `0.5551`, cuts ID $R^2$ `51.0% → 36.9%`, improves pooled AUC/Brier); 508-recording oculometry audit confirms `96.5%` coverage, binoc $r = 0.852$, and a `+121 px/s` gaze-speed step at `t = 0 s`, yet fails the strict pupil-mean gate globally and is at chance for stress (`0.5058`); conditional nested OOF stacking fails H5 on all tasks (`stress −0.0084`, `ECG −0.0011`, `EDA abstained`, `Resp −0.0057`).

**Study 08, self-supervised learning & Multimodal Physiological JEPA (`PhysioJEPA`):** [protocol and provenance](journal/08_ssl_multimodal_jepa.md),
[French results](docs/research/ssl/RESULTS.md), [aggregates](docs/research/ssl/results), [figures](docs/research/ssl/figures).
Evaluated five genuinely accessible pretrained EEG foundation models (`Signal-JEPA`, `CBraMod`, `LaBraM`, `EEGPT`, `REVE-base`), 3-seed end-to-end neural baselines (`EEGNet`, `EEGConformer`, `MultimodalConformer_Scratch`), and a HABS-native label-free Multimodal JEPA World Model (`PhysioJEPA`, EEG + ECG + EDA + respiration, 0.90M encoder) with full modality/window/strategy/data-scale ablations on a fresh sealed split (`habs-ssl-2026-11-24-b`, 2.30 H100 GPU-h). Result: a clear **double dissociation** — pretraining scale drives strong monotonic gains on unseen-person cross-modal physiological forecasting (`EEG [0,5)s -> Body [5,10)s`: `REVE-base` HR `+0.255`, SC `+0.136`, Resp `+0.213` vs handcrafted `+0.168 / +0.051 / +0.066`), and `PhysioJEPA` beats matched-capacity `Scratch` on continuous affect (`test ρ 0.096 vs 0.055`), seed stability (`±0.001`), and positional-encoding-free directed temporal dynamics (`pre/vid 0.922`, `early/late 0.680`, `AoT 0.802`, collapsing to chance `0.556 → 0.484` when shuffled); **yet neither pretrained foundation models nor `PhysioJEPA` nor their fusions deliver any statistically significant generalization gain over Study 07's 0-GPU linear baselines on double-unseen subjective stress** (`0.596–0.622`, all paired fusion gains `−0.036` to `+0.008`, 95% CIs spanning zero).

**Study 07, comprehensive revision:** [protocol and provenance](journal/07_comprehensive_revision.md),
[French results](docs/research/revision/RESULTS.md), [aggregates](docs/research/revision/results), [figures](docs/research/revision/figures).

**Study 06, the person or the video:** [protocol and outcome](journal/06_person_or_video.md),
[aggregates](docs/research/person/results), [figures](docs/research/person/figures).

**Study 05, time course:** [protocol and outcome](journal/05_stress_timecourse.md),
[aggregates](docs/research/timecourse/results).

**Study 04, can we detect stress from physiology?** [results](docs/research/stress/RESULTS.md),
[five-page paper](docs/research/stress/paper.pdf), [all 43 experiments](docs/research/stress/experiments.tsv).
A trial counts as a stress reaction when the participant rated it negative and activating. On the
sealed test (new people, new video themes) EEG plus heart, skin and breathing detect these reactions
above chance but weakly: within-person AUC 0.605 [0.544, 0.668]. A plain logistic regression did as
well as our best searched model. Stressful videos lower alpha power, raise skin conductance responses,
make breathing shallower and slow the heart.

**Study 03, rating prediction for new viewers and new videos:**
[results](docs/research/affect/RESULTS.md), [four-page paper](docs/research/affect/paper.pdf),
[all 72 experiments](docs/research/affect/experiments.tsv). Every scored prediction concerns a
person and a video theme the model never saw. On the sealed test (55 people, 21 themes, opened
once), within-person ρ between predicted and reported ratings went from 0.083 for the starting
model to 0.139 [0.083, 0.196]; paired gain +0.056 [+0.012, +0.101]. Code in [`decoding/`](decoding).

**Study 02:** [subjective–EEG carryover results](docs/research/carryover/RESULTS.md),
[five-page paper](docs/research/carryover/manuscript.pdf), and
[three-page supplement](docs/research/carryover/supplement.pdf).
On 53 paired identifiers / 4,554 transitions, adjusted personal contrast
repeatability was not established: covariance +0.003496, 95% model-based interval
[−0.010277, +0.015055], p=0.253. The frozen personal J1 correction did not establish
a J2 predictive gain. See [verification](docs/research/carryover/VALIDATION.md).

**Completed results:** [four-page paper](docs/research/manuscript.pdf),
[two-page supplement](docs/research/supplement.pdf),
[French summary](docs/research/RESULTS.md), and
[verified numeric evidence](docs/research/results/study_summary.json).
On 997 future trials from 22 reserved identifiers, multimodal physiology did not
establish an improvement over stimulus/context and personal calibration:
gain −0.000383, 95% paired interval [−0.002877, +0.001822] on the experimental 0–1 proxy.

The [literature review](docs/research/literature/2026-11-09/REVIEW.md), completed
on 9 November, documents 21 Exa searches plus five Europe PMC queries. Its
[next-question proposal](docs/research/literature/2026-11-09/QUESTION.md) concerns
cross-session reproducibility of subjective–EEG sequential effects; it was
subsequently executed and audited as study 02.

This repository now implements the [selected research project](docs/research/PROJECT.md):
predict individual aversive emotional reactivity beyond video context and personal
calibration. The [journal](journal/JOURNAL.md) records current evidence and the
[source audit](docs/research/KICKOFF.md) explains the release and execution constraints.
The entry point audits annotations, materializes native signal representations,
runs the real-data structural smoke test, selects models on development, freezes
their settings, evaluates reserved J2 and exports aggregate evidence and figures.
The local sample lacks video ratings and stops after its loading audit.

`Code checks` validates code on pushes. `HABS` remains a manual dataset execution.
Its downloadable artifact contains aggregate evidence and figures; individual
annotations and report datasets remain in the team's Azure results container.
The manual `audit_job` option recovers a completed job and checks saved reports on
a CPU runner, without another GPU submission. It selects the primary report
dependencies within the published transfer limit and preserves their original
absolute workspace path. See [validation and recovery](docs/research/VALIDATION.md).

Local environment: `python3 -m venv .venv`, then install `requirements.txt`,
`requirements-dev.txt` and the package with `.venv/bin/pip install -e .`.

You write `process.py`. Run it on your own machine against data you download, or
send it to one of our H100s when you want one. **The GPU is optional** — a lot of
good work in this hackathon will never need it.

---

## 0 · Three accounts

All free, all quick. Make them before anything else.

|   |   |
| - | - |
| [**GitHub**](https://github.com/signup) | Your team's code lives here, and it is what a GPU run is authorised against. |
| [**Hugging Face**](https://huggingface.co/join) | Where you download the dataset from. |
| **Slack** | Mentors, announcements, and where your team key comes back. We email you the invitation — there is no public join link. |

## 1 · Setup, once

|   |   |
| - | - |
| **Copy this repo** | **Use this template** → *Create a new repository*. Call it anything, public or private. |
| **Join Slack** | Accept the invitation we email you. Mentors, announcements and everyone else are there. Nothing to find: if it has not arrived, ask us. |
| **DM `@HABS Bot` your repo** | Send your repo link, or `your-name/your-repo`. One message per team. Your **team key** and **results link** come back in that DM. Tell us even if you never plan to use a GPU: it is also how we know where your results belong. |
| **Paste the key** | In [`.github/workflows/submit.yml`](.github/workflows/submit.yml), replace `PASTE-YOUR-TEAM-KEY-HERE`, keeping the quotes. Only needed for GPU runs. |

> **Do not rename or move the repository afterwards.** Your key is bound to its exact
> address, so renaming it silently breaks GPU runs. If you must, tell `@HABS Bot` and
> we will issue a new key.

## 2 · The data

📖 **[DATASET.md](DATASET.md)** — 355 participants, three tasks, EEG with ECG, skin
conductance, respiration, gaze and pupil size, plus trial labels.
**Read it before you write anything.**

|   |   |
| - | - |
| [`sample/`](sample) | P001, day 1 — already here. Enough to write code against. |
| [🤗 the full dataset](https://huggingface.co/datasets/HABSteam/multimodal-eeg-emotion) | Pull what you want onto your own machine: one file, one folder, or all 351 GB. |
| `/data` | The whole dataset, mounted read-only on the GPU. Nothing to download. |

Access is per person. Open the dataset page with a Hugging Face account, accept
the terms, and HABS approves your request — so ask early, not the night before.

```bash
pip install huggingface_hub && hf auth login

hf download HABSteam/multimodal-eeg-emotion --repo-type dataset --local-dir ./data \
    --include "CURATED/P001/*"                 # one participant
#   --include "CURATED/*/*resting-state*"      # one task, everyone
#   --include "*.csv"                          # the tables only
#   (no --include)                             # everything, 351 GB
```

This download is for working on your own machine. A GPU run needs none of it —
`/data` is already mounted there.

## 3 · Your code

[`process.py`](process.py) is the only file you have to change.

```python
def process(data_dir: Path, out_dir: Path) -> None:
```

We call it once. `data_dir` is the dataset root, read-only and identical for every
team. Everything you write into `out_dir` is kept and is yours.

Put dependencies in [`requirements.txt`](requirements.txt) — CUDA and PyTorch are
already in the image. For examples and setup instructions, see the
[starting notebooks](starting_notebooks/README.md).

## 4 · Run it on your machine

This is the normal way to work, and for many projects it is the only way you need.

```bash
pip install -r requirements.txt
pip install "habs-hack[storage] @ git+https://github.com/HABS-BRAIN/habs-hack@main"

habs run --data ./sample --out ./out
```

Same contract, no queue. Worth doing before every submission.

## 5 · Run it on a GPU — optional

Only if your work needs one: heavy training, or the whole 351 GB at once. Everything
else belongs on your own machine, where the loop is faster anyway.

**Actions** → **HABS** → **Run workflow**. That is the submission — pushing does not
spend GPU time, so commit as often as you like. Jobs share a pool of H100s with every
other team.

|   |   |   |
| - | - | - |
| **build your image** | red here means `requirements.txt` | seconds |
| **run on a GPU** | red here means your code; the log is your own output | minutes to an hour |


## 6 · Your results

Open them with your **results link**:

- **[Azure Storage Explorer](https://azure.microsoft.com/products/storage/storage-explorer)**
  → *Connect to a resource* → *Blob container* → *SAS URL* → paste.
  `out/` holds a folder per job, `logs/` the full output.
- or `export HABS_SAS='<your link>'`, then `habs status` · `habs watch` · `habs ls` ·
  `habs logs` · `habs get`.

---

**Stuck?** Contact your mentor on Slack.

Good luck!
