# autoresearch: affect from physiology, for new people watching new videos

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
An agent edits one file, scores it on a fixed objective, keeps what helps,
discards what does not, and repeats.

## The question

Can EEG and peripheral physiology recorded during a 10-second video predict how
that viewer rates it (valence, arousal) when both the viewer and the video theme
are new to the model? Other teams report AUCs near 0.58. With splits by person
only, the same video appears in training and evaluation, so a model can score by
recognising the stimulus, not the viewer's response. The harness removes that
route: every scored prediction concerns an unseen person and an unseen theme,
and stimulus identity columns, including `start_marker`, are hidden.

## Files

- `prepare.py`: data loading, the person × theme splits, the 420-second fit
  budget, the metric. Read-only. CI takes it from `main`, never from the
  experiment commit.
- `train.py`: the only file you edit. It exposes
  `fit_predict(train, test) -> DataFrame[rating_valence, rating_arousal]`.
- `extract.py`: the fixed feature table, built once on the HABS GPU backend.
- `requirements.txt`: the only packages `train.py` may import.

## Features available

One row per audited trial (~23k trials, ~314 IDs, 5 cohorts):

- `bp_vid_<ch>_<band>`, `bp_pre_<ch>_<band>`: log10 band power (delta, theta,
  alpha, beta, gamma) for 62–64 EEG channels, over the 10 s video and the 2 s
  before onset. ICA-cleaned `PREPROCESSED` EEG when available (`eeg_source`).
- `erp_<ch>_<k>`: mean amplitude in 200 ms bins 0–1 s after onset, baseline
  corrected.
- `cov_<a>_<b>`: 1–45 Hz covariance of 20 channels over the video (SPD matrix,
  upper triangle).
- `ecg_hr`, `ecg_change`, `eda_log_level`, `eda_slope`, `eda_log_sd`,
  `eda_pre_log_level`, `resp_amp`, `resp_rate`, validity flags.
- Metadata visible at prediction time: `participant_id`, `cohort`,
  `recorded_day`, `event_order`, `age`, `gender`, `eeg_source`.

`train` additionally has the ratings, `video` and `theme` (useful for grouping,
useless as predictors of unseen themes). Label-free use of a test person's own
trials is allowed (for example z-scoring features within person): it assumes a
batch of unlabelled trials from the new person, and the paper must say so.

## The objective

`score` = mean over valence and arousal of the within-person Spearman
correlation between prediction and rating. Higher is better; chance is 0. It
ignores how each person uses the scale, so it measures trial-level decoding.

Run an experiment:

```bash
git commit -am "<idea>" && git push origin autoresearch/nov19
gh workflow run autoresearch.yml --ref main -f split=dev \
  -f refs="$(git rev-parse HEAD)" -f source_job=<extraction job suffix>
```

Several SHAs can be passed at once; each is scored in parallel. Read the result
in the run summary (`score:` line) or `gh run view <id> --log | grep score:`.

## Rules

- Edit `train.py` only. No new packages. No network access.
- `fit_predict` must finish all nine development folds within 420 s on a
  4-vCPU runner, or the run counts as a crash.
- Never read the sealed test. `--split test` runs once, at the end, for the
  baseline and the final model together.
- Log every run in `results.tsv`: `commit  score  valence_rho  arousal_rho
  status  description`, with status `keep`, `discard` or `crash`.
- Keep a change only if it beats the current best by more than the noise floor
  measured from repeated baseline runs with different seeds. Otherwise revert.
- Simplicity criterion, as in the original: equal score with less code is a
  win; a tiny gain bought with lots of code is not.

## Status

Run `autoresearch/nov19` finished on 2026-11-19 after 72 experiments; see
`results.tsv`, `journal/03_autoresearch.md` and `docs/research/autoresearch/`. Its sealed
test has been used. A new run needs a new salt in `prepare.py`, hence a new sealed split.

## Loop

1. Start from the current best `train.py`.
2. Make one change with a stated hypothesis.
3. Commit, push, dispatch, wait.
4. Log the result. Keep or reset (`git reset --hard <best>`; the branch is
   force-pushed, `main` is never rewritten).
5. Repeat. When stuck, revisit near-misses, combine them, or try a different
   family (Riemannian tangent space, per-person normalisation, target
   transforms, modality ablations, nonlinear models).
