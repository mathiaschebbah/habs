# 03_autoresearch: agent-driven search for physiology-to-affect decoding

Written 2026-11-19, before any score on real data was read.

## Question

Can EEG and peripheral physiology from a 10-second video predict the viewer's
valence and arousal ratings when both the viewer and the video theme are new to
the model? And how much of what a person-split evaluation reports comes from
recognising stimuli already seen in training?

## Why this design

- Slack, 2026-11-10: another team reports AUC 0.58 (arousal) and 0.593
  (valence) for high/low ratings. DATASET.md warns that videos recur across
  people, so a person-only split lets a model score by recognising the video.
- Study 01 found that stimulus context dominates individual ratings. A
  physiology decoder evaluated on seen videos can inherit that context.
- karpathy/autoresearch: one editable file, a fixed harness, one metric, keep or
  discard. Its weakness for science is adaptive overfitting of the validation
  set over many runs; a sealed test, scored once, measures it.

## Frozen protocol

- **Data:** every audited trial (annotation audit job `20261108T205355-ade0ef`),
  all cohorts. Features from `autoresearch/extract.py`, built once on HABS.
  Eye tracking excluded (organiser announcement, 2026-11-09).
- **Splits:** SHA-256 buckets of participant ID and video theme (salt
  `habs-autoresearch-2026-11-19`). People bucket 0 × theme bucket 0 is the
  primary sealed test (new person, new theme). Test people × other themes is
  "new person, seen theme"; other people × test themes is "seen person, new
  theme". The rest is development, cross-fitted in 3 × 3 person × theme
  blocks.
- **Hidden at prediction:** ratings, video, theme, default emotion, start
  marker (encodes the intended emotion), recording path, onset.
- **Metric:** mean over valence and arousal of the within-person Spearman
  correlation, people with at least five varied trials. Pooled AUC of
  rating > 5 is reported for comparison only.
- **Loop:** each experiment changes `train.py` only; the CI harness takes
  everything else from `main`. Keep a change when the development score beats
  the current best by at least 0.005 (about half the expected between-person
  standard error); otherwise discard. Ablations are logged but never kept.
- **Sealed test:** once, at the end, the baseline and the final model together,
  with a paired bootstrap over test people (5,000 draws) for the gain and
  percentile intervals for each regime. No change after the sealed test.

## What would count as a result

- A positive primary-regime score whose 95% interval excludes 0: physiology
  carries trial-level affect information that transfers to new people and
  new videos.
- The gap between "new person, seen theme" and "new person, new theme":
  how much a person-only evaluation overstates decoding.
- The gap between the final model's development score and its sealed score:
  the cost of adaptive search.

## Outcome (2026-11-19, after the sealed test)

- Features: HABS job `20261119T185024-a98c67` (22,974 trials, 1,180 feature columns,
  ICA-cleaned EEG for 22,898 trials).
- Search: 72 experiments on branch `autoresearch/nov19`, 6 keeps (baseline, four gains,
  one identical-output rewrite), 57 discards, 8 ablations, 1 reference. Development score
  0.0305 → 0.1054 [0.089, 0.122]. Stopped after three batches without a kept change.
- Sealed test, Actions run 44598712146, run once: final 0.139 [0.083, 0.196], starting
  model 0.083 [0.031, 0.136], paired gain +0.056 [+0.012, +0.101]. New person, seen theme:
  0.152; seen person, new theme: 0.106.
- Deviation from the original loop: variants were written in parallel batches of 2 to 11,
  and a 0.005 keep threshold replaced "any improvement".
