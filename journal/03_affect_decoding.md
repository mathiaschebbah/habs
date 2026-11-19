# 03_affect_decoding: rating prediction for new people watching new videos

Written 2026-11-19, before we read any score on real data. Outcome added after
the sealed test.

## Question

Can EEG and peripheral physiology from a 10-second video predict the viewer's
valence and arousal ratings when both the viewer and the video theme are new to
the model? And how much of what a person-only split reports comes from
recognising videos already seen in training?

## Why this design

- Slack, 2026-11-10: another team reports AUC 0.58 (arousal) and 0.593
  (valence) for high/low ratings. DATASET.md warns that videos recur across
  people, so a person-only split lets a model score by recognising the video.
- Study 01 found that stimulus context dominates individual ratings, so a
  decoder evaluated on seen videos can inherit that context.
- We want to iterate on the model many times. Reusing one development set for
  many comparisons can overfit it, so we keep a sealed test and open it once.

## Frozen protocol

- **Data:** every audited trial (annotation audit job `20261108T205355-ade0ef`),
  all cohorts. Features from `decoding/extract.py`, built once on HABS. Eye
  tracking excluded (organiser announcement, 2026-11-09).
- **Splits:** SHA-256 buckets of participant ID and video theme (salt in
  `decoding/affect.py`). People bucket 0 × theme bucket 0 is the primary
  sealed test (new person, new theme). Test people × other themes is "new
  person, seen theme"; other people × test themes is "seen person, new theme".
  The rest is development, cross-fitted in 3 × 3 person × theme blocks.
- **Hidden at prediction:** ratings, video, theme, default emotion, start
  marker (it encodes the intended emotion), recording path, onset.
- **Metric:** mean over valence and arousal of the within-person Spearman
  correlation, people with at least five varied trials. Pooled AUC of
  rating > 5 is reported for comparison only.
- **Search:** each experiment changes the model file only; CI takes everything
  else from `main`. Keep a change when the development score beats the best by
  at least 0.005 (about half the expected between-person standard error);
  otherwise discard. Ablations are logged but never kept.
- **Sealed test:** once, at the end, the starting and final models together,
  with a paired bootstrap over test people (5,000 draws).

## Outcome (2026-11-19)

- Features: HABS job `20261119T185024-a98c67` (22,974 trials, 1,180 feature
  columns, ICA-cleaned EEG for 22,898 trials).
- Search: 72 experiments on branch `experiments/affect`, 6 keeps (start, four
  gains, one rewrite with identical output), 57 discards, 8 ablations,
  1 reference. Development score 0.0305 → 0.1054 [0.089, 0.122]. We stopped
  after three batches without a kept change.
- Sealed test (run once): final 0.139 [0.083, 0.196], starting model 0.083
  [0.031, 0.136], paired gain +0.056 [+0.012, +0.101]. New person, seen theme:
  0.152; seen person, new theme: 0.106.
- Deviation: we wrote variants in parallel batches of 2 to 11.
