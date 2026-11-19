# 04_stress_detection: can we tell from physiology when someone is stressed?

Written 2026-11-19, before we looked at any stress score.

## Question

When a participant watches a short emotional video, can we tell from their EEG,
heart, skin and breathing whether they are having a stress reaction? And which
physiological signals change when they are?

## What "stressed" means here

The release has no stress questionnaire, no cortisol and no stress task: the
participant file only has age and gender. So we use the participant's own
ratings of each video. A trial counts as a **stress reaction** when they rated
it negative (valence ≤ 4) and activating (arousal ≥ 6), the "distress" corner of
the valence-arousal plane. Every other trial counts as not stressed. This is a
self-reported acute reaction to a video, not chronic stress or a clinical state,
and the participants are healthy volunteers, not patients.

The obvious trap is that a detector could just pick up arousal: excitement
also speeds the heart and raises skin conductance. So we fixed two control
contrasts in advance:

- **Stress vs excitement:** among activating trials (arousal ≥ 6), negative
  (valence ≤ 4) versus positive (valence ≥ 6). Same arousal, opposite valence.
- **Stress vs calm negative:** among negative trials (valence ≤ 4), activating
  (arousal ≥ 6) versus calm (arousal ≤ 4).

A stricter definition (valence ≤ 3 and arousal ≥ 7) is reported as a
sensitivity check.

## Protocol (frozen)

- **Data:** the same trial table as study 03 (all audited trials, all cohorts,
  1,180 label-free features per trial, no eye tracking).
- **Splits:** new salt `habs-stress-2026-11-19`, so this study gets its own
  sealed test. People bucket 0 × theme bucket 0 is the primary test (new person,
  new video theme); the two partial overlaps are secondary regimes. The rest is
  development, scored by the 3 × 3 person × theme cross-fit. Ratings and
  stimulus columns, including the start marker, are hidden from predicted rows.
- **Metric:** ROC AUC within each person (at least three stressed and three
  other trials), averaged over people. Chance is 0.5. We also report pooled AUC,
  average precision, the two control contrasts and the strict definition.
- **Search:** we only change `decoding/models/stress.py`; CI scores it with the
  code from `main`. A change is kept if it beats the best development score by
  at least 0.005; ablations are logged, never kept. Every variant is a commit on
  `experiments/stress` and a line in `docs/research/stress/experiments.tsv`.
- **Correlates:** on development data only, the within-person standardised
  difference of each interpretable feature (regional band power, relative
  power, heart, skin, breathing) between stressed and other trials, and for the
  two control contrasts, with 95% bootstrap intervals over people.
- **Sealed test:** once, at the end, for the starting model and the final model
  together (plus EEG-only and peripheral-only ablations of the final model),
  5,000 bootstrap draws over test people.

## What would count as an answer

- Primary score above 0.5 with an interval excluding 0.5: physiology detects
  stress reactions in new people watching new videos.
- Stress vs excitement above 0.5: the detector carries something beyond arousal.
  If it sits at 0.5 while stress vs calm negative is high, we are detecting
  arousal, not stress.
