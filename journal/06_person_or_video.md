# 06_person_or_video: does physiology read the person or the video?

Written 2026-11-20, after the feasibility audit (counts only) and before
computing any outcome of this study. The heartbeat-evoked extraction (part B)
was running when this was written; nothing from it had been read.

## Why

Studies 04 and 05 found that stress reactions slow the heart, raise skin
conductance, make breathing shallower and shift EEG power, and that a model
detects them weakly in new people. But the same videos are watched by many
people. A video that frightens most viewers also changes everyone's
physiology, and a gory clip can change heart rate or EEG through its images and
sound, whatever the viewer feels. Gerster et al. (bioRxiv 2026) showed on the
FACED dataset that EEG emotion classifiers mostly recognise the stimulus. So
our earlier results may say "this video is stressful", not "this person is
stressed". For anyone who wants to monitor a person, that difference is the
whole question.

The HABS design lets us separate the two: each of the 252 videos was watched by
13 to 234 people (median 83), from every cohort. For each trial we can compare
what this person reported and showed with what the other viewers of the same
video reported and showed.

Part B comes from the course on brain networks of consciousness (B. Hermann,
2024) that a teammate shared. It presents the cortical response to each
heartbeat (heartbeat-evoked response, Park et al. 2014; Candia-Rivera et al.
2021) as a marker of brain–body interaction. Since the heart slows during
stress reactions (study 05), we ask whether the brain's response to each beat
changes too.

## Feasibility audit (counts only, run 44652511915)

22,974 trials, 314 people, 252 videos; every video has at least 9 other viewers;
every person has at least 10 trials with a video mean; 24% of trials are
stress reactions; 140 videos have at least 3 stressed and 3 non-stressed
viewers (265,019 stressed/non-stressed pairs); all 252 videos appear in
several cohorts. Response coverage: heart 97%, skin 73%, breathing 96%, EEG
96%. Dominance is rated on 32% of trials.

## Frozen protocol, part A

- **Data:** the time-resolved table of study 05 (job 20261119T230123-4e5f45)
  joined to the trial table; all 22,974 trials, no split (nothing is tuned).
- **Responses** (mean baseline-corrected change over a window chosen from the
  study-05 development curves): heart rate 1–8 s; skin conductance 4–10 s;
  breathing amplitude 2–6 s; delta power averaged over six regions 0–2 s;
  alpha power averaged over six regions 0–10 s.
- **Shared and own parts.** For a value (rating or response) of person p on
  video v: centre it on the person's mean; its *shared part* is the mean
  centred value of the other viewers of v (at least 9 of them); its *own part*
  is the centred value minus the shared part, centred again per person.
- **Coupling.** For each person with at least 10 complete trials, regress the
  response (z-scored within person) on the shared and own parts of the rating
  (each z-scored within person). Average the two coefficients over people,
  95% interval from 2,000 person bootstraps. Done for valence and arousal and
  every response. Also reported: the within-person correlation of each
  response with the other viewers' mean response to the same video (stimulus
  locking).
- **Who in the room.** Scores: the fixed study-05 model (all signals, first
  10 s, logistic regression), cross-fitted so that every trial is predicted by
  a model blind to its person fold and theme fold; each single response,
  signed so that higher means more stress-like; and, as a benchmark, the other
  viewers' stress rate for the video. Metrics: within-person AUC, and the AUC
  over all pairs of a stressed and a non-stressed viewer of the *same* video
  (videos with at least 3 of each), 95% interval from 2,000 video bootstraps.
  The stimulus is identical inside a pair, so this AUC can only come from the
  person.
- **Primary hypotheses** (two-sided; an effect counts if its 95% interval
  excludes zero, or 0.5 for AUCs):
  - A1: the heart follows the person's own valence (own-part coefficient of
    valence on the heart-rate change differs from 0; expected positive).
  - A2: the skin follows the person's own arousal (own-part coefficient of
    arousal on the skin response differs from 0; expected positive).
  - A3: the fixed model tells stressed from non-stressed viewers of the same
    video (same-video AUC differs from 0.5).
- **Robustness:** A1–A3 recomputed within cohort 1, cohort 2 and cohorts 3–5.
- **Exploratory:** heart-rate change of stress trials rated low (≤ 4) versus
  high (≥ 6) in perceived control (dominance), against the person's
  non-stress trials (freezing-like bradycardia should be stronger with low
  control).

## Frozen protocol, part B

- **Extraction** (`decoding/hep_extract.py`): R peaks from the ECG (refined to
  the largest QRS deflection within ±60 ms); EEG from the ICA-cleaned release,
  0.5–30 Hz; epochs −200 to +600 ms around each R peak, baseline −200 to
  −50 ms, epochs above 100 µV rejected; averaged per trial over the beats of
  the 3 s before onset and of the 10 s video, per scalp region, in 50 ms bins.
  The ECG is averaged the same way.
- **Primary measure:** mean heartbeat-evoked amplitude 250–600 ms after R,
  averaged over the frontal and central regions, during the video.
- **Primary hypotheses** (two-sided):
  - B1: within-person Cohen's d of this amplitude between stress reactions and
    other trials differs from 0 (people with at least 3 of each, 2,000 person
    bootstraps).
  - B2: the within-person stress coefficient on the amplitude stays different
    from 0 when the ECG amplitude in the same window and the mean RR interval
    are added as covariates (the heart's own field and its rate).
- **Descriptive:** amplitude change from pre-onset to video; the two control
  contrasts (versus excitement, versus calm negative); per-region waveforms of
  d; the ECG waveform of d; shared and own coupling of the amplitude with
  valence and arousal.

## What would count as an answer

- If the own-part coefficients are near zero while the shared ones are not, our
  physiological stress signal mostly reads the video. If they differ from zero,
  physiology carries the person's own reaction beyond what the video does to
  everyone. A same-video AUC above 0.5 is the direct evidence that the body
  tells who in the room is stressed.
- For part B, a stress effect on the heartbeat-evoked response that survives
  the cardiac covariates would show that the brain processes heartbeats
  differently during stress reactions. An effect that vanishes with them would
  be the heart's field, not the brain.
- Four primary tests (A1–A3, B1) at 95%; we report every interval and say so.
