# 05_stress_timecourse: when does the body betray a stress reaction?

Written 2026-11-20, before extracting or looking at any time-resolved data.

## Why

Study 04 showed that stress reactions to 10-second videos are detectable above
chance in new people and new videos (within-person AUC about 0.60), with
cardiac deceleration, larger skin conductance responses, shallower breathing
and lower alpha power. It only looked at whole 10-second windows. Our team's
original questions were about timing: Ash asked "at what moment" the brain and
body react, Aurélien raised classification latency, and the mentors listed
delays between signals, real-time detection and heart-rate variability on
windows shorter than 30 seconds. This study asks those questions directly.

## Questions

1. **Time course.** Second by second around video onset, how do heart rate,
   skin conductance, breathing and regional EEG band power differ between
   stress reactions and other trials, and when does each difference start
   (onset latency per signal)?
2. **Detection latency.** Using only the first t seconds of a video
   (t = 1, 2, ..., 10), how well can a fixed model detect a stress reaction in
   new people watching new videos, and from which second is it reliably above
   chance? Which signal family gets there first?
3. **Not all stress is the same.** Within stress reactions, does the cardiac
   response differ by the emotion the participant reported (fear, disgust,
   sadness, anger)? Film studies report heart-rate increases for fear and
   anger but decreases for disgust and sadness (Kreibig 2010).
4. **Habituation.** Does the stress response shrink over a session (first
   versus second half) and from day 1 to day 2?
5. **Ultra-short HRV.** How well do RMSSD and mean heart rate computed on 5,
   10, 20 and 30 seconds agree with the same 60-second window?

## Frozen protocol

- **Data:** the same audited trials as studies 03 and 04 (22,974 trials). A new
  extraction adds, per trial, 1-second bins from 3 s before to 12 s after video
  onset: instantaneous heart rate (from R peaks, interpolated at 4 Hz), skin
  conductance, breathing amplitude (Hilbert envelope) and breathing rate, and
  log band power (delta, theta, alpha, beta, gamma) averaged over six scalp
  regions. It also adds the participant's final emotion label and first
  self-reported emotion. For question 5, it computes RMSSD and heart rate on
  consecutive 60-second windows of each recording and on the first 5, 10, 20
  and 30 seconds of each window.
- **Label:** stress reaction = valence ≤ 4 and arousal ≥ 6, as in study 04.
  Controls: stress vs excitement (arousal ≥ 6; valence ≤ 4 vs ≥ 6) and stress
  vs calm negative (valence ≤ 4; arousal ≥ 6 vs ≤ 4).
- **Splits:** new salt `habs-timecourse-2026-11-20`, people × video themes in
  five buckets each, bucket 0 × 0 is the sealed test; development is the rest.
- **Baseline:** each bin minus the mean of that trial's 3 s before onset.
- **Question 1 (curves):** for each bin and signal, within-person Cohen's d
  between stressed and other trials (people with at least three of each),
  averaged over people with a 95% bootstrap interval. Onset latency = first
  bin after onset where the interval excludes zero for two consecutive bins.
  Discovery on development, replication on the sealed-test people.
- **Question 2 (detection latency):** one fixed model, chosen now and never
  tuned on scores: logistic regression (`LogisticRegressionCV`, inverse
  penalty on a 9-value grid from 1e-4 to 1, log-loss) on baseline-corrected,
  person-centred bins from onset to t, fitted on development, scored by
  within-person AUC on the sealed test (new person, new theme). Signal
  families: EEG, heart, skin, breathing, body (heart + skin + breathing), all.
  Primary outcome: the earliest t at which the "all" model's 95% lower bound
  exceeds 0.5. Development cross-fit curves are reported descriptively.
- **Question 3:** among stress trials with a final emotion of fear, disgust,
  sadness or anger, the within-person heart-rate change curve relative to the
  person's non-stress trials, per emotion. Development discovery, test
  replication.
- **Question 4:** within-person d of the heart-rate change (1–6 s), skin
  conductance rise (1–6 s) and alpha power (0–10 s) between stress and other
  trials, in the first versus second half of each session and on day 1 versus
  day 2; difference with a bootstrap interval. All trials, exploratory.
- **Question 5:** Pearson correlation and mean difference (Bland-Altman bias
  and limits) between each short window and its 60-second window, across all
  windows, plus the within-recording correlation. All recordings.

## What would count as an answer

- A clear ordering of onset latencies across signals, replicated on the test
  people, and a detection curve showing from which second a stress reaction is
  detectable in new people. If nothing is detectable before 10 s, that is the
  answer.
- Distinct cardiac curves by emotion would show that "stress" mixes
  physiologically different reactions; similar curves would support one
  shared defensive response.

## Outcome (2026-11-20)

- Extraction: HABS job `20261119T230123-4e5f45` (22,974 trials; bin-5 coverage
  heart 98%, skin 74%, breathing 96%, EEG 96%; 10,263 ECG windows from 508
  recordings).
- Splits: development 16,706 trials, 257 people, 71 themes. The primary sealed
  cell drew 13 themes and held only 9 eligible people: the frozen primary
  outcome (earliest reliable second on that cell) is uninformative. We report
  it as a split failure. Intervals for the two larger test regimes were added
  afterwards and labelled post hoc; the model stayed fixed.
- Q1 curves (development, replicated on 52 test people): slow EEG power up
  from 0–1 s (scalp-wide, not peaking frontopolar), heart rate down from 1 s,
  breathing amplitude down 2–6 s, skin conductance up from 4 s (replicated,
  same onset).
- Q2 latency: development cross-fit "all" reliable from 1 s (0.517
  [0.504, 0.531]), 0.546 by 8 s; EEG from 2 s; body from 4 s. Test, new people:
  from 3 s, 0.570 at 9 s; held-out themes: from 4 s, 0.642 at 8 s. Exploratory:
  pre-onset state alone 0.483 [0.471, 0.496].
- Q3: heart slowed for fear, disgust, sadness and anger in development; no
  emotion differed from zero on the test people.
- Q4: no habituation of heart or skin responses (session halves, days).
  Alpha effect larger on day 2 (−0.115 [−0.207, −0.023]), one of six tests.
- Q5: heart rate at 10 s r = 0.94 with 60 s; RMSSD r = 0.62 (5 s), 0.75 (10 s),
  0.86 (20 s), 0.92 (30 s).
- Literature: docs/research/stress/LITERATURE.md (42 verified references).
- Final paper for the submission: docs/paper/paper.pdf (studies 04 and 05).
