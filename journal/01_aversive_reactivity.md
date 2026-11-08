# 01 aversive reactivity

## Question / hypothesis

Does EEG and peripheral physiology reduce participant-macro MAE for aversive
emotional activation on a later session beyond stimulus context and J1 personal
calibration, and are any physiological response profiles stable and useful?

## Motivation

- **Sourcing strategy:** user-selected [research project](../docs/research/PROJECT.md).
- **Sources:** [measured data understanding](../data/eda.md); [verified references](../docs/research/REFERENCES.md).
- **Why this matters:** the audit supports 76 paired cohort-2 identifiers and
  both valence directions at high arousal. Stimulus-aware controls are needed to
  distinguish incremental physiology from recognizing the videos.

## Method

The target is `S=max(0,(5-V)/4)*(A-1)/8`. This is an experimental aversive-activation
proxy. Predictions use current physiology but never current ratings, reported
emotions, final labels or screening flags. The full ten-second video is the
primary window; any five-second analysis predicts the subsequent report, not a
timestamped stress onset.

### Population and allocation

Use the aligned cohort-2 data with the already recorded seed, label fingerprint
and allocation fingerprint in `data/eda.md`. The 26 reserved identifiers do not
enter population fitting or model selection. Their J1 can be used only for the
specified personal calibration. The 50 paired development identifiers support
validation; remaining cohort-2 development identifiers can support population
fitting. Exclude future trials whose video was previously presented to the same
person, using all recorded events rather than just labelled events. Do not
reassign participants after measuring sensor quality.

Keep cohorts separate because biological overlap between different IDs is
unknown. Cohorts 1 and 4 are separate exploratory replications when their usable
paired sample is sufficient. They do not validate cross-cohort transfer.

### Representations and quality

Read native sample rates and units. Use selected F3/F4/Fz and P3/P4/O1/O2 EEG
channels, rather than assuming Cz is an available acquisition channel. Compute
regional relative theta 4–8 Hz, alpha 8–13 Hz and beta 13–30 Hz power with Welch
two-second windows and half overlap, relative to 1–40 Hz. Require at least two
usable channels in each region. Mark a channel unusable if non-finite, flat,
or showing >500 uV peak-to-peak or >150 uV standard deviation in the trial.
These are transparent conservative artifact screens, not validated stress rules.
Current-trial prediction uses CURATED recordings to avoid post-window information
introduced by released acausal EEG preprocessing. An already observed J1
calibration session may use the audited PREPROCESSED fallback, since the whole
past session is available before J2. Source-specific counts are reported.

ECG yields median heart rate and its within-trial change from a causal bandpass
and peak detector, with plausible RR intervals of 0.35–1.5 s and sufficient beats.
No frequency-domain HRV is claimed from ten seconds. EDA follows HABS's supplied
voltage-to-resistance calibration; values outside its domain are missing rather
than silently clamped. A window with >1% invalid values, a flat trace or obvious
plateau saturation is unavailable. Record unit conflicts. The actual video
release uses uV, unlike the local sample's erroneous uS header. Report EDA
availability and a no-EDA sensitivity even when conversion is numerically valid.

Signal extraction is target-independent and must preserve row identities.
Peripheral missingness is retained and imputed from population training data;
all main models share the same EEG-usable future trial grid and J1 history.
Quality exclusions and counts are reported before any model conclusion.
Physiological models receive raw values and differences from their own J1
medians; both use population-training scales. This normalization never estimates
parameters from the person's future session. A history-only physiological
control distinguishes stable J1 summaries from information in the current trial.

### Models and selection

M0 is a regularized stimulus/context model using video identity, pre-response
expected emotion/source and recorded presentation position. M1 adds a J1
personal residual-offset calibration. M2 adds EEG, M3 adds ECG and EDA, and M4
adds all these modalities with the same calibration history. Include constant
zero/mean/median reference predictions and a context-plus-missingness control.

Use ridge regression initially, with population-only encoding, median imputation
and scaling. Candidate regularization strengths are 0.1, 1, 10, 100 and 1000.
Personal offsets use the J1 median residual, shrunk by n/(n+lambda), with lambda
in {0,10,40}; choose lambda for M1 on development and hold it across the modality
comparisons. Prediction bounds are clipped to [0,1] consistently. Train-person
weights give each person equal total weight.

Before any real development performance is inspected, the context representation
also includes the video-by-expected-emotion interaction, and calibration selection
includes its disabled limit. This prevents making the primary reference weaker
than its uncalibrated counterpart by construction. The selected calibration
strategy is shared across modality comparisons.
Session number and linear/quadratic recorded order are shared context covariates.
The nonlinear physiological variant adds splines only to continuous physiological
values and J1-centered differences, retaining the same context and quality controls.

Cross-validation geometry is selected at the evaluation implementation step,
from the measured participant/session structure. Every validation fold must
exclude its people entirely from population fitting and score only their novel
J2 trials; only those people's J1 labels may enter calibration. The source-bound
skrub graph marks X before the history-dependent calibration and references J1
as a separate upstream source. Use skore for evaluation reports and persistence.

Freeze all selected settings, feature definitions, inputs, allocation, validation
scores and code version before loading the reserved J2 responses for evaluation.
The final test cannot choose a model, threshold, feature variant or cluster count.

### Primary contrast and uncertainty

Report the mean of each person's MAE, with `delta=MAE(M1)-MAE(M4)` as primary
contrast. Bootstrap paired participant errors with seed 20261108 and 10,000
resamples for a percentile 95% interval. Report both the effect and uncertainty,
including the proportion of people helped. A gain of 0.01 on the 0–1 score is
the prespecified minimum useful engineering increment, with 0.005 and 0.02
sensitivities. These thresholds are not clinically validated.

This interval describes participant uncertainty for the observed stimulus
catalogue. It does not support claims about entirely new population-level videos.
Report within-person rank associations only where the response varies.

### Secondary and conditional analyses

1. Refit the same families for valence and arousal separately and an alternative
   monotone composite. These are prespecified sensitivities, not alternate
   primary endpoints chosen after results.
2. For specificity, model negative valence with current reported arousal included
   as an explanatory control, together with stimulus and calibration context.
   Clearly separate this conditional analysis from prospective prediction. Check
   per-person positive/negative support at comparable arousal.
3. Compare linear physiology with spline features selected on development. Plot
   controlled continuous relationships with participant uncertainty. If distinct
   extreme groups and sufficient intermediate observations exist in development,
   use an extreme-trained score on those withheld intermediate observations; do
   not interpret that score as a calibrated quantity of stress.
4. Fit physiological clusters using development J1 without response labels.
   Compare continuous/PCA descriptions to K-means with 2–5 clusters. Choose any
   discrete representation only on development, assess sensor/missingness
   confounding, and compare J1/J2 within-person similarity to between-person
   controls. Test M5's increment over continuous M4 as a secondary endpoint.
   No cluster receives a fight/flight/freeze or clinical label.
5. Shift joint physiological vectors within participant/session to break trial
   correspondence, retaining person/session distributions. This is a diagnostic
   control, not an exact exchangeability p-value. Prioritize the missingness-only
   control and the EEG+ECG versus all-modalities comparison.
6. A five-second feature window and respiration/cross-modal forecasting are
   extensions contingent on usable timing and signal support. No physiological
   coupling is interpreted as causal.

The five-second comparison uses five-second EEG quality eligibility, never the
future ten-second quality mask. Its reference and physiological model are selected
on development and compared on the identical early-window trial grid. Consequently
its population may differ from the full-window analysis and its absolute errors
are not a paired five-versus-ten-second contrast. Conditional matching uses
arousal >=6, valence <=4 versus >=6, and a fixed 0.5-point arousal caliper within
participant without trial reuse. Dose plots use the prespecified frontal alpha,
heart-rate and log-conductance markers, development video means/scales, J1
centering, and participant-level intervals. The extreme-score threshold is the
development 75th percentile among positive scores; only the supervised extreme
rows fit that diagnostic score, with no personal response offset.

Deliver a compact main paper and supporting evidence. Retain measured exclusions,
uncertainty, positive and negative findings, and the selected project's figure
categories where an analysis is feasible. Explicitly report unsupported or
unidentifiable hypotheses rather than replacing them with simulations.

## Risks / things that could invalidate the result

- Shared stimuli and personalized selection can drive apparent performance.
- The proxy is not independently validated stress; post-video reports cannot
  date its onset. Many scores lie near zero, favoring simple MAE baselines.
- Sensor quality and availability can encode participants or protocol conditions.
- The released labels may already exclude screened trials; this conditions the
  population of inference. Cross-cohort biological identity is unavailable.
- Only 26 reserved people support primary uncertainty. Clusters or small gains
  can be unstable, even when large trial counts appear convincing.
- Full-video features support an offline task. Extra signal windows, repeat
  exclusion and processing boundaries must be audited before early-prediction claims.

## Status

- **State:** done
- **Approved by user on:** 2026-11-08, research direction supplied and autonomous
  completion authorized; detailed implementation choices recorded by the agent.
- **Headline result:** on 22 reserved identifiers and 997 novel J2 trials,
  participant-macro MAE is 0.112443 for M1 and 0.112826 for M4; paired gain
  −0.000383, 95% CI [−0.002877, +0.001822]. The prespecified 0.01 gain is excluded
  for this representation/model family. Saved metrics, source/freeze hashes and
  paired uncertainty verified by the [executed audit](../docs/research/results/audit.md).
- **Implication for next iteration:** no demonstrated physiological increment or
  predictive utility of the two-cluster history. Profile similarity is descriptive
  and not robust to every count-stratified identity control. Further confirmatory
  research needs independent stress criteria or a new reserved sample, rather
  than selecting variants retrospectively on these 22 identifiers.
