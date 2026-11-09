# Subjective–EEG carryover: analysis plan v1

Approved under the user's autonomous execution instruction, 9 November 2026.
This version follows structural feasibility only and precedes scientific fits.

## Observed structural support

Read-only [run 42594747849](https://github.com/mathiaschebbah/habs/actions/runs/42594747849),
source `8cb7ba0`, finds 6,759 valid consecutive cohort-2 pairs, 3,694 in J1 and
3,065 in J2. Fifty-four identifiers have at least twenty pairs in each session
(4,627 pairs before subsequent novelty/normalization restrictions). There are
252 current videos and 6,222 distinct video transitions. Every current video has
at least three distinct predecessors; the median gap is 14.749 s, range
4.672–87.767 s. No outcome association was fitted. The numeric evidence is
[data/temporal_feasibility.json](../../../data/temporal_feasibility.json).

## Population, measurements and eligibility

Use cohort 2 only, recorded days 1 and 2. Find predecessors on the entire video
inventory, including unusable events; require consecutive event orders, same
recording, disjoint ten-second windows, nonnegative gap at most 180 s, and valid
arousal and CURATED EEG for both videos. Exclude current videos previously seen
by that person using the complete inventory. Select people with at least twenty
remaining pairs per day and a nonzero finite J1 standard deviation in both
measurements. These are structural rules, not choices based on associations.

Let A be reported arousal and P be minus log posterior relative alpha, 8–13 Hz
relative to 1–40 Hz, as extracted by the immutable study-01 native feature code.
That code requires usable frontal and posterior channels; this conservative
whole-EEG validity rule is retained. Center and scale both A and P using each
person's usable CURATED J1 observations only. The contrast outcome is D=A−P on
these standardized scales. Neither P nor D is a clinical stress measurement.

## Individual history effects

Analyze each session separately, using identical predictors for A and P:

- fixed current-video and previous-video identity, person intercepts and
  pre-existing expected-emotion/source descriptors;
- order, squared order, onset, squared onset, centered log(1+gap), its square,
  and global interactions of both lagged measurements with centered log gap;
- separate person-specific coefficients for lagged A and lagged P.

The reference gap is the median J1 gap, fixed before associations. The parameter
delta_i,d is the person-specific lagged-A slope for A minus the corresponding
slope for P, evaluated at that reference gap. Fit the two responses jointly by
linear least squares after projection on the nuisance subspace. This is
equivalent to fitting D with the same design, and retains covariance between the
two measurements. Verify full rank of the residualized personal-history design.
Report unidentified cases rather than replacing this with unreported penalties.

Estimate the complete sampling covariance of personal delta estimates using
their linear influence matrix and a Bartlett/Newey–West kernel over actual event
orders within people, maximum lag two, with HC3 leverage adjustment. Retain
off-diagonal covariance induced by the shared nuisance fit. Missing trials are
not compressed into adjacent time indices. Clamp only numerical eigenvalues to
a positive floor and report the adjustment. A HC1/no-lag sensitivity is separate.

## Cross-session variance components

Fit a two-session normal-normal measurement-error model to all delta estimates:

```text
b = H mu + u + e
Cov(e) = blockdiag(S_J1, S_J2)
Cov(u) = [[tau1² I, covariance I], [covariance I, tau2² I]]
```

H contains a separate intercept, centered log number of pairs and the known
J1 scale ratio s_A/s_P for each day. The ratio control is necessary because
common physical-unit slopes can otherwise appear individually different after
personal standardization. Estimate components
by REML with multiple starts and bounded positive variances. The primary
reproducibility parameter is the cross-day covariance of the adjusted personal
contrasts. The latent correlation is reported only with its profile uncertainty;
it is not interpretable when between-person variance is near zero or the profile
does not identify it. Do not substitute the raw correlation of noisy slopes.

Use a parametric null bootstrap (999 draws) for the covariance likelihood-ratio
test, fitting the null with zero cross-day covariance. Use a parametric bootstrap
(999 draws) for covariance uncertainty and a likelihood profile for correlation.
These are conditional model-based intervals; the estimated sampling covariance
and normal random-effects assumptions remain limitations. A positive stable
contrast requires positive covariance evidence and a compatible predictive check.

Methodological reference for the normal-normal sampling/random-effect distinction:
[metafor multivariate documentation](https://wviechtb.github.io/metafor/reference/rma.mv.html).
The numerical implementation uses installed NumPy/SciPy, not an R dependency.

## Precision simulation before empirical effects

First construct the real predictor geometry without multiplying it by current
outcomes. Simulate sampling errors from that geometry under unit measurement
noise and a prespecified serial-correlation scenario. Simulate no, small,
moderate and large between-person contrast variation with cross-day correlation
zero or 0.6. Estimate variance components on the simulated slopes and report bias,
dispersion, boundary behavior and correlation identifiability. Synthetic outcomes
are a design diagnostic only. Freeze the measured geometry, simulation report,
code hash, input hashes and all parameters before fitting real associations.

If the personal design is rank deficient, or the simulation cannot separate
personal variation from sampling error, report that limitation and do not claim
a stable individual signature. Weak precision for small effects is not a reason
to choose a more favorable empirical representation.

## Predictive corroboration

Use five participant-grouped folds; score only their J2 pairs. Population fitting
uses other people's observations. Only the predicted person's prior J1 can
calibrate their predictions. Current responses never enter predictors; preceding
J2 responses are allowed only as explicitly already observed one-step history.

Predict standardized D with a source-bound skrub graph and a ridge population
model, alpha 10. Fit a J1 residual calibration with intercept, lagged A and lagged
P, ridge penalty 20 on slopes and an unpenalized intercept. Compare two otherwise
identical predictors: personal lagged-A correction enabled versus set to zero,
retaining the same fitted intercept and lagged-P correction. This isolates the
personal delta increment. No hyperparameter selection on J2.
As a secondary check against a weak ablation reference, also refit the restricted
J1 calibration with only its intercept and lagged EEG; this baseline is not
selected in place of the primary comparison after observing results.

Evaluate through skore, preserve its default metrics and additionally register
participant-macro MAE to avoid weighting people by their trial count. Estimate
the paired MAE gain and a 95% interval from 10,000 participant bootstrap samples.
Store the reports privately and audit cached predictions read-only. Require the
real-data cold-grid smoke test, not only cross-validation.

Skore 0.29.1 cross-validation deserialization requires the original private source
bindings as well as the saved workspace datasets. Restore features, recorded
events and J2 query grid at their original absolute paths for the audit. A unit
test forbids estimator fitting during this restoration and verifies cached
predictions/metrics. None of these private source files is a public artifact.

## Prespecified sensitivities and interpretation

Sensitivity analyses: common population J1 scales instead of personal scales;
at least thirty pairs per day; sampling covariance scaled
by 0.5 and 2; zero versus two HAC lags; no pair-count moderator; exclude unstable
J1 scales A<0.25 or P<0.05; restrict to the J1 gap interquartile range. These do
not replace the primary result. Peripheral or apériodic representations are
optional future extensions if the primary study is already answered.

The unit is an identifier within cohort 2. The video catalogue is fixed; no
population-new-video or cross-cohort biological replication claim is made.
The previously reserved sample is open, so this is exploratory internal
longitudinal evidence. Persistence of rating habits, lagged-variable error,
normalization noise and sensor dynamics are not fully identifiable from true
emotional persistence. No result establishes a qualia mechanism or stress onset.
