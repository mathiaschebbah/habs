# HABS data understanding

Measured on 2026-11-08 by [HABS run 42522381425](https://github.com/mathiaschebbah/habs/actions/runs/42522381425), code `1b53f01`.
The portable [aggregate evidence](eda_summary.json) is a direct copy of that run's
artifact. Individual trial tables and interactive TableReports remain in team
storage under `20261108T205355-ade0ef/data/`; they are not redistributed here.

## Dataset at a glance

The label file contains 29,514 rows. Strict reconciliation retains 22,974 rated
trials from 314 participant identifiers, across 598 available video recordings.
All retained trials have valence and arousal in [1,9], fit a ten-second recording
window, and agree with a same-code BDF annotation within 25 ms. No header errors
were encountered. These are released identifiers, not verified unique biological
people across cohorts.

| Cohort | Matched trials | Identifiers |
|---|---:|---:|
| 1 | 5,832 | 105 |
| 2 | 9,726 | 113 |
| 3 | 2,494 | 30 |
| 4 | 4,615 | 52 |
| 5 | 307 | 14 |

The accounting closes exactly: 22,974 retained + 4,213 with no matching recorded
event + 2,293 missing/out-of-range ratings + 34 outside/short windows = 29,514.
The missing-rating category includes masked official test observations; it does
not indicate that all those rows are data errors. There are 2,690 label-day
corrections. The current count is two above the old arousal-only count quoted in
the supplied project, so that historical number is not substituted for this audit.

## Per-column findings

After removing all responses from the 26 reserved cohort-2 participants, the
development overview contains 20,332 rows and 46 annotation/recording columns,
covering 288 identifiers and 252 videos. Dominance is missing in 63.5% of these
rows, the first emotion report in 28.7%, and the third in 83.1%. Expected emotion
and its source are complete. Reported presentation order is missing in 28.7%,
so recorded event order should be used instead.

The outlier flag has only one observed value in development. This field cannot
support a meaningful within-release comparison between flagged and unflagged
observations. The delivery may already be conditioned on upstream screening.

All 598 video headers identify ECG in mV, GSR in uV and respiration in arbitrary
units. This differs from the supplied local sample's erroneous GSR uS label.
Native rates have three distinct values across recordings; extraction must read
each file's rate. The selected seven frontal/posterior electrodes are present
throughout the development overview. Signal amplitudes, clipping and feature
availability still require the next quality step; headers alone do not prove
signal validity.

## Target

The chosen target is `S=max(0,(5-V)/4)*(A-1)/8`, an unvalidated experimental proxy
of aversive activation, not a direct stress measurement. In cohort-2 development,
3,884 trials are from recorded J1 and 3,200 from J2. The first histogram interval,
`0 <= S < 0.05`, contains 2,199 J1 trials and 1,926 J2 trials, respectively 56.6%
and 60.2%. These bins include both exact zeros and small positive scores; they
must not be reported as exact zero counts. This concentration argues against
assuming that ordinary lower/upper quartiles define balanced distinct classes.

## Structure

Seventy-six cohort-2 identifiers have at least twenty rated, aligned trials in
each recorded session. A deterministic hash seeded with 20261108 reserves 26
people for the final test and leaves 50 paired people for development. Another
37 cohort-2 identifiers can support population fitting if their signals pass
quality checks. All target-response distributions for the reserved people are
excluded from this exploration.

The allocation fingerprint is
`6717f50a5c86658a35f175417c463fca9f2c46ba5b98c3c1bdb11edab52f9027`.
The labels fingerprint is
`c981a6836dfcd6af2b4c68ca3fab0283453f934522217b7b5c66e03119430fcd`.

Nine participant–video combinations repeat across sessions in the aligned data.
New-to-person evaluation must exclude all earlier exposures using the complete
recorded-event inventory, including events without a usable rating.

## Associations

In cohort-2 development, 574/583 first-ranked anger reports and 494/547 fear
reports have a positive proxy, compared with 2/817 joy reports and 12/690
relaxation reports. This is descriptive convergence with emotional reports, not
independent validation of stress. Sadness also frequently produces a positive
score, demonstrating that the proxy is not specific to fear or threat.

At arousal >=6, there are 1,666 negative-valence trials with V<=4 and 1,341
positive-valence trials with V>=6, plus 198 intermediate-valence observations.
Thus both sides of a high-arousal specificity contrast exist at the aggregate
level. Support within individuals, stimuli and sensor-quality strata remains
to be checked before interpretation.

Current ratings, self-reports and the refined final emotion label would leak the
target if used as prospective predictors. Current arousal can only enter the
separately labelled conditional explanation analysis.

## Modelling implications

- Use the continuous score and participant-macro MAE as selected in the project.
- Validate on entire unseen development participants' J2, while allowing only
  their J1 for the specified personal calibration. Random trial splits do not
  match this question.
- Include stimulus identity and prior context, then compare physiology on the
  identical future trial grid. Add a missingness-only control and a no-EDA
  sensitivity to separate values from sensor availability and calibration issues.
- Learn imputation, scaling and model choices on development data only.
- Preserve cohort separation; no cross-cohort identity linkage is available.

## Signal-quality follow-up

The [native-signal preparation](https://github.com/mathiaschebbah/habs/actions/runs/42526673694)
completed successfully and passed the real-data disjoint-grid smoke test. Its
[aggregate output](feature_summary.json) covers 20,173 trials in cohorts 1, 2
and 4. Preparation took 140.5 seconds after reusing the audited annotations.

| Cohort | EEG usable | ECG usable | EDA usable | All three usable |
|---|---:|---:|---:|---:|
| 1 | 4,134 / 5,832 | 3,791 | 4,644 | 2,094 |
| 2 | 7,795 / 9,726 | 6,458 | 6,984 | 3,672 |
| 4 | 3,277 / 4,615 | 3,215 | 4,606 | 2,364 |

EDA is outside the supplied calibration domain in 1,188, 2,742 and 9 trials,
respectively. Those values are unavailable rather than clamped to a plausible
endpoint. Availability differences require the planned missingness-only control,
no-EDA sensitivity and complete-observation subgroup. This quality table precedes
the final restriction to CURATED, prior-calibration support and novel J2 videos;
it is not the final test population or evidence of predictive performance.

## Consecutive-trial feasibility, study 02

The read-only [transition audit](https://github.com/mathiaschebbah/habs/actions/runs/42594747849)
reconstructs predecessors on the complete video inventory before filtering.
It finds 6,759 CURATED-EEG/rating pairs in cohort 2. Fifty-four IDs have twenty
pairs in both sessions before novelty/normalization restrictions. After requiring
novel current videos and paired support, 53 IDs remain with 4,554 transitions:
2,294 J1 and 2,260 J2. See [structural evidence](temporal_structure.json).

The [pre-effect precision run](https://github.com/mathiaschebbah/habs/actions/runs/42598671314)
finds full rank of all 106 personal-history columns in each day, conditional on
video, person and timing nuisance terms. Condition numbers are 5.17 and 19.13.
The J1 reference gap is 15.957 s. [Simulated precision](temporal_precision.json)
shows broad correlation estimates for small random-effect variation. A
shared-scale-only simulation gives a median raw slope correlation about 0.56
despite zero generating personal covariance; the scale-adjusted covariance is
centered near zero. This is a synthetic estimator diagnostic, not an empirical
association or a power guarantee. No real carryover effect was fitted in these runs.

## Remaining limitations

The release does not establish a direct stress criterion, biological identity
across cohort IDs, or independent validation of GSR calibration. The acquisition
and upstream screening history limit interpretation. The next step is to measure
signal validity and predictive increments, rather than infer them from coverage.
