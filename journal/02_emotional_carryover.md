# 02_emotional_carryover

## Question / hypothesis

Is the stimulus- and timing-adjusted difference between previous-arousal effects
on current reported arousal and current posterior EEG alpha reproducible within
people across recorded sessions J1 and J2?

## Motivation

- **Sourcing strategy:** backlog:B9, promoted explicitly by the user.
- **Sources:** [literature review](../docs/research/literature/2026-11-09/REVIEW.md)
  and [question v0.1](../docs/research/literature/2026-11-09/QUESTION.md).
- **Why this matters:** sequential effects and response-system discordance are
  known, but their individually reproducible conditional contrast was not found
  in the reviewed corpus. The first study did not test this temporal question.

## Method

The user authorizes feasibility, implementation, execution, auditing and the paper
without additional questions on 2026-11-09. Proceed in two recorded phases.

1. Reconstruct consecutive videos from the complete recorded-event inventory,
   before rating/EEG exclusions. Use the existing private audited feature cache,
   actual recorded days, strictly CURATED EEG, disjoint ten-second epochs and
   finite positive posterior relative alpha. Count eligible transitions, paired
   people, actual gaps, shared videos, previous-video diversity and support of
   the nuisance design. Do not fit target associations in this phase. Read-only
   transport may run on a GitHub CPU worker; publish aggregate evidence only.
2. After these structural findings, freeze a versioned statistical analysis and
   simulation-based precision assessment before fitting the scientific effects.
   Use a participant/session hierarchical or measurement-error variance-component
   analysis that carries uncertainty of individual slopes. Include current and
   previous video, order/time, inter-video gap, prior arousal and prior EEG;
   normalization uses J1/development only. The primary endpoint is reproducibility
   of the standardized subjective–EEG history-effect contrast. J1-to-J2 predictive
   validation must compare personal history effects with the same population
   history model, on identical rows and respecting participant grouping.

- **Files:** reusable code under `habs/temporal/`, one paired experiment
  `experiments/02_emotional_carryover.py`, matching audit and real-data smoke test.
  Existing study-01 model files and frozen settings remain unchanged.
- **Pipeline:** source-bound skrub DataOps; mark the prediction grid before joins
  to upstream historical sources. Any forecast uses only an already observed
  predecessor, never the current response. Evaluate learners with skore and store
  local reports in the private HABS output tree.
- **Cross-validation:** decide from measured transition support after feasibility;
  J2-only participant-grouped validation with explicit J1 history is the candidate.
- **Outputs:** measured feasibility, frozen protocol/inputs, simulation precision,
  empirical estimates and uncertainty, diagnostic figures, read-only audit, short
  PDF paper and supporting evidence. Unsupported personal-stability claims are
  reported as unidentifiable rather than forced into a positive finding.

### Structural findings and method resolution, 2026-11-09

[Read-only feasibility run 42594747849](https://github.com/mathiaschebbah/habs/actions/runs/42594747849)
finds 6,759 valid consecutive transitions. Fifty-four people have twenty pairs
per day, before novelty/scale restrictions; 252 videos have diverse predecessors.
The [resolved analysis plan](../docs/research/carryover/PROTOCOL.md) defines the
full sampling covariance, REML variance components, pre-effect precision
simulation, five participant-grouped J2 folds and fixed personal-slope ablation.
This resolution uses structural counts only; empirical associations remain unfitted.

## Risks / things that could invalidate the result

- The old reserved outcomes are already open. This is an exploratory new study,
  not an independent confirmation, even when decisions precede its own results.
- Missing or invalid intermediate events can create false adjacency. Never bridge
  them after filtering, join across days, or treat incomplete order labels as clocks.
- Deterministic video sequences, weak within-person history variation or too few
  CURATED pairs can make stimulus-adjusted individual slopes unidentifiable.
- Regression-to-the-mean, rating habits, slow sensor dynamics and noisy lagged
  variables can mimic or attenuate persistence. Carry uncertainty and avoid claims
  about intrinsic emotional durations, clinical stress, interoceptive deficits or qualia.
- Cross-cohort identities are unresolved. No pooled cross-cohort replication claim.
- A simulation with synthetic responses assesses the design/estimator only; it
  is never empirical evidence for the scientific hypothesis.

## Status

- **State:** done
- **Approved by user on:** 2026-11-09 — “Vas-y, vérifie la faisabilité sur nos données,
  puis mène l’étude du décalage ressenti–EEG entre J1 et J2 jusqu’aux résultats et au
  papier, en autonomie sans plus rien me demander :)”.
- **Headline result:** 53 paired IDs / 4,554 transitions. Adjusted cross-day delta
  covariance +0.003496, model-based 95% CI [−0.010277, +0.015055], null-bootstrap
  p=0.253; latent correlation is weakly identified. Personal-slope J2 MAE gain
  −0.005539 [−0.012251, +0.000938]. No established repeatable personal contrast or
  predictive increment. [Results](../docs/research/carryover/RESULTS.md), five-page
  paper and three-page supplement delivered after saved-report audit.
- **Implication for next iteration:** small personal variation remains hard to
  separate from noisy slope estimation. Independent repeated measurements and
  better precision are needed for a personal-stability claim; do not retune this
  completed study or treat its opened outcomes as new confirmation.
