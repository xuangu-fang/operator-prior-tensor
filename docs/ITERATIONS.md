# Iteration log

## Migration baseline — 2026-08-15

- Split from the Physics-Informed Tensor Learning Hub.
- Preserved the 2%/5%/10% fixed-budget protocol and calibrated operator-mismatch phase diagram.
- Next research target: replace synthetic subspace rotation with PDE/operator parameter perturbations while preserving a measurable projection-residual axis.

## Physical iteration 1 — variable-coefficient diffusion operator

**Falsifiable hypothesis.** The phase-boundary signal should survive when
mismatch comes from a PDE coefficient rather than an artificial principal-angle
rotation.  In particular, a reference-operator Tucker model should outperform
rank-matched Neural Functional Tucker when the measured projection residual is
small, with the advantage weakening as the diffusivity operator changes.

**Implementation.** Added a conservative finite-volume discretization of

\[
\partial_tu+[-\partial_x(a(x)\partial_x)+\kappa I]u=0
\]

with zero-flux boundaries.  The tensor is the Green response
`time × receiver × source`.  Truth uses the variable-coefficient eigenpairs;
the learner uses a truncated constant-diffusivity spectrum.  The code records
the exact relative residual after projection onto the learner's three-mode
product space.  Protocol: contrast `0, .5, 1, 1.5, 2`, cutoff 8, 2%/5%/10%,
seeds 41/42/43, 400 steps, 10% observed noise, random cold start.

**Result.** Contrast increases measured residual from `0.0459` to `0.0965`, but
does not make the task monotonically harder because the physical Green field
also changes.  At contrast 0, Operator Tucker beats Neural F-Tucker by
`0.093/0.072/0.069` NRMSE at 2%/5%/10% and wins 3/3 seeds.  Across nonzero
contrasts, 10% remains mostly positive (`2/3` or `3/3` wins); 2% and 5% are
near-ties or high variance.  At contrast 1, means are
`0.273/0.206/0.158` for Operator Tucker versus
`0.262/0.210/0.189` for Neural F-Tucker.

**Decision.** Partial GO.  A physical-operator signal exists, but it is stable
only at 10%; do not reuse the stronger synthetic claim for the physical data.
Artifact: `results/diffusion_contrast_summary_r1/summary.json`.

## Physical iteration 2 — basis cutoff as bias–variance control

**Falsifiable hypothesis.** If approximation bias is the dominant error,
increasing the learner cutoff from 5 to 12 should monotonically improve held-out
NRMSE as its oracle residual decreases.

**Protocol.** Fix contrast 1 and all training settings; sweep cutoff 5/8/12.
Measured residuals are `0.1645/0.0699/0.0253`.

**Result (negative for monotonicity).** At 2%, Operator Tucker NRMSE is
`0.293/0.273/0.331`: cutoff 12 has the lowest oracle bias but the worst recovery.
At 5%, results are `0.235/0.206/0.205`; at 10%,
`0.201/0.158/0.159`.  Thus cutoff 8 is the best practical compromise here.
At 2% all cutoffs lose Neural F-Tucker on mean; at 10%, cutoffs 8 and 12 win all
three seeds while cutoff 5 does not.

**Decision.** The hypothesis is rejected and replaced by the actual mechanism:
finite operator spectra trade approximation bias against factor-estimation
variance.  Oracle residual is necessary but insufficient to select a cutoff.
Artifact: `results/diffusion_cutoff_summary_r2/summary.json`.

## Physical iteration 3 — matched Tucker rank sensitivity

**Falsifiable hypothesis.** The positive result should not disappear when both
Operator and Neural Functional Tucker receive the same under/matched/over-sized
multilinear ranks.  Conversely, if the default core was hand-picked, changing
rank should reverse the 10% ordering.

**Protocol.** Fix contrast 1, cutoff 8; sweep ranks `(3,4,4)`, `(4,5,5)`, and
`(6,7,7)` (core sizes 48/100/294).  All other models, masks, seeds and budgets
are unchanged.

**Result.** At 10%, Operator vs Neural F-Tucker NRMSE is
`0.186 vs 0.183`, `0.158 vs 0.189`, and `0.145 vs 0.154`.  Default and large
cores retain a positive mean signal; the small core ties.  At 2% the differences
are `+0.039/-0.011/+0.006` in neural-minus-operator NRMSE with large standard
deviations, so there is no stable rank-insensitive extreme-sparse claim.  At 5%
all differences are within about `0.02`.

**Decision.** The 10% signal is not an artifact of one exact core size, but
rank selection is itself part of the bias–variance problem.  No further
architecture was added.  Artifact: `results/diffusion_rank_summary_r3/summary.json`.

## Current gate after three physical iterations

- **Keep:** physical Green-response benchmark, measured projection residual,
  cutoff/rank phase analysis, Operator CP and method-matched neural baselines.
- **Do not claim yet:** superiority at 2%--5%, monotonic benefit from lower
  residual, or generic PDE performance.
- **Next confirmation gate:** freeze cutoff/rank using the present seeds, then
  test fresh seeds and structured source/receiver fibers.  Promote the line only
  if Operator Tucker wins at least 4/5 fresh seeds at one ratio no larger than
  10%, while remaining absolutely useful (NRMSE well below 1).
