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

## Physical iteration 4 — frozen fresh-seed and structured-fiber confirmation

**Predeclared protocol.** No architecture or hyperparameter was selected on
this round.  Freeze diffusion contrast 1, basis cutoff 8, truth modes 14,
Tucker rank `(4,5,5)`, regularization `.002`, 10% noise, random initialization
and 400 AdamW updates.  Replace selection seeds 41--43 by fresh seeds
101--105.  Evaluate 2%/5%/10% under random entries, complete source fibers and
complete receiver fibers.  A source fiber observes all source positions at a
fixed `(time, receiver)` pair; receiver fibers are analogous.

**Fair baselines and control.** The main strong baseline is a wide Neural
Functional Tucker with the same ranks/core and 8,130 parameters.  A hidden
width-3 version has 210 parameters, almost exactly matching Operator Tucker's
212.  A permuted-basis Tucker has the same 212 parameters, optimizer and
eigenvalues and is the destructive negative control.

**Result.** The random 10% cell passes with Operator Tucker
`0.1645±0.0102` vs wide neural `0.2065±0.0536`, 4/5 paired wins.  Receiver
fibers at 10% independently pass with `0.2165±0.0517` vs `0.2695±0.1117`,
also 4/5.  The matched neural model is `0.4279` and `0.4225` respectively;
the wrong operator is `0.9417` and `0.9597`.  Thus the result is not explained
by parameter count or by the Tucker decoder alone.

**Negative result.** Source fibers do not meet the gate: at 10%, Operator is
`0.2937±0.1890` vs wide neural `0.2562±0.1177`, only 3/5 wins.  Both complete
fiber masks strongly favor the neural baseline at 2%; receiver-fiber Operator
even exceeds NRMSE 1 on one seed.  No cutoff, rank, initializer or step budget
was changed in response.

**Decision.** Conditional promotion.  The predeclared gate is met by random
and one genuinely structured mask at 10%, but the line must be presented as a
bias--variance phase-boundary paper.  Extreme sparse and source-fiber regimes
remain explicit NO-GO regions.  Raw artifacts:
`results/diffusion_confirmation_r4`,
`results/diffusion_confirmation_matched_r4`; audited aggregate:
`results/diffusion_confirmation_summary_r4/summary.json`.

## Planned iteration 5 — group-wise operator and separability phase diagram

**Status.** Method/experiment design only; no R5 result exists yet.

**Correction to the formulation.** A physical operator belongs to the joint
coordinate domain on which it is defined. It is not assumed that every tensor
axis has its own PDE. We therefore generalize mode-wise Operator Tucker to a
coordinate partition: a group such as $(x,y)$ receives a joint operator factor,
while a group without a reliable operator uses a neural functional factor.
The existing Green tensor remains unchanged because its time, receiver, and
source factors are all derivable from one Green operator.

**Falsifiable hypothesis.** On a regular 2-D diffusion family, per-axis factors
should match the joint grouped factor when the operator is an exact Kronecker
sum. As a controlled nonseparable coupling grows, operator separability and
low-frequency subspace residuals should grow, and joint-minus-per-axis recovery
advantage should follow the same phase trend.

**Protocol to implement.** Use tensors with axes time × x × y × scenario and
groups $\{\{t\},\{x,y\},\{s\}\}$. Compare joint grouped Operator Tucker,
per-axis Operator Tucker, wrong-joint control, grouped Neural Functional Tucker,
and Laplacian-regularized discrete Tucker at 2%/5%/10%, random entries and fixed
spatial sensors. Report matched-parameter and matched-spatial-latent budgets,
basis construction cost, $\epsilon_{\mathrm{sep}}$,
$\epsilon_{\mathrm{sub}}$, projection residual, and held-out NRMSE.

**Predeclared screen.** Three fresh development seeds and 400 updates. At exact
separability, joint and per-axis must be close; otherwise audit fairness. At
nonzero residual, joint must show a coherent trend and win 3/3 in at least one
5% or 10% structured-sensor cell before the irregular-domain implementation is
promoted. Per-axis remains an efficient approximation and ablation, not a
straw-man baseline.

**Execution order.** Preserve R4 → implement regular 2-D grouped phase diagram
→ implement irregular FEM/hole grouped mode → test hybrid neural unknown modes
→ consume an external PDE benchmark.
