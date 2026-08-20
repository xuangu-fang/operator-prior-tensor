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

## Design probe R5a — the regular 2-D joint-vs-per-axis POC is degenerate

**Status.** Pre-registration probe only.  No seeds, no gate and no confirmation
budget were consumed; the models below were never scored for promotion.

**What was built.** `src/geoaware/operator_diagnostics.py` (separability,
low-frequency subspace and grouped projection residuals, mass-orthonormal
generalized eigenpairs, eigenvalue-ranked product basis),
`src/geoaware/joint_diffusion_2d.py` (Q1 finite-element anisotropic diffusion
`A_eta = [[a_x, eta c],[eta c, a_y]]` on the unit square) and
`src/geoaware/grouped_operator_tucker.py` (order-M Tucker over a coordinate
partition, with operator / smoothness-penalized table / neural group factors).
The frozen order-3 implementation and all R4 artifacts are untouched.

**Correctness checks that did pass.** At `eta = 0` the Q1 stiffness matrix
equals `K_x (x) M_y + M_x (x) K_y` to `1e-10`, the consistent mass matrix
factors as `M_x (x) M_y`, and both `epsilon_sep` and `epsilon_sub` vanish, so
the joint and per-axis bases span the same space by construction rather than by
luck.  Generalized eigenvectors satisfy `Phi^T M Phi = I` and
`K Phi = M Phi Lambda`; grouping and ungrouping preserve every entry; a fixed
seed reproduces the operator, mask and fit bitwise.

**Falsified hypothesis 1: `epsilon_sub` predicts representational advantage.**
At `eta = .6` the joint and per-axis low-frequency subspaces are far apart
(`epsilon_sub = .374`) while their projection residuals on the field are
indistinguishable (`.0556` vs `.0546`).  Two very different 16-dimensional
low-frequency spaces are equally good for a generically smooth field, so
operator-level non-separability does not by itself create a completion
advantage.

**Falsified hypothesis 2: the advantage regime is usable.**  The joint basis
does win decisively at long diffusion times — residual `.0008` versus `.0460`
at `eta = .9`, and still `.0008` versus `.0190` when the product basis is given
36 columns.  But the spatial 99%-energy rank of the tensor falls from 11 to 5
over the same interval: the regime where the joint operator wins is the regime
where free decay has collapsed onto the operator's own slowest eigenvectors,
which the learner is handed exactly.  That is the eigenbasis-generated truth
listed as a stop condition in the report, not evidence.

**Two structural limits of this generator.**  The PSD constraint `|eta s| <= 1`
keeps the coupling perturbative, so `epsilon_sep` never exceeds `.090`; and a
linear semigroup driven by a few smooth bumps is intrinsically low rank
(time-mode participation ratio `1.1--1.5` at every time span), which is why a
400-step seven-model smoke placed every method near NRMSE `.25` at 10%.

**Decision.** Do not promote the standalone regular-grid POC-B.  Fixing it
would require both forced broad-spectrum responses and a rotated
strongly-anisotropic operator, and would still argue the grouping question
indirectly.  Fold the comparison into the irregular-domain POC instead, where
the per-axis comparator is the bounding-box product basis that ignores holes —
a control practitioners actually use, on a field whose structure is not the
learner's own eigenbasis.  `grouped_operator_tucker.py`,
`operator_diagnostics.py` and the `spatial_sensors` mask carry over unchanged;
`joint_diffusion_2d.py` is retained as the exactness fixture for the grouped
implementation.

## Iteration 5b — geometry-aware Green completion on a holed domain (screen)

**Status.** Three-seed screen on selection seeds `41--43`, 400 updates.  No
confirmation seeds were consumed and no gate was declared passed.

**Setting.** ``Y(t, receiver-node, source-node)`` for a diffusion Green response
on the unit square minus two circular discs, Neumann outside and Dirichlet on
the hole rims, P1 finite elements, shape `16 x 211 x 20`.  Both spatial axes
index the same mesh nodes, so the setting is the frozen 1-D benchmark with the
spatial coordinate replaced by a mesh.  Truth uses a variable diffusivity; every
learner basis comes from the constant-coefficient operator, so geometry is known
metadata and the material is not.  All four spatial bases have 32 columns on the
identical node set, which removes any need for interpolation between controls.

**Result at 10% random entries.**  The geometry-aware basis is best of seven:
`0.363±0.020` versus topology-erased `0.400±0.023` (3/3 paired wins),
bounding-box cosines `0.422±0.023` (3/3), a wider coordinate MLP
`0.383±0.056` (2/3), a free factor table `1.082±0.482` (3/3), the same table
with the mesh operator as a smoothness penalty `1.027±0.439` (3/3), and the
node-permuted control `1.647±0.231` (3/3).  Near the hole rims the ordering is
the same: `0.585 / 0.613 / 0.659`.  Ordinary discrete tensor completion and
Laplacian-smoothed completion are both around NRMSE 1 where the geometry-aware
model reaches `0.36`, and matching the smoothness penalty without truncating the
spectrum does not recover the gap.

**Negative result 1: the advantage does not extend to sparser regimes.**  At 5%
the geometry-aware model `0.888±0.059` loses to bounding-box `0.706±0.086`
(0/3) and to topology-erased `0.745±0.213` (1/3).  At 2% every spectral variant
exceeds NRMSE 1.  The coordinate MLP is far stronger at both ratios
(`0.458` and `0.682`) and remains competitive at 10%.  With 32 basis columns
and ranks `(4,8,8)` the model has 832 parameters, four times the frozen 1-D
configuration, so cutoff and rank have not yet been selected for this benchmark;
that selection is still allowed on seeds 41--43 and must precede any
confirmation.

**Negative result 2: the receiver-fiber mask does not run.**  Every model lies
between `0.97` and `1.8` at all three ratios and the permuted control beats the
proposed model 3/3 at 10%.  With 16 times and 20 sources a 10% fiber mask keeps
only about 32 of 320 `(time, source)` pairs, so the factors are not identifiable
rather than badly estimated.  This cell is reported as a non-functioning
protocol, not as a comparison.

**Prior diagnostic that must be reported with the above.**  The geometry-aware
basis has the *worst* product-space projection residual of the three spectral
variants (`0.041` versus `0.023` and `0.019`, in both the entrywise and the
mass-weighted norm).  Its advantage at 10% therefore comes from estimation
variance, not from a lower approximation floor — the same mechanism the cutoff
sweep established in physical iteration 2, now reproduced on a mesh.

**Decision.** Continue.  The claim that ordinary and merely-smoothed tensor
completion fail on an irregular domain while an operator-defined subspace works
is supported at 10%.  Before any fresh-seed confirmation, select basis cutoff
and Tucker rank on seeds 41--43, and either fix or drop the fiber protocol.
Artifact: `results/irregular_green_screen_r5b/results.json`.

## Iteration 6 — geometry families at the recovery level, and why the first two failed

**Two settings, five polygonal domains, three seeds, 400 updates, selection
seeds 41--43.**  Settings: the Green response `Y(t, receiver-node, source-node)`
and a plainer field `Y(scenario, time, node)`; domains: plain square,
centred obstacle, two obstacles, L shape, U shape; obstacles insulating.

**What held.**  Ordinary discrete Tucker (`0.56--0.98` at 10%) and a
graph-smoothed table (`0.46--0.91`) fail on every domain where the proposed
model reaches `0.15--0.36`, and the node-permuted control never drops below
`1.3`.  On the plain square the geometry-aware and topology-erased models return
*identical* numbers, as they must: with no obstacle they are the same operator.

**What did not hold.**  Geometry-aware versus geometry-blind *continuous*
factors was inside seed noise at every ratio, on both settings, despite the
geometry-aware basis having a 12--17x lower projection residual.  A coordinate
MLP was the strongest baseline and beat the proposed model at 2% and 5%.

**Diagnosis, and it is arithmetic rather than tuning.**  The bias floors were
`0.002` (geometry-aware) against `0.02--0.03` (blind) while total held-out error
was `0.25--0.70`.  Approximation bias was one to ten percent of the error
budget, so the basis could not matter: everything was estimation variance.  A
second confound: with 20 scenarios and 16 times, a 5% random mask observes each
spatial node about sixteen times, which is not sparse in the coordinate the
geometry prior speaks about.

**The two changes that follow from the diagnosis.**  First, barriers.  Thin
impermeable baffles inside one fixed mesh raise the blind bias floor from `0.03`
to `0.17--0.25` while leaving the geometry-aware floor at `0.003`, a ratio of
55--101x; all layouts share one mesh and one node set, so controls differ only
in whether their operator knows the walls.  Second, the mask: `spatial_sensors`
observes complete trajectories at a few nodes, making nodes the scarce
coordinate.

**A control that turned out to be mislabelled.**  The Laplacian-regularized
table built its penalty from the wall-aware operator, so it was a second
*geometry-aware* method rather than the "any smoothness would do" control.
Splitting it into `laplacian_geo` and `laplacian_blind` turns the comparison
into a 2x2 — geometry known or not, crossed with spectral truncation or a
penalized free table — which isolates geometry as the active ingredient instead
of confounding it with the representation.

Artifacts: `results/geometry_family_screen_r6`, `results/field_family_screen_r6`
and their summaries.

## Iteration 7 — barriers, sensor placement, and an identifiability rule

**Design.** One fixed square mesh, five barrier layouts ordered by how much
geometry there is to know: `open` (none), `labyrinth`, `arc` (curved),
`chamber`, `sealed_4` (four sealed quadrants).  Barriers are thin bands of
near-zero conductivity inside the mesh, so every layout shares one node set and
a control differs from the proposed model in exactly one respect.  The learner
is told where the barriers are and never sees the smooth background material.

**The 2x2 that isolates geometry.**  Two axes crossed: whether the operator
knows the barriers, and whether the factor is a truncated spectral basis or a
free table under the matching smoothness penalty.  On `sealed_4` at 5% sensors,
the geometry-aware spectral model reaches `0.173` and the geometry-aware
penalized table `0.180`, while their geometry-blind counterparts reach `0.372`
and `1.078`.  On `open`, where there is nothing to know, the two spectral
variants return identical numbers and so do the two penalized ones.  Geometry,
not the representation, is the active ingredient — and the spectral form buys
the same accuracy with 5.4x fewer parameters (528 against 2864).

**A protocol bug found and fixed before it consumed a run.**  The experiment
runner had been calling the sensor mask without naming the node axis, so
"spatial sensors" was selecting `(time, node)` pairs rather than nodes.  The
corrected protocol observes complete trajectories at a few mesh nodes and holds
out every other node.

**An identifiability rule, not a tuning knob.**  Under the corrected protocol
the first configuration collapsed: with 16 observed nodes out of 324, a basis
cutoff of 32 and rank 8 leaves 256 node coefficients constrained at 16
locations, and every spectral model returned NRMSE near 1 while only a
coordinate network survived.  The cutoff has to stay commensurate with the
number of observed nodes.  At 10% sensors (32 of 324) with cutoff 10 and rank
`(4,4,6)`, the geometry-aware model reaches `0.151` against `0.512` for the
geometry-blind spectral basis, `0.422` for a wide coordinate network and
`0.456` for a parameter-matched one — a 2.6--3.4x margin that is stable across
cutoffs 6, 10 and 16 (`0.168 / 0.151 / 0.166`), so it does not rest on one
hand-picked truncation.

**Status.** Full five-layout, three-ratio, three-seed tables running under both
the random-entry and sensor protocols on selection seeds 41--43.  Cutoff and
rank were chosen here and must be frozen before any fresh-seed confirmation.

### 7b — spectral ARD does not remove the cutoff hyperparameter (negative)

Since the cutoff turned out to be an identifiability constraint, the obvious
next move was to learn it: one precision per basis column, updated by the
sparse-Bayesian fixed point ``alpha_k = R / sum_r W_kr^2``, so that an oversized
cutoff would be pruned instead of fatal.  It does not work here, and the reason
is structural rather than a matter of tuning.  With ``W`` a point estimate the
update has no posterior-covariance term, so the penalty contributes
``reg * R`` per column irrespective of its coefficients and nothing is driven
out: at cutoff 32 the effective dimension stayed 32 and held-out NRMSE moved
only from `0.655` to `0.643`, against `0.152` at cutoff 10.  Recovering real
pruning would require a factor posterior, which is exactly the component the
report rules out for this paper.  The code was reverted; the cutoff is reported
as a selected-and-frozen bias--variance hyperparameter with an explicit rule.

## Iteration 8 — predeclared fresh-seed confirmation (written before the run)

Everything below is fixed now, on selection seeds only, and must not be changed
after the confirmation numbers are seen.

**Frozen configuration.**  Barrier setting `wall_field_tensor`; tensor
`Y(scenario, time, node)` with 20 scenarios, 16 times, mesh resolution 18 on the
unit square; truth modes 60; background log-diffusivity contrast `.3`; reaction
`.15`; time span `(.15, 3.)`; observation noise 10% of the observed standard
deviation; observed-only normalization.  Model: basis cutoff 10, ranks
`(4, 4, 6)`, Sobolev power 1.5, regularization `.002`, AdamW at `3e-3`, 400
updates, random cold start, no early stopping.

**Layouts.** `open`, `labyrinth`, `arc`, `chamber`, `sealed_4`.  `open` is the
control in which the advantage must vanish.

**Protocols.** Random entries and spatial sensors, at 2%, 5% and 10%.

**Baselines.** `topology_erased` and `bounding_box` (geometry-blind spectral),
`laplacian_geo` and `laplacian_blind` (penalized free table, geometry known or
not), `neural_coords` (wide coordinate MLP) and `neural_matched` (parameter
matched), `discrete_table` (no prior), `permuted` (destructive control).

**Confirmation seeds.** `101, 102, 103, 104, 105`.  These have never been run on
this benchmark and may not be used to choose anything.

**Predeclared gate.**  On the sensor protocol at 10%, the geometry-aware model
must reach at least 4/5 paired wins against *both* geometry-blind spectral
bases and against the wide coordinate network, on at least the two strongest
layouts (`chamber`, `sealed_4`), with mean NRMSE below `.5`.  On `open` the
geometry-aware and geometry-blind spectral models must agree to within seed
noise; a win there would indicate a confound and invalidates the round.

**What a failure means.** If the gate fails only at 2%, the claim is scoped to
5--10% and extreme sparsity becomes a stated NO-GO region, as in the frozen
one-dimensional work.  If it fails on `chamber` and `sealed_4` at 10%, the
barrier mechanism does not survive fresh seeds and the line returns to
diagnostics rather than being re-tuned.

### 8a — barrier-family main tables on selection seeds

Five layouts, two protocols, three ratios, three seeds, nine models, 400
updates, cutoff 10, ranks `(4,4,6)`.  Held-out NRMSE, mean ± sample std.

**Spatial sensors, 10% of nodes observed for their whole trajectory.**

| model | open | labyrinth | arc | chamber | sealed_4 |
|---|---:|---:|---:|---:|---:|
| geometry operator (ours) | 0.217 | 0.218 | 0.223 | **0.197** | **0.158** |
| topology erased | 0.217 | 0.221 | 0.230 | 0.401 | 0.502 |
| bounding-box product | 0.222 | 0.234 | 0.255 | 0.399 | 0.502 |
| neural coordinates (wide) | 0.269 | 0.249 | 0.289 | 0.326 | 0.408 |
| neural coordinates (matched) | 0.284 | 0.308 | 0.326 | 0.393 | 0.468 |
| discrete Tucker | 1.476 | 1.472 | 1.495 | 1.506 | 1.617 |
| permuted control | 1.204 | 1.202 | 1.221 | 1.310 | 1.220 |

Three things hold at once.  On `open` the geometry-aware and topology-erased
models return *identical* numbers, as they must when there is no barrier to
know, so the advantage is not a capacity or conditioning artifact.  The margin
then grows monotonically with how much geometry there is: nothing on `open`,
about one percent on `labyrinth` and `arc`, 2.0x on `chamber` and 3.2x on
`sealed_4`.  And the proposed model is nearly flat across the family
(`0.217 -> 0.158`) while every geometry-blind method degrades
(`0.217 -> 0.502`): barriers do not make the task harder for a model that knows
about them.

**Random entries.**  The same ordering, with smaller margins, and one honest
exception.  At 10% on `sealed_4` the proposed model reaches `0.155` against
`0.299` for the blind spectral bases and `0.218` for the wide coordinate
network; on `open` it again matches topology-erased exactly (`0.260`), while the
coordinate network is the better model on the weak-geometry layouts
(`0.218` against `0.260`).  Under uniformly random entries every node is
observed many times, so a coordinate regressor has little to gain from
geometry; the sensor protocol is where the prior is actually needed.

**A regime split in the 2x2.**  Under random entries the geometry-aware
penalized table is competitive (`0.161` on `sealed_4` at 10%, against `0.155`
for the spectral form), reproducing the earlier finding that geometry rather
than truncation is the active ingredient.  Under sparse spatial sensors it
collapses (`0.796--1.46`), because a free table carries 1944 node parameters
constrained at 32 nodes.  Geometry alone is therefore not sufficient when the
scarce coordinate is space: the truncation is what makes the geometry usable.

Artifacts: `results/wall_family_sensors_r7`, `results/wall_family_random_r7`
and their summaries.

### 8b — where the geometry advantage switches on

Reading the sensor table against the bias floor a geometry-blind basis pays
gives a quantitative boundary rather than a qualitative claim.

| layout | blind bias floor | blind / ours | neural / ours |
|---|---:|---:|---:|
| open | 0.082 | 1.00 | 1.24 |
| labyrinth | 0.104 | 1.01 | 1.14 |
| arc | 0.116 | 1.03 | 1.29 |
| chamber | 0.303 | 2.03 | 1.65 |
| sealed_4 | 0.391 | 3.18 | 2.58 |

The proposed model attains `0.158--0.223` across the family.  The advantage
against the blind spectral basis is exactly 1.00 while the blind bias floor
stays below that attainable error, and rises steeply once the floor exceeds it.
So the operative statement is not "geometry helps" but: **a geometry prior pays
off precisely when the approximation error of ignoring the geometry becomes
comparable to the error the estimator could otherwise achieve.**  That is the
same bias--variance boundary the frozen one-dimensional work established for
spectral cutoff, now measured along a geometric axis, and it predicts where the
method is and is not worth using.

### 8c — the factor space transfers between geometries, with a stated direction

Fit on one barrier layout at 10% sensors, then move to another layout by
swapping only the spatial basis.  Three adaptations are compared so that a
core-only refit — the natural move for an operator model and a poor one for a
network — is not quoted as if it were everyone's best option: core-only,
full fine-tuning, and training on the target observations alone.  Target data is
2% sensors.  Averages over five pairs and three seeds:

| model | source | zero-shot | few-shot (core) | few-shot (full) | scratch |
|---|---:|---:|---:|---:|---:|
| geometry operator (ours) | 0.191 | 1.512 | **0.535** | 0.976 | 1.013 |
| topology erased | 0.370 | 0.649 | 7.703 | 4.128 | 3.443 |
| neural coordinates | 0.340 | 0.651 | 3.764 | 0.742 | 1.039 |
| discrete Tucker | 1.543 | 1.502 | 20.086 | 1.425 | 1.482 |

The cleanest statement is internal: refitting only the core after a basis swap
beats training the same model on the target alone in **15 of 15** pair-seed
cells (`0.535` against `1.013`).  The representation is genuinely carried over
rather than relearned, and full fine-tuning *hurts* (`0.976`), so the value lies
in not re-estimating the factors from 2% of a new domain.

Against each baseline's own best adaptation the result has a direction.
Transferring *into* a strongly barriered target wins 3/3 everywhere
(`chamber -> sealed_4`, `open -> sealed_4`), while `sealed_4 -> open` loses 0/3
to topology-erased.  That is a consistency check rather than a defect: on a
barrier-free target the topology-erased operator *is* the correct one, and a
model carrying structure from a sealed source brings structure the target does
not have.  The rule to state is that transfer helps when the target geometry is
at least as structured as the source.

**Limitation.** Zero-shot is poor for the proposed model (`1.512`, worse than
the blind baselines' `0.65`): swapping the basis changes the column
normalization, so transferred coefficients are not calibrated in absolute terms.
Only the function space transfers, not the coordinates in it.

## Iteration 9 — frozen fresh-seed confirmation: the predeclared gate passes

Seeds `101--105`, never previously run on this benchmark.  Configuration exactly
as predeclared in Iteration 8: cutoff 10, ranks `(4,4,6)`, 400 updates, 10%
noise, random cold start.  Nothing was tuned in response to these numbers.

**Sensor protocol at 10%, mean ± sample std over five fresh seeds.**

| model | open | labyrinth | arc | chamber | sealed_4 |
|---|---:|---:|---:|---:|---:|
| geometry operator (ours) | 0.260±0.019 | 0.237±0.026 | 0.238±0.009 | **0.246±0.096** | **0.167±0.013** |
| topology erased | 0.260±0.019 | 0.271±0.025 | 0.267±0.019 | 0.477±0.111 | 0.629±0.223 |
| bounding-box product | 0.247±0.035 | 0.258±0.041 | 0.269±0.050 | 0.475±0.114 | 0.627±0.221 |
| neural coordinates (wide) | 0.269±0.020 | 0.279±0.017 | 0.305±0.026 | 0.342±0.048 | 0.467±0.078 |
| neural coordinates (matched) | 0.274±0.025 | 0.290±0.039 | 0.319±0.062 | 0.372±0.026 | 0.540±0.063 |
| discrete Tucker | 1.578±0.194 | 1.575±0.193 | 1.584±0.190 | 1.596±0.151 | 1.578±0.152 |
| permuted control | 1.232±0.054 | 1.222±0.067 | 1.244±0.092 | 1.254±0.073 | 1.199±0.045 |

**Gate, item by item.**  On `chamber`: 5/5 paired wins against topology-erased,
5/5 against the bounding-box basis, 4/5 against the wide coordinate network,
mean `0.246 < .5`.  On `sealed_4`: 5/5, 5/5, 5/5, mean `0.167 < .5`.  On the
barrier-free control the geometry-aware and topology-erased models return
*identical* numbers to three decimals with identical standard deviations, so the
invalidation clause is not triggered — there is no capacity or conditioning
confound to explain the margin elsewhere.  **The predeclared gate passes.**

**Random entries, all three ratios.**  Stronger than the sensor protocol and,
unlike the frozen one-dimensional result, it holds down to 2%.  On `sealed_4`
the proposed model reaches `0.150 / 0.160 / 0.180` at 10% / 5% / 2% against
`0.418 / 0.426 / 0.452` for the blind spectral bases and `0.236 / 0.238 / 0.263`
for the wide coordinate network, with 5/5 paired wins against both on every
layout that has barriers, and identical numbers to topology-erased on `open`.

**Scope.**  The claim confirmed here is conditional and stated as such: an
operator-defined factor space helps exactly on domains whose geometry carries
information, by a margin that grows with how much of it there is, and helps not
at all when there is none.  Extreme sparsity is no longer a NO-GO region under
random entries, but the sensor protocol at 5% remains high-variance
(`0.304±0.278` on `sealed_4`) and is not claimed.

Artifacts: `results/wall_confirmation_sensors_r9`,
`results/wall_confirmation_random_r9`.

## Iteration 10 — a second geometry family, and a prediction rather than a description

Iteration 8b read a phase boundary off the barrier family: the geometry prior
pays off once the bias floor of ignoring the geometry approaches the error the
estimator could otherwise attain.  Fitted on one family, that is a description.
This iteration tests it out of sample on a family whose geometry enters by a
different mechanism entirely.

**Design.**  Polygonal domains instead of internal barriers: a plain square (the
control), one circular obstacle, two obstacles, an L and a U.  Geometry now
enters through the *shape of the domain* — nodes are absent where the obstacle
is, and a reentrant corner forces the field around — rather than through
material inside a fixed mesh.  Nothing else changes: the same PDE, the same
tensor semantics `Y(scenario, time, node)`, the same frozen learner
configuration (cutoff 10, ranks `(4,4,6)`, 400 updates, 10% noise), the same
nine models and the same fresh seeds `101--105`.

Because the domains differ, layouts no longer share a node set, so this family
cannot support the exactly-one-difference argument the barrier family carries.
It is reported as an out-of-sample test of the boundary, not as a replacement
main table.

**Result — the advantage is real, consistent, and modest.**  Random entries at
10%, mean over five seeds, with ours-versus-topology-erased paired wins:

| layout | blind bias floor | ours | topology erased | neural coords | discrete Tucker | paired wins |
|---|---:|---:|---:|---:|---:|---|
| `square` (control) | 0.081 | 0.240 | 0.240 | 0.253 | 0.796 | 0/5 (exact tie) |
| `L_shape` | 0.086 | 0.181 | 0.206 | 0.215 | 0.507 | 5/5 |
| `center_hole` | 0.114 | 0.180 | 0.209 | 0.230 | 0.707 | 5/5 |
| `two_holes` | 0.130 | 0.213 | 0.241 | 0.271 | 0.766 | 4/5 |
| `U_shape` | 0.162 | 0.162 | 0.229 | 0.184 | 0.484 | 5/5 |

On the plain square the geometry-aware and topology-erased models again return
*identical* numbers, so the negative control holds in a second family.  The
margins against the blind spectral basis are 1.13--1.42x, well below the
1.74--2.79x the barrier family reaches, and that is exactly what the boundary
predicts from these smaller bias floors.  Under spatial sensors the ordering is
the same (1.12--1.27x, 4/5 wins on every geometry-bearing layout).

**Limitation, stated plainly.**  Under spatial sensors at 2% and 5% the spectral
models lose to a coordinate network on this family (at 2%, `0.99` against
`0.56` on `U_shape`).  With roughly six observed nodes out of 300, cutoff 10 and
rank 6 leave sixty node coefficients constrained at six locations: the same
identifiability rule from Iteration 7, now on the wrong side of it.  The claim
under sparse sensors remains a 10% claim.

### 10b — one training-free scalar orders both families

Pooling the ten geometries from the two families and asking whether the bias
floor — computable from the data and the candidate bases *before any model is
fitted* — orders the realized advantage:

| protocol | Spearman(floor, blind/ours) | control: Spearman(1/ours, blind/ours) |
|---|---:|---:|
| spatial sensors, 10% | **+0.915** | +0.600 |
| random entries, 10% | +0.806 | +0.903 |

The control is the point.  A normalized x axis (floor divided by the error ours
attains) scores higher — `+0.915` and `+0.879` — but it shares a denominator
with the reported advantage, and under random entries `1/ours` *alone* reaches
`+0.903`.  So the normalized version cannot be quoted as evidence, and the
random-entry correlation is not claimed either.  What survives is the sensor
protocol, where the un-shared pre-fit scalar reaches `+0.915` while the
shared-denominator control only manages `+0.600`: there, the boundary predicts
across mechanisms rather than merely describing one family.

That is the strongest form the claim has taken so far — the method comes with a
diagnostic that says, without training anything, whether it is worth using.

Artifacts: `results/irregular_domain_sensors_r10`,
`results/irregular_domain_random_r10`, their summaries, and
`results/phase_curve_r10`.

### 10c — the advantage survives without the inverse crime

Every result so far solved the truth with the same discretization the learner's
operator is built from: one mesh, one node ordering, one P1 assembly, differing
only in material.  The truth field therefore lay exactly in the span of sixty
eigenvectors of a close relative of the learner's own operator, and the forward
model error was identically zero — the inverse crime.  Since the claim is
precisely that the operator's eigenbasis spans the field well, that is circular,
and it favours the geometry-aware basis specifically, because its operator is
the near relative and the blind one is not.

**What changed.**  The truth is now solved on an independently seeded mesh two
or three times finer — 1097 and 2381 nodes against the learner's 324, with
different node positions and a different triangulation — and interpolated onto
the learner's nodes with P1 elements.  Point location uses the fine mesh's own
triangle list, so no triangle spanning a barrier is ever used to interpolate
across it.  The learner's operators are untouched, and the interpolation happens
before any basis is applied, so every basis pays the same error.

**Cost, measured.**  The geometry-aware bias floor on `sealed_4` rises from
`0.062` to `0.125` (2x) and `0.065` (3x); the blind floor barely moves
(`0.391` to `0.335` and `0.383`).  Removing the crime does penalize the proposed
model, and only the proposed model — as it should.

**Result — sensors at 10%, ours against topology-erased, paired wins over the
five confirmation seeds:**

| refinement | open | labyrinth | arc | chamber | sealed_4 |
|---|---|---|---|---|---|
| 1x (crime present) | 0/5 (1.00x) | 4/5 (1.14x) | 5/5 (1.12x) | 5/5 (1.94x) | 5/5 (3.76x) |
| **2x (crime avoided)** | 0/5 (1.00x) | 5/5 (1.12x) | 5/5 (1.39x) | 4/5 (1.79x) | 5/5 (2.21x) |
| **3x (crime avoided)** | 0/5 (1.00x) | 5/5 (1.13x) | 5/5 (1.53x) | 5/5 (1.89x) | 5/5 (3.54x) |

Random entries give the same picture (`1.17--2.69x` at 3x, 5/5 everywhere with a
barrier).  Absolute errors rise across the board — `labyrinth` from `0.237` to
`0.375` — because the learner's coarse operator now carries a real
discretization error where before it carried none.  The ordering, the paired
wins and the exact tie on `open` all survive, and two independent refinement
factors agree, so this is not one lucky mesh.

**A caveat that is reported rather than hidden.**  Below resolution 18 the
result is not reliable: the baffles are `0.04` wide against a coarse element of
`0.056`, so the learner's own operator represents them as a jagged
single-element layer.  At resolution 16 with a 2x truth the blind-to-aware bias
ratio collapses to `1.2--1.4` from `3.3--5.6`.  That is a statement about
under-resolved sub-element barriers, not about the method, but it bounds where
the benchmark is meaningful.

Artifacts: `results/wall_refined_truth_x2_r10`,
`results/wall_refined_truth_x3_r10`, `results/inverse_crime_summary_r10`,
`results/inverse_crime_summary_random_r10`.
