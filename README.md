# Operator-Prior Tensor

**Geometry-aware tensor factorization through operator-defined functional
subspaces.**

The whole repository tests one sentence:

> **Whether a geometry prior is worth using can be computed before any model is
> fitted.**  Where it is worth using, writing the geometry into the function
> space that the spatial factor lives in reconstructs a sparsely observed field
> substantially better; where it is not, the advantage is exactly zero.

Ordinary tensor completion treats a node as an index.  Two points on opposite
sides of an impermeable wall are then as related as two neighbours, which is
wrong for the physics and, when the observations are sensors rather than random
entries, leaves an unobserved node constrained by nothing at all.  We take the
known geometry, assemble the corresponding operator, and let its leading
eigenfunctions be the dictionary the spatial factor is written in.

This repository is Track 1 of the [Physics-Informed Tensor Learning
Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor).

## The benchmark: four geometries, one code path

`src/geoaware/benchmark.py` builds every dataset.  Four families vary only *how*
geometry enters the problem, and each defines its own geometry-blind control —
the basis a practitioner would use having ignored that geometry:

| family | geometry is | nodes | geometry-blind control |
|---|---|---:|---|
| `plane_barrier` | impermeable walls inside a square | 5 520 | the same operator, walls removed |
| `plane_domain` | circular holes and reentrant corners | 3 941–5 520 | a triangulation connecting straight across |
| `volume_barrier` | partitions inside a cube | 8 000 | the same operator, partitions removed |
| `sphere` | curvature of a closed surface (shallow water) | 10 242 | the latitude–longitude product basis |

The proposed model and its blind counterpart share the node set, the decoder,
the optimizer, the prior and the closed-form core posterior.  They differ in one
thing.  The learner is told where the obstacles are and never sees the smooth
background material, so this is a geometry prior, not known physics.

Two sampling protocols ask different questions.  Under **random entries** every
node appears in many observed entries, so a classical factorization is
well-posed and the comparison is a fair fight.  Under **spatial sensors** an
unobserved node appears in *no* observed entry: its factor row is constrained by
zero equations, and CP-ALS returns exactly nothing there.  That is a property
asserted in `tests/test_benchmark.py`, not a margin.

Baselines are the ones a tensor reader asks for first — CP by ALS and Tucker by
HOOI, taken from TensorLy rather than reimplemented — plus their functional
counterparts whose factors are coordinate networks.  CP is also available inside
the proposed model as a diagonal core, so a CP baseline can share every part of
the fitting procedure and differ only in the model.

## The main result

Held-out NRMSE at 10% of nodes observed for their whole trajectory, five fresh
seeds, ours against **the same model with the geometry removed** -- same node
set, same decoder, same optimizer, same prior, same closed-form core posterior:

| layout | ours | geometry removed | ratio | paired wins |
|---|---:|---:|---:|---|
| `plane_barrier/open` *(control)* | 0.117 | 0.117 | **1.00** | 1/5 |
| `plane_domain/square` *(control)* | 0.117 | 0.117 | **1.00** | 1/5 |
| `volume_barrier/open` *(control)* | 0.273 | 0.273 | **1.00** | 0/5 |
| `plane_barrier/labyrinth` | 0.111 | 0.245 | 2.20 | 5/5 |
| `plane_barrier/chamber` | 0.109 | 0.202 | 1.84 | 5/5 |
| `plane_barrier/sealed_4` | 0.094 | 0.279 | **2.98** | 5/5 |
| `plane_domain/U_shape` | 0.065 | 0.087 | 1.33 | 5/5 |
| `volume_barrier/sealed_8` | 0.299 | 0.381 | 1.27 | 5/5 |
| `sphere/open_ocean` | 0.315 | 0.591 | **1.87** | 5/5 |

Twelve layouts carry geometry and all twelve win five seeds out of five, under
both sampling protocols.  The three layouts that carry none tie to three
decimals and win at chance.  Nothing was selected on these numbers.

Against a coordinate network in two dimensions the margin is modest and the
parameter count is not: **288** parameters against **2 982** on `sealed_4`, for
a better reconstruction, because the operator supplies the spatial structure the
network has to learn.

## When it works, and when it does not

The condition is not that the operator class is right.  Adding a
divergence-free flow to the truth while leaving the learner untouched, a sealed
layout keeps a **6.5x** advantage at a hundred times the diffusive rate, because
nothing crosses a sealed wall however fast it is carried.  A layout with an
aperture loses the advantage completely (**0.95x**) at the same Peclet number,
because the flow simply carries the field through it.

> The geometry has to still constrain where the field can be.  A barrier the
> dynamics can route around has stopped being geometry.

Two external datasets sit outside that condition and show no advantage, as it
predicts: measured flow past a cylinder (RealPDEBench, particle-image
velocimetry) at `1.00--1.08x`, and 24 lid-driven cavities across aspect ratios
`0.2` to `5.0` (CFDBench) at `0.74--1.07x`.  A cylinder in an open channel and
an open box do not constrain a flow.

Known limitations, with mechanisms rather than excuses, are in
`docs/ITERATIONS.md`: a ceiling in three dimensions under sensor sampling, where
mode counts grow like `k^d` and the identifiability limit arrives first; and two
falsified design hypotheses and one failed attempt to learn the spectral cutoff,
kept rather than discarded.

## Frozen one-dimensional anchor

- Observation ratios: 2%, 5%, and 10%; the confirmation uses five fresh seeds
  and exactly 400 updates after freezing cutoff 8 and rank `(4,5,5)` on the
  earlier three selection seeds.
- The synthetic principal-angle phase boundary is now complemented by a
  variable-coefficient diffusion Green-response benchmark.  Its mismatch
  changes physical eigenfunctions and decay rates; every result stores the
  measured oracle projection residual.
- On the physical benchmark, 10% random and receiver-fiber masks both reach
  the predeclared 4/5 paired-win gate against a wide Neural Functional Tucker.
  Random NRMSE is `0.165±0.010` vs `0.207±0.054`; receiver-fiber is
  `0.217±0.052` vs `0.269±0.112`.
- The claim has a sharp boundary: source-fiber reaches only 3/5 wins at 10%,
  and 2% structured masks favor the neural baseline.  This is a conditional GO,
  not a claim of universal superiority under extreme sparsity.
- Basis cutoff has a real bias--variance tradeoff: reducing projection residual
  from 0.165 to 0.025 does not monotonically improve 2% reconstruction.
- Matched Tucker rank sweeps show that the 10% signal is not explained by one
  hand-picked core size, while 2% remains optimization/data limited.

The main claim is a measurable bias--variance phase boundary, not universal superiority over neural functional tensor models.

![Advantage against how much geometry there is to know](results/wall_sensors_summary_r7/advantage_vs_geometry_strength.png)

## Repository map

- `src/geoaware/`: `benchmark.py` builds every dataset in the main experiment;
  `grouped_operator_tucker.py` is the model (Tucker or CP, operator / table /
  neural factors, closed-form core posterior); `simplex_fem.py` is P1 finite
  elements on simplices of any dimension, flat or curved, dense or sparse;
  `als_baselines.py` wraps TensorLy; `operator_diagnostics.py` holds the
  pre-fit diagnostics — projection residuals, sparse eigenpairs, and the mode
  observability screen.  `irregular_fem.py`, `irregular_green_data.py` and
  `manifold_barrier_data.py` back the earlier rounds.  The frozen
  one-dimensional path — `tensor_bayes.py`, `tensor_data.py`,
  `operator_tucker_baselines.py`, `bases.py` — is unchanged.
- `experiments/`: fixed-budget phase-diagram runners and analysis.
- `results/`: immutable raw artifacts migrated from the hub.
- `docs/TECHNICAL_REPORT.md`: formulation, inference, baselines, dataset cards, and current evidence.
- `docs/PAPER_TECHNICAL_REPORT_ZH.md`: paper-facing Chinese Introduction/Method,
  frozen confirmation design, complete fresh-seed table, claim boundaries, exact
  operator provenance, the group-wise formulation, joint-vs-axis phase experiment,
  irregular-domain POC plan, and a new-session handoff.
- `docs/DATASETS_AND_RESOURCES.md`: local/shared datasets, official resources,
  operator-metadata requirements, leakage rules, and dataset implementation order.
- `docs/ITERATIONS.md`: repository-local research diary.
- `docs/SHARED_PROTOCOL.md`: shared audit discipline inherited from the hub.

## Quick check

```bash
PYTHONPATH=src python -m pytest -q
```

Large datasets and caches are not committed. Generated results must record seeds, masks, observation ratios, optimization budgets, and the exact generator or dataset version.
