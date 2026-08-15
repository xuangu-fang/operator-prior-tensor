# Operator-Prior Tensor

Operator-informed Bayesian CP/Tucker for recovering continuous physical tensors from at most 10% observations.

This repository is Track 1 of the [Physics-Informed Tensor Learning Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor). It is intentionally scoped to one question:

> When does an operator-defined factor space improve sample efficiency, and when does operator mismatch create irreducible bias?

## Current evidence

- Observation ratios: 2%, 5%, and 10%; three seeds; 400--500 updates.
- The synthetic principal-angle phase boundary is now complemented by a
  variable-coefficient diffusion Green-response benchmark.  Its mismatch
  changes physical eigenfunctions and decay rates; every result stores the
  measured oracle projection residual.
- On the physical benchmark, Operator Tucker has a stable positive signal at
  10% observations.  At 2%--5%, differences from Neural Functional Tucker are
  small or high-variance; this is not yet a publication-ready win.
- Basis cutoff has a real bias--variance tradeoff: reducing projection residual
  from 0.165 to 0.025 does not monotonically improve 2% reconstruction.
- Matched Tucker rank sweeps show that the 10% signal is not explained by one
  hand-picked core size, while 2% remains optimization/data limited.

The main claim is a measurable bias--variance phase boundary, not universal superiority over neural functional tensor models.

![Physical operator perturbation](results/diffusion_contrast_summary_r1/operator_advantage_vs_contrast.png)

## Repository map

- `src/geoaware/`: migrated implementation snapshot; active Track-1 modules are `tensor_bayes.py`, `tensor_data.py`, `operator_tucker_baselines.py`, `bases.py`, and `masks.py`.
- `experiments/`: fixed-budget phase-diagram runners and analysis.
- `results/`: immutable raw artifacts migrated from the hub.
- `docs/TECHNICAL_REPORT.md`: formulation, inference, baselines, dataset cards, and current evidence.
- `docs/ITERATIONS.md`: repository-local research diary.
- `docs/SHARED_PROTOCOL.md`: shared audit discipline inherited from the hub.

## Quick check

```bash
PYTHONPATH=src python -m pytest -q
```

Large datasets and caches are not committed. Generated results must record seeds, masks, observation ratios, optimization budgets, and the exact generator or dataset version.
