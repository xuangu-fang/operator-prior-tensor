# Operator-Prior Tensor

Operator-informed Bayesian CP/Tucker for recovering continuous physical tensors from at most 10% observations.

This repository is Track 1 of the [Physics-Informed Tensor Learning Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor). It is intentionally scoped to one question:

> When does an operator-defined factor space improve sample efficiency, and when does operator mismatch create irreducible bias?

## Current evidence

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

![Physical operator perturbation](results/diffusion_contrast_summary_r1/operator_advantage_vs_contrast.png)

## Repository map

- `src/geoaware/`: migrated implementation snapshot; active Track-1 modules are `tensor_bayes.py`, `tensor_data.py`, `operator_tucker_baselines.py`, `bases.py`, and `masks.py`.
- `experiments/`: fixed-budget phase-diagram runners and analysis.
- `results/`: immutable raw artifacts migrated from the hub.
- `docs/TECHNICAL_REPORT.md`: formulation, inference, baselines, dataset cards, and current evidence.
- `docs/PAPER_TECHNICAL_REPORT_ZH.md`: paper-facing Chinese Introduction/Method,
  frozen confirmation design, complete fresh-seed table, and claim boundaries.
- `docs/ITERATIONS.md`: repository-local research diary.
- `docs/SHARED_PROTOCOL.md`: shared audit discipline inherited from the hub.

## Quick check

```bash
PYTHONPATH=src python -m pytest -q
```

Large datasets and caches are not committed. Generated results must record seeds, masks, observation ratios, optimization budgets, and the exact generator or dataset version.
