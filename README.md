# Operator-Prior Tensor

Operator-informed Bayesian CP/Tucker for recovering continuous physical tensors from at most 10% observations.

This repository is Track 1 of the [Physics-Informed Tensor Learning Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor). It is intentionally scoped to one question:

> When does an operator-defined factor space improve sample efficiency, and when does operator mismatch create irreducible bias?

## Current evidence

- Observation ratios: 2%, 5%, and 10%; three seeds; 500 updates.
- The mismatch axis is the oracle relative projection residual of the truth outside the learner's three-mode operator product space.
- Operator Tucker wins consistently for mismatch at most 0.30.
- The ordering reverses around 0.30--0.45 at 2% observations and around 0.45--0.60 at 5%--10%.

The main claim is a measurable bias--variance phase boundary, not universal superiority over neural functional tensor models.

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

