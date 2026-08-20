#!/usr/bin/env python3
"""The classical baselines, run separately and given their best shot.

CP-ALS and Tucker-HOOI use the NumPy backend, so they compete for CPU rather
than for the GPU the learned models occupy, and they are cheap enough to sweep
over ranks.  The rank is then selected on held-out error -- oracle knowledge the
baseline would not have -- which makes each reported number an upper bound on
what the classical method achieves on that cell.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from geoaware.als_baselines import best_of_ranks, cp_als, tucker_als
from geoaware.benchmark import FAMILIES, build_family
from geoaware.masks import make_observation_split

CP_RANKS = (4, 6, 10)
TUCKER_RANKS = ((3, 3, 4), (4, 4, 6), (6, 6, 8))
# Under sensor sampling every rank returns exactly the same thing -- nothing at
# all, since an unobserved node appears in no observed entry -- so the sweep is
# skipped there and one representative rank is reported.
SENSOR_CP_RANKS = (6,)
SENSOR_TUCKER_RANKS = ((4, 4, 6),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--ratios", default=".10")
    parser.add_argument("--masks", default="spatial_sensors,random")
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--n-scenarios", type=int, default=12)
    parser.add_argument("--n-time", type=int, default=12)
    # Omitted by default: each family carries the cutoff that equalizes its
    # geometry-aware approximation floor, fixed before any fitting.
    parser.add_argument("--basis-cutoff", type=int, default=None)
    parser.add_argument("--truth-modes", type=int, default=60)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--als-iters", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for family in args.families.split(","):
        for layout in FAMILIES[family].layouts:
            data = build_family(family, layout, n_scenarios=args.n_scenarios,
                                n_time=args.n_time,
                                basis_cutoff=args.basis_cutoff,
                                truth_modes=args.truth_modes)
            truth = data.values
            for mask in args.masks.split(","):
                for ratio in (float(v) for v in args.ratios.split(",")):
                    for seed in (int(v) for v in args.seeds.split(",")):
                        split = make_observation_split(data, ratio, mask, seed,
                                                       sensor_axes=(2,))
                        observed = torch.where(split.observed)[0]
                        generator = torch.Generator().manual_seed(seed + 4401)
                        noisy = truth.flatten().clone()
                        noisy[observed] += (
                            torch.randn(len(observed), generator=generator)
                            * args.noise * truth.flatten()[observed].std())
                        cube = noisy.reshape(truth.shape)
                        held = split.held_out
                        sweep = mask != "spatial_sensors"
                        for name, fit, candidates in (
                                ("cp_als",
                                 lambda r: cp_als(cube, split.observed, r,
                                                  seed=seed,
                                                  n_iter_max=args.als_iters),
                                 CP_RANKS if sweep else SENSOR_CP_RANKS),
                                ("tucker_als",
                                 lambda r: tucker_als(cube, split.observed, r,
                                                      seed=seed,
                                                      n_iter_max=args.als_iters),
                                 TUCKER_RANKS if sweep else SENSOR_TUCKER_RANKS)):
                            started = time.perf_counter()
                            score, rank, predicted = best_of_ranks(
                                fit, candidates, truth, held)
                            error = predicted[held] - truth.flatten()[held]
                            rows.append({
                                "family": family, "layout": layout,
                                "model": name, "mask": mask, "ratio": ratio,
                                "seed": seed, "n_observed": int(len(observed)),
                                "n_nodes": int(data.shape[2]),
                                "selected_rank": list(rank) if isinstance(rank, tuple) else rank,
                                "rank_selection": ("oracle: best held-out over a sweep"
                                                   if sweep else
                                                   "single rank: every rank is "
                                                   "equivalent under sensor sampling"),
                                "parameters": None,
                                "metrics": {
                                    "rmse": float(error.square().mean().sqrt()),
                                    "nrmse": score,
                                    "mae": float(error.abs().mean())},
                                "elapsed_seconds": time.perf_counter() - started})
                            print(f"{family}/{layout} {mask} {ratio} s{seed} "
                                  f"{name:11s} NRMSE={score:.3f} rank={rank} "
                                  f"({rows[-1]['elapsed_seconds']:.0f}s)", flush=True)
    (args.output / "results.json").write_text(json.dumps(
        {"arguments": vars(args), "results": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
