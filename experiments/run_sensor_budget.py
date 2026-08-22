#!/usr/bin/env python3
"""How many sensors does the geometry buy you?

Reconstruction accuracy is the wrong currency for someone deciding whether to
use this.  What they have is a budget: a building, a number of instruments they
can afford to install, and an accuracy they need.  The question that answers is
how far the budget goes, and a geometry prior that reaches a target with a
quarter of the sensors is worth more than one that lowers an error bar.

Sweeping the number of sensors and reading off where each method crosses a
target turns the method's advantage into that number.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from geoaware.benchmark import build_family
from geoaware.grouped_operator_tucker import (GroupedOperatorTucker,
                                              grouped_indices)
from geoaware.masks import make_observation_split

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_geometry_main import build_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="floorplan/apartment,"
                                           "floorplan/lab_suite,"
                                           "plane_barrier/sealed_4")
    parser.add_argument("--ratios", default=".01,.02,.03,.05,.08,.12,.20")
    parser.add_argument("--models",
                        default="geometry_operator,blind_operator,neural_tucker")
    parser.add_argument("--seeds", default="201,202,203")
    parser.add_argument("--ranks", default="12,10,16")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranks = tuple(int(v) for v in args.ranks.split(","))

    rows = []
    for case in args.cases.split(","):
        family, layout = case.split("/")
        data = build_family(family, layout, n_scenarios=12, n_time=12)
        index = grouped_indices(data.shape, ((0,), (1,), (2,)))
        truth = data.values.flatten()
        for ratio in (float(v) for v in args.ratios.split(",")):
            for seed in (int(v) for v in args.seeds.split(",")):
                split = make_observation_split(data, ratio, "spatial_sensors",
                                               seed, sensor_axes=(2,))
                observed = torch.where(split.observed)[0]
                seen = split.observed.reshape(data.shape).reshape(
                    -1, data.shape[2]).any(0)
                generator = torch.Generator().manual_seed(seed + 4401)
                noisy = truth.clone()
                noisy[observed] += (torch.randn(len(observed), generator=generator)
                                    * args.noise * truth[observed].std())
                centre = float(noisy[observed].mean())
                scale = float(noisy[observed].std().clamp_min(1e-6))
                y = (noisy[observed] - centre) / scale
                held = split.held_out
                for name in args.models.split(","):
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    started = time.perf_counter()
                    model = GroupedOperatorTucker(
                        build_specs(name, data, ranks, 48, seen),
                        device=args.device)
                    model.fit(index[observed], y, steps=args.steps, seed=seed)
                    mean = model.predict(index).mean * scale + centre
                    error = mean[held] - truth[held]
                    rows.append({
                        "family": family, "layout": layout, "model": name,
                        "ratio": ratio, "seed": seed,
                        "n_sensors": int(seen.sum()),
                        "n_nodes": int(data.shape[2]),
                        "metrics": {"nrmse": float(
                            error.square().mean().sqrt()
                            / truth[held].std().clamp_min(1e-8))},
                        "elapsed_seconds": time.perf_counter() - started})
                    print(f"{case} sensors={int(seen.sum()):4d} ({ratio:.0%}) "
                          f"s{seed} {name:18s} "
                          f"NRMSE={rows[-1]['metrics']['nrmse']:.4f}", flush=True)
    (args.output / "results.json").write_text(json.dumps(
        {"arguments": vars(args), "results": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
