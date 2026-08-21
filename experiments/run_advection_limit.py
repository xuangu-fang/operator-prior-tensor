#!/usr/bin/env python3
"""Where the operator prior stops working, measured rather than argued.

Two real datasets in this repository show no advantage: measured flow past a
cylinder and lid-driven cavity flow.  Both are transport-dominated, and the
operator the learner assembles is a Laplacian.  The hypothesis is therefore that
the method needs the geometry *and* an operator class that still dominates the
dynamics -- and that is testable in a setting where everything else is held
fixed.

The barriers, the mesh, the learner and its basis are identical at every Peclet
number.  Only the physics generating the truth changes, and the velocity is zero
inside the barriers so the geometry stays exactly as real as it was.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from geoaware.benchmark import build_advected_barrier
from geoaware.grouped_operator_tucker import (GroupFactorSpec,
                                              GroupedOperatorTucker,
                                              grouped_indices)
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import (observable_modes,
                                           product_projection_residual)

SPECTRAL = {"geometry_operator": "geometry_operator",
            "blind_operator": "blind_operator", "permuted": "permuted"}


def build_specs(name, data, ranks, hidden, seen):
    matrices = data.operator_matrices
    specs = [GroupFactorSpec("table", ranks[0], data.shape[0], name="scenario"),
             GroupFactorSpec("table", ranks[1], data.shape[1], name="time")]
    if name in SPECTRAL:
        basis = matrices[f"{SPECTRAL[name]}_basis"]
        eigenvalues = matrices[f"{SPECTRAL[name]}_eigenvalues"]
        keep = observable_modes(basis, seen)
        basis, eigenvalues = basis[:, keep], eigenvalues[keep]
        specs.append(GroupFactorSpec("operator", min(ranks[2], basis.shape[1]),
                                     data.shape[2], basis=basis,
                                     eigenvalues=eigenvalues, name="node"))
    else:
        specs.append(GroupFactorSpec("neural", ranks[2], data.shape[2],
                                     coordinates=matrices["coordinates"],
                                     hidden=hidden, name="node"))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", default="chamber,sealed_4")
    parser.add_argument("--peclet", default="0,3,10,30,100")
    parser.add_argument("--models",
                        default="geometry_operator,blind_operator,neural_tucker")
    parser.add_argument("--mask", default="spatial_sensors")
    parser.add_argument("--ratio", type=float, default=.10)
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--ranks", default="4,4,6")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranks = tuple(int(v) for v in args.ranks.split(","))

    rows = []
    for layout in args.layouts.split(","):
        for peclet in (float(v) for v in args.peclet.split(",")):
            data = build_advected_barrier(layout, peclet)
            matrices = data.operator_matrices
            residuals = {n: product_projection_residual(
                            data.values, [None, None, matrices[f"{k}_basis"]])
                         for n, k in SPECTRAL.items()}
            print(f"[{layout} Pe={peclet:g}] " + " ".join(
                f"{k}={v:.4f}" for k, v in residuals.items()), flush=True)
            index = grouped_indices(data.shape, ((0,), (1,), (2,)))
            truth = data.values.flatten()
            for seed in (int(v) for v in args.seeds.split(",")):
                split = make_observation_split(data, args.ratio, args.mask,
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
                        build_specs(name, data, ranks, args.hidden, seen),
                        device=args.device)
                    model.fit(index[observed], y, steps=args.steps, seed=seed)
                    mean = model.predict(index).mean * scale + centre
                    error = mean[held] - truth[held]
                    rows.append({
                        "layout": layout, "peclet": peclet, "model": name,
                        "seed": seed, "mask": args.mask, "ratio": args.ratio,
                        "projection_residuals": residuals,
                        "metrics": {"nrmse": float(
                            error.square().mean().sqrt()
                            / truth[held].std().clamp_min(1e-8))},
                        "elapsed_seconds": time.perf_counter() - started})
                    print(f"{layout} Pe={peclet:g} s{seed} {name:18s} "
                          f"NRMSE={rows[-1]['metrics']['nrmse']:.3f}", flush=True)
    (args.output / "results.json").write_text(json.dumps(
        {"arguments": vars(args), "results": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
