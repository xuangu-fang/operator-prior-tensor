#!/usr/bin/env python3
"""The same benchmark in a volume and on a closed surface.

Two settings share one runner because they share one design: a fixed mesh, a
tensor ``Y(scenario, time, node)``, and controls that differ from the proposed
model in exactly one respect.

``box``     tetrahedra filling a cube, divided by partitions with apertures.
``sphere``  triangles on a geodesic sphere carrying the linearized shallow-water
            equation, where the geometry under test is the manifold itself.

On the sphere the interesting control is ``lat_lon``: a separable cosine basis
of the latitude-longitude rectangle, which is what spherical data is usually
factorized with.  Its failure is structural rather than a matter of resolution,
so it is the control that cannot be answered by giving the baseline more rank.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from geoaware.grouped_operator_tucker import (GroupFactorSpec,
                                              GroupedOperatorTucker,
                                              grouped_indices,
                                              sobolev_penalty_operator)
from geoaware.manifold_barrier_data import (BOX_LAYOUTS, SPHERE_LAYOUTS,
                                            barrier_field_tensor)
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import (generalized_eigenpairs,
                                           product_projection_residual)

MODELS = ("fem_operator", "topology_erased", "lat_lon", "bounding_box",
          "neural_coords", "neural_matched", "discrete_table", "laplacian_geo",
          "laplacian_blind", "permuted")
BASIS_KEY = {"fem_operator": "fem_correct", "topology_erased": "topology_erased",
             "lat_lon": "lat_lon_product", "bounding_box": "bounding_box_product",
             "permuted": "permuted"}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_specs(name, data, ranks, hidden, power, matched_hidden=10):
    """Same decoder everywhere; only the spatial factor's function space moves."""
    matrices = data.operator_matrices
    specs = [GroupFactorSpec("table", ranks[0], data.shape[0], name="scenario"),
             GroupFactorSpec("operator", ranks[1], data.shape[1],
                             basis=matrices["time_basis"],
                             eigenvalues=matrices["time_eigenvalues"], name="time")]
    size = data.shape[2]
    if name in BASIS_KEY:
        key = BASIS_KEY[name]
        specs.append(GroupFactorSpec("operator", ranks[2], size,
                                     basis=matrices[f"{key}_basis"],
                                     eigenvalues=matrices[f"{key}_eigenvalues"],
                                     name="node"))
    elif name in ("neural_coords", "neural_matched"):
        width = hidden if name == "neural_coords" else matched_hidden
        specs.append(GroupFactorSpec("neural", ranks[2], size,
                                     coordinates=matrices["coordinates"],
                                     hidden=width, name="node"))
    elif name in ("discrete_table", "laplacian_geo", "laplacian_blind"):
        penalty = None
        if name.startswith("laplacian"):
            stiffness = matrices["nominal_stiffness" if name == "laplacian_geo"
                                 else "blind_stiffness"]
            reference, _ = generalized_eigenpairs(stiffness, matrices["mass"], 2)
            penalty = sobolev_penalty_operator(stiffness, matrices["mass"],
                                               power, reference)
        specs.append(GroupFactorSpec("table", ranks[2], size,
                                     penalty_operator=penalty, name="node"))
    else:
        raise ValueError(name)
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", choices=["box", "sphere"], required=True)
    parser.add_argument("--layouts", default="")
    parser.add_argument("--dynamics", choices=["diffusion", "wave"],
                        default="diffusion")
    parser.add_argument("--models", default="")
    parser.add_argument("--ratios", default=".02,.05,.10")
    parser.add_argument("--masks", default="spatial_sensors,random")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--n-scenarios", type=int, default=20)
    parser.add_argument("--n-time", type=int, default=16)
    parser.add_argument("--basis-cutoff", type=int, default=16)
    parser.add_argument("--truth-modes", type=int, default=60)
    parser.add_argument("--contrast", type=float, default=.3)
    parser.add_argument("--time-span", default=".15,3.0")
    parser.add_argument("--ranks", default="4,4,6")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--matched-hidden", type=int, default=10)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--reg", type=float, default=.002)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    catalogue = BOX_LAYOUTS if args.geometry == "box" else SPHERE_LAYOUTS
    layouts = (args.layouts.split(",") if args.layouts else list(catalogue))
    models = (args.models.split(",") if args.models else
              [m for m in MODELS if args.geometry == "sphere" or m != "lat_lon"])
    ranks = tuple(int(v) for v in args.ranks.split(","))

    rows, geometries = [], {}
    for layout in layouts:
        data = barrier_field_tensor(
            catalogue[layout], geometry=args.geometry, resolution=args.resolution,
            subdivisions=args.subdivisions, n_scenarios=args.n_scenarios,
            n_time=args.n_time, basis_cutoff=args.basis_cutoff,
            truth_modes=args.truth_modes, contrast=args.contrast,
            dynamics=args.dynamics,
            time_span=tuple(float(v) for v in args.time_span.split(",")))
        matrices = data.operator_matrices
        residuals = {name: product_projection_residual(
                        data.values, [None, None, matrices[f"{key}_basis"]])
                     for name, key in BASIS_KEY.items()
                     if f"{key}_basis" in matrices}
        geometries[layout] = {"name": data.name, "shape": list(data.shape),
                              "metadata": data.metadata,
                              "projection_residuals": residuals}
        print(f"[{layout}] nodes={data.shape[2]} residuals="
              + " ".join(f"{k}={v:.3f}" for k, v in residuals.items()), flush=True)

        index = grouped_indices(data.shape, ((0,), (1,), (2,)))
        truth = data.values.flatten()
        for mask in args.masks.split(","):
            for ratio in (float(v) for v in args.ratios.split(",")):
                for seed in (int(v) for v in args.seeds.split(",")):
                    split = make_observation_split(data, ratio, mask, seed,
                                                   sensor_axes=(2,))
                    observed = torch.where(split.observed)[0]
                    generator = torch.Generator().manual_seed(seed + 4401)
                    noisy = truth.clone()
                    noisy[observed] += (torch.randn(len(observed), generator=generator)
                                        * args.noise * truth[observed].std())
                    center = float(noisy[observed].mean())
                    scale = float(noisy[observed].std().clamp_min(1e-6))
                    y = (noisy[observed] - center) / scale

                    for name in models:
                        seed_all(seed)
                        started = time.perf_counter()
                        model = GroupedOperatorTucker(
                            build_specs(name, data, ranks, args.hidden,
                                        args.power, args.matched_hidden),
                            power=args.power, device=args.device)
                        model.fit(index[observed], y, steps=args.steps,
                                  reg_weight=args.reg, seed=seed)
                        mean = model.predict(index).mean * scale + center
                        held = split.held_out
                        error = mean[held] - truth[held]
                        rmse = float(error.square().mean().sqrt())
                        rows.append({
                            "geometry": args.geometry, "dynamics": args.dynamics,
                            "layout": layout, "model": name, "mask": mask,
                            "ratio": ratio, "seed": seed,
                            "n_observed": int(len(observed)),
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "metrics": {
                                "rmse": rmse,
                                "nrmse": float(rmse / truth[held].std().clamp_min(1e-8)),
                                "mae": float(error.abs().mean())},
                            "elapsed_seconds": time.perf_counter() - started})
                        print(f"{layout} {mask} {ratio} s{seed} {name:16s} "
                              f"NRMSE={rows[-1]['metrics']['nrmse']:.3f} "
                              f"params={rows[-1]['parameters']}", flush=True)

    (args.output / "results.json").write_text(json.dumps(
        {"arguments": vars(args), "geometries": geometries, "results": rows},
        indent=2, default=str))


if __name__ == "__main__":
    main()
