#!/usr/bin/env python3
"""Main table: sparse Green-tensor completion on a domain with holes.

The setting is deliberately the simplest one that can carry the claim.  A
diffusion Green response ``Y(t, receiver-node, source-node)`` lives on a square
with circular holes; both spatial axes index the same mesh nodes.  Every model
shares the tensor, the mask, the noise realization, the observed-only
normalization, the Tucker ranks and the optimization budget.  The only thing
that changes is what the spatial factor is allowed to be:

* ``fem_operator``     — eigenbasis of the mesh operator, holes and all (ours)
* ``topology_erased``  — same nodes, triangulated across the holes
* ``bounding_box``     — separable cosines of the enclosing square
* ``neural_coords``    — an MLP on node coordinates, geometry-blind, wider
* ``discrete_table``   — a free factor table: ordinary tensor completion
* ``laplacian_table``  — free table with the mesh operator as a smoothness
                         penalty, so "any smoothness would do" can be ruled out
* ``permuted``         — the geometry-aware basis with its rows shuffled

The learner is given the mesh and the boundary but never the material
coefficient, so this is a geometry prior, not an exact-physics prior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from geoaware.grouped_operator_tucker import (GroupFactorSpec,
                                              GroupedOperatorTucker,
                                              grouped_indices,
                                              sobolev_penalty_operator)
from geoaware.irregular_fem import (L_SHAPE, U_SHAPE, UNIT_SQUARE, Hole)
from geoaware.irregular_green_data import irregular_green_tensor
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import (generalized_eigenpairs,
                                           product_projection_residual)

MODELS = ("fem_operator", "topology_erased", "bounding_box", "neural_coords",
          "discrete_table", "laplacian_table", "permuted")
BASIS_KEY = {"fem_operator": "fem_correct", "topology_erased": "topology_erased",
             "bounding_box": "bounding_box_product", "permuted": "permuted"}

# Deliberately plain shapes: a square as the geometry-free control, then one
# obstacle, two obstacles, and two non-convex domains whose boundary alone
# changes how the field can travel.
LAYOUTS = {
    "square": (UNIT_SQUARE, ()),
    "center_hole": (UNIT_SQUARE, (Hole((.5, .5), .20),)),
    "two_holes": (UNIT_SQUARE, (Hole((.32, .62), .15), Hole((.68, .33), .13))),
    "L_shape": (L_SHAPE, ()),
    "U_shape": (U_SHAPE, ()),
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_specs(name, data, ranks, hidden, power):
    matrices = data.operator_matrices
    nt, nr, ns = data.shape
    sources = matrices["source_nodes"]
    specs = [GroupFactorSpec("operator", ranks[0], nt,
                             basis=matrices["time_basis"],
                             eigenvalues=matrices["time_eigenvalues"], name="time")]
    if name in BASIS_KEY:
        key = BASIS_KEY[name]
        basis = matrices[f"{key}_basis"]
        eigenvalues = matrices[f"{key}_eigenvalues"]
        specs.append(GroupFactorSpec("operator", ranks[1], nr, basis=basis,
                                     eigenvalues=eigenvalues, name="receiver"))
        specs.append(GroupFactorSpec("operator", ranks[2], ns,
                                     basis=basis[sources], eigenvalues=eigenvalues,
                                     name="source"))
    elif name == "neural_coords":
        coordinates = matrices["coordinates"]
        specs.append(GroupFactorSpec("neural", ranks[1], nr, coordinates=coordinates,
                                     hidden=hidden, name="receiver"))
        specs.append(GroupFactorSpec("neural", ranks[2], ns,
                                     coordinates=coordinates[sources], hidden=hidden,
                                     name="source"))
    elif name in ("discrete_table", "laplacian_table"):
        penalty = None
        source_penalty = None
        if name == "laplacian_table":
            stiffness, mass = matrices["nominal_stiffness"], matrices["mass"]
            reference, _ = generalized_eigenpairs(stiffness, mass, 2)
            penalty = sobolev_penalty_operator(stiffness, mass, power, reference)
            source_penalty = penalty[sources][:, sources]
        specs.append(GroupFactorSpec("table", ranks[1], nr, penalty_operator=penalty,
                                     name="receiver"))
        specs.append(GroupFactorSpec("table", ranks[2], ns,
                                     penalty_operator=source_penalty, name="source"))
    else:
        raise ValueError(name)
    return specs


def boundary_band(data, width: float = .12) -> torch.Tensor:
    """Entries whose receiver node sits within ``width`` of a hole rim.

    Reported separately because that is where a geometry-blind basis has to
    smooth across an obstacle, and where the claim should be visible if it is
    true anywhere.
    """
    coordinates = data.operator_matrices["coordinates"].double()
    holes = data.metadata["holes"]
    near = torch.zeros(len(coordinates), dtype=torch.bool)
    for hole in holes:
        center = torch.tensor(hole["center"], dtype=torch.float64)
        distance = (coordinates - center).norm(dim=1) - hole["radius"]
        near |= distance < width
    return near


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", default="square,center_hole,two_holes,L_shape,U_shape")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ratios", default=".02,.05,.10")
    parser.add_argument("--masks", default="random,receiver_fibers")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--resolution", type=int, default=18)
    parser.add_argument("--n-time", type=int, default=16)
    parser.add_argument("--n-sources", type=int, default=20)
    parser.add_argument("--basis-cutoff", type=int, default=32)
    parser.add_argument("--truth-modes", type=int, default=60)
    parser.add_argument("--contrast", type=float, default=.3)
    parser.add_argument("--hole-condition", choices=["neumann", "dirichlet"],
                        default="neumann",
                        help="insulating obstacles by default; Dirichlet rims add "
                             "material-dependent boundary layers and are reported "
                             "as a separate ablation, never merged with the main table")
    parser.add_argument("--time-span", default=".15,3.0")
    parser.add_argument("--ranks", default="4,6,6")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--reg", type=float, default=.002)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ranks = tuple(int(v) for v in args.ranks.split(","))
    rows, geometries = [], {}
    for layout in args.layouts.split(","):
        polygon, holes = LAYOUTS[layout]
        data = irregular_green_tensor(
            holes, polygon=polygon, resolution=args.resolution, n_time=args.n_time,
            n_sources=args.n_sources, basis_cutoff=args.basis_cutoff,
            truth_modes=args.truth_modes, contrast=args.contrast,
            time_span=tuple(float(v) for v in args.time_span.split(",")),
            hole_condition=args.hole_condition)
        matrices = data.operator_matrices
        residuals = {}
        for name, key in BASIS_KEY.items():
            basis = matrices[f"{key}_basis"]
            residuals[name] = product_projection_residual(
                data.values, [matrices["time_basis"], basis,
                              basis[matrices["source_nodes"]]])
        near = boundary_band(data)
        geometries[layout] = {"name": data.name, "shape": list(data.shape),
                              "metadata": data.metadata,
                              "projection_residuals": residuals,
                              "boundary_band_nodes": int(near.sum())}
        index = grouped_indices(data.shape, ((0,), (1,), (2,)))
        truth = data.values.flatten()
        near_entry = near[None, :, None].expand(data.shape).reshape(-1)

        for mask in args.masks.split(","):
            for ratio in (float(v) for v in args.ratios.split(",")):
                for seed in (int(v) for v in args.seeds.split(",")):
                    split = make_observation_split(data, ratio, mask, seed)
                    observed = torch.where(split.observed)[0]
                    generator = torch.Generator().manual_seed(seed + 4401)
                    noisy = truth.clone()
                    noisy[observed] += (torch.randn(len(observed), generator=generator)
                                        * args.noise * truth[observed].std())
                    center = float(noisy[observed].mean())
                    scale = float(noisy[observed].std().clamp_min(1e-6))
                    y = (noisy[observed] - center) / scale

                    for name in args.models.split(","):
                        seed_all(seed)
                        started = time.perf_counter()
                        model = GroupedOperatorTucker(
                            build_specs(name, data, ranks, args.hidden, args.power),
                            power=args.power, device=args.device)
                        model.fit(index[observed], y, steps=args.steps,
                                  reg_weight=args.reg, seed=seed)
                        mean = model.predict(index).mean * scale + center
                        held = split.held_out
                        error = mean[held] - truth[held]
                        rmse = float(error.square().mean().sqrt())
                        band = held & near_entry
                        band_nrmse = None
                        if bool(band.any()):
                            band_error = mean[band] - truth[band]
                            band_nrmse = float(
                                band_error.square().mean().sqrt()
                                / truth[band].std().clamp_min(1e-8))
                        rows.append({
                            "layout": layout, "model": name, "mask": mask,
                            "ratio": ratio, "ratio_actual": split.ratio_actual,
                            "n_observed": int(len(observed)), "seed": seed,
                            "metrics": {
                                "rmse": rmse,
                                "nrmse": float(rmse / truth[held].std().clamp_min(1e-8)),
                                "mae": float(error.abs().mean()),
                                "boundary_band_nrmse": band_nrmse},
                            "observed_fit_nrmse": float(
                                (mean[observed] - noisy[observed]).square().mean().sqrt()
                                / noisy[observed].std().clamp_min(1e-8)),
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "core_size": math.prod(model.ranks),
                            "projection_residual": residuals.get(name),
                            "elapsed_seconds": time.perf_counter() - started})
                        print(f"{layout} {mask} {ratio:g} s{seed} {name:16s} "
                              f"NRMSE={rows[-1]['metrics']['nrmse']:.3f} "
                              f"band={band_nrmse if band_nrmse is None else round(band_nrmse,3)} "
                              f"params={rows[-1]['parameters']}", flush=True)

    (args.output / "results.json").write_text(json.dumps(
        {"geometries": geometries, "arguments": vars(args), "results": rows},
        indent=2, default=str))


if __name__ == "__main__":
    main()
