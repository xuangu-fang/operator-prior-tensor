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
from geoaware.irregular_green_data import (WALL_LAYOUTS, irregular_field_tensor,
                                           irregular_green_tensor,
                                           wall_field_tensor)
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import (generalized_eigenpairs,
                                           product_projection_residual)

MODELS = ("fem_operator", "topology_erased", "bounding_box", "neural_coords",
          "neural_matched", "discrete_table", "laplacian_geo", "laplacian_blind",
          "permuted")
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


def build_specs(name, data, ranks, hidden, power, setting, matched_hidden=10):
    """Same decoder everywhere; only the spatial factor's function space moves.

    For the Green setting geometry enters two modes (receiver and source nodes);
    for the field setting it enters one, and the scenario mode is a free table
    because an enumeration of initial conditions carries no operator.
    """
    matrices = data.operator_matrices
    spatial_axes = (1, 2) if setting == "green" else (2,)
    sources = matrices.get("source_nodes")
    specs = []
    for axis, size in enumerate(data.shape):
        if axis not in spatial_axes:
            if setting == "green" or axis == 1:      # the time mode
                specs.append(GroupFactorSpec(
                    "operator", ranks[axis], size, basis=matrices["time_basis"],
                    eigenvalues=matrices["time_eigenvalues"], name="time"))
            else:                                     # scenario enumeration
                specs.append(GroupFactorSpec("table", ranks[axis], size,
                                             name="scenario"))
            continue
        rows = sources if (setting == "green" and axis == 2) else None
        label = "source" if rows is not None else "node"
        if name in BASIS_KEY:
            key = BASIS_KEY[name]
            basis = matrices[f"{key}_basis"]
            specs.append(GroupFactorSpec(
                "operator", ranks[axis], size,
                basis=basis if rows is None else basis[rows],
                eigenvalues=matrices[f"{key}_eigenvalues"], name=label))
        elif name in ("neural_coords", "neural_matched"):
            # Two capacities on purpose: a wide network answers "can a strong
            # geometry-blind regressor learn the field anyway", a width matched
            # to the operator model's parameter count answers "is any advantage
            # just extra capacity".
            width = hidden if name == "neural_coords" else matched_hidden
            coordinates = matrices["coordinates"]
            specs.append(GroupFactorSpec(
                "neural", ranks[axis], size,
                coordinates=coordinates if rows is None else coordinates[rows],
                hidden=width, name=label))
        elif name in ("discrete_table", "laplacian_geo", "laplacian_blind"):
            penalty = None
            if name.startswith("laplacian"):
                mass = matrices["mass"]
                stiffness = matrices["nominal_stiffness" if name == "laplacian_geo"
                                     else "blind_stiffness"]
                reference, _ = generalized_eigenpairs(stiffness, mass, 2)
                penalty = sobolev_penalty_operator(stiffness, mass, power, reference)
                if rows is not None:
                    penalty = penalty[rows][:, rows]
            specs.append(GroupFactorSpec("table", ranks[axis], size,
                                         penalty_operator=penalty, name=label))
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
    holes = data.metadata.get("holes", [])
    if not holes:
        return torch.zeros(len(coordinates), dtype=torch.bool)
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
    parser.add_argument("--setting", choices=["green", "field", "wall"],
                        default="green")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ratios", default=".02,.05,.10")
    parser.add_argument("--masks", default="random,receiver_fibers")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--resolution", type=int, default=18)
    parser.add_argument("--n-time", type=int, default=16)
    parser.add_argument("--n-sources", type=int, default=20)
    parser.add_argument("--basis-cutoff", type=int, default=32)
    parser.add_argument("--truth-modes", type=int, default=60)
    # Above one, the truth is solved on an independently seeded mesh this many
    # times finer and interpolated onto the learner's nodes, so the learner's
    # operator is no longer the object that generated the data.
    parser.add_argument("--truth-refinement", type=int, default=1)
    parser.add_argument("--contrast", type=float, default=.3)
    parser.add_argument("--hole-condition", choices=["neumann", "dirichlet"],
                        default="neumann",
                        help="insulating obstacles by default; Dirichlet rims add "
                             "material-dependent boundary layers and are reported "
                             "as a separate ablation, never merged with the main table")
    parser.add_argument("--time-span", default=".15,3.0")
    parser.add_argument("--ranks", default="4,6,6")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--matched-hidden", type=int, default=10)
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
        # wall layouts live in WALL_LAYOUTS; polygonal ones in LAYOUTS
        if args.setting == "wall":
            data = wall_field_tensor(
                WALL_LAYOUTS[layout], resolution=args.resolution,
                n_scenarios=args.n_sources, n_time=args.n_time,
                basis_cutoff=args.basis_cutoff, truth_modes=args.truth_modes,
                contrast=args.contrast,
                truth_refinement=args.truth_refinement,
                time_span=tuple(float(v) for v in args.time_span.split(",")))
        else:
            polygon, holes = LAYOUTS[layout]
            common = dict(
                polygon=polygon, resolution=args.resolution,
                n_time=args.n_time, basis_cutoff=args.basis_cutoff,
                truth_modes=args.truth_modes, contrast=args.contrast,
                time_span=tuple(float(v) for v in args.time_span.split(",")),
                hole_condition=args.hole_condition)
            data = (irregular_green_tensor(holes, n_sources=args.n_sources, **common)
                    if args.setting == "green" else
                    irregular_field_tensor(holes, n_scenarios=args.n_sources,
                                           truth_refinement=args.truth_refinement,
                                           **common))
        matrices = data.operator_matrices
        residuals = {}
        for name, key in BASIS_KEY.items():
            basis = matrices[f"{key}_basis"]
            projectors = ([matrices["time_basis"], basis,
                           basis[matrices["source_nodes"]]]
                          if args.setting == "green" else [None, None, basis])
            residuals[name] = product_projection_residual(data.values, projectors)
        node_axis = 1 if args.setting == "green" else 2
        spatial_setting = "green" if args.setting == "green" else "field"
        near = boundary_band(data)
        geometries[layout] = {"name": data.name, "shape": list(data.shape),
                              "metadata": data.metadata,
                              "projection_residuals": residuals,
                              "boundary_band_nodes": int(near.sum())}
        index = grouped_indices(data.shape, ((0,), (1,), (2,)))
        truth = data.values.flatten()
        shape_view = [1, 1, 1]
        shape_view[node_axis] = len(near)
        near_entry = near.reshape(shape_view).expand(data.shape).reshape(-1)

        for mask in args.masks.split(","):
            for ratio in (float(v) for v in args.ratios.split(",")):
                for seed in (int(v) for v in args.seeds.split(",")):
                    # A spatial sensor is a mesh node observed for its whole
                    # trajectory.  The node axis differs between settings, so it
                    # is passed explicitly rather than left to a default.
                    split = make_observation_split(
                        data, ratio, mask, seed, sensor_axes=(node_axis,))
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
                            build_specs(name, data, ranks, args.hidden,
                                        args.power, spatial_setting,
                                        args.matched_hidden),
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
