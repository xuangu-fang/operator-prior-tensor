#!/usr/bin/env python3
"""POC-B: does a joint operator on the coordinate group ``(x, y)`` pay off?

The predeclared question is not "is the operator prior good".  It is whether the
gap between a *joint* group operator and its *per-axis* approximation tracks the
measured separability of the discrete operator.  Everything else — mesh, PDE
family, boundary condition, ranks, budget, noise, mask, seeds — is held fixed
while the coupling ``eta`` is swept.

Two per-axis variants are run because they answer different questions and it
would be misleading to show only one:

``axis_product``
    Same grouped decoder and same latent dimension as the joint model, but the
    spatial factor lives in the per-axis product basis.  This isolates the
    subspace disagreement ``epsilon_sub`` with nothing else changed.
``axis_split``
    x and y become separate Tucker modes.  This is the genuinely cheaper
    approximation people would actually deploy, and it additionally imposes a
    rank-one separability constraint on each spatial factor.

Controls: a misspecified coupling (same family, still a plausible operator) and
a node permutation (destroys index-operator alignment).  They must be reported
separately; the first is not expected to be destructive.
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
                                              group_coordinates,
                                              grouped_indices,
                                              sobolev_penalty_operator)
from geoaware.joint_diffusion_2d import joint_diffusion_2d_tensor
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import (generalized_eigenpairs,
                                           product_projection_residual)

MODELS = ("joint_operator", "axis_product", "axis_split", "wrong_joint",
          "permuted_joint", "grouped_neural", "laplacian_table")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_specs(name: str, data, ranks: tuple[int, int, int],
                axis_ranks: tuple[int, int], neural_hidden: int,
                power: float, scenario_kind: str = "table") -> tuple[list[GroupFactorSpec], tuple[tuple[int, ...], ...]]:
    """Every variant differs only in how the spatial group is represented."""
    matrices = data.operator_matrices
    nt, nx, ny, ns = data.shape
    # Every variant shares one time basis, taken from the joint semigroup, so
    # that the only thing under comparison is how the spatial group is
    # represented.  Giving each variant its own time basis would confound the
    # spatial question with a temporal one.
    time_spec = GroupFactorSpec("operator", ranks[0], nt,
                                basis=matrices["time_basis"],
                                eigenvalues=matrices["time_eigenvalues"], name="time")
    # The scenario index enumerates unrelated initial conditions.  It has no
    # operator and no meaningful ordering, so a free table is both the honest
    # prior and the cheap one; an MLP here would impose smoothness across an
    # arbitrary enumeration and would also dominate the parameter budget.
    scenario_spec = (
        GroupFactorSpec("neural", ranks[2], ns,
                        coordinates=group_coordinates(data.shape, (3,)),
                        hidden=neural_hidden, name="scenario")
        if scenario_kind == "neural" else
        GroupFactorSpec("table", ranks[2], ns, name="scenario"))
    groups = ((0,), (1, 2), (3,))

    if name == "axis_split":
        specs = [time_spec,
                 GroupFactorSpec("operator", axis_ranks[0], nx,
                                 basis=matrices["axis_bases"][0],
                                 eigenvalues=matrices["axis_eigenvalues"][0], name="x"),
                 GroupFactorSpec("operator", axis_ranks[1], ny,
                                 basis=matrices["axis_bases"][1],
                                 eigenvalues=matrices["axis_eigenvalues"][1], name="y"),
                 scenario_spec]
        return specs, ((0,), (1,), (2,), (3,))

    spatial_size = nx * ny
    if name == "grouped_neural":
        spatial = GroupFactorSpec("neural", ranks[1], spatial_size,
                                  coordinates=group_coordinates(data.shape, (1, 2)),
                                  hidden=neural_hidden, name="space")
    elif name == "laplacian_table":
        reference, _ = generalized_eigenpairs(matrices["joint_stiffness"],
                                              matrices["joint_mass"], 2)
        spatial = GroupFactorSpec(
            "table", ranks[1], spatial_size,
            penalty_operator=sobolev_penalty_operator(
                matrices["joint_stiffness"], matrices["joint_mass"], power, reference),
            name="space")
    else:
        key = {"joint_operator": "joint_basis", "axis_product": "product_basis",
               "wrong_joint": "wrong_joint_basis",
               "permuted_joint": "permuted_joint_basis"}[name]
        eigen_key = key.replace("_basis", "_eigenvalues")
        spatial = GroupFactorSpec("operator", ranks[1], spatial_size,
                                  basis=matrices[key], eigenvalues=matrices[eigen_key],
                                  name="space")
    return [time_spec, spatial, scenario_spec], groups


def basis_construction_cost(data) -> dict:
    """Wall time of the eigensolves each variant actually needs.

    The joint operator is a single ``N_x N_y`` eigenproblem; the per-axis
    approximation solves two small ones.  Reporting this keeps "the joint basis
    is better" from hiding "and it costs much more to build".
    """
    matrices = data.operator_matrices
    cutoff = data.metadata["joint_cutoff"]
    started = time.perf_counter()
    generalized_eigenpairs(matrices["joint_stiffness"], matrices["joint_mass"], cutoff)
    joint = time.perf_counter() - started
    started = time.perf_counter()
    for stiffness, mass in zip(matrices["axis_stiffness"], matrices["axis_mass"]):
        generalized_eigenpairs(stiffness, mass, data.metadata["axis_cutoff"])
    axis = time.perf_counter() - started
    return {"joint_eigensolve_seconds": joint, "axis_eigensolve_seconds": axis,
            "joint_operator_size": int(matrices["joint_stiffness"].shape[0]),
            "axis_operator_sizes": [int(m.shape[0]) for m in matrices["axis_stiffness"]]}


def projection_residuals(data) -> dict:
    """Bias floor of each spatial basis, measured on the noiseless tensor.

    This is an oracle diagnostic reported for interpretation only; no model
    selects anything with it.
    """
    grouped = data.grouped_values()
    matrices = data.operator_matrices
    out = {}
    for label, key in (("joint", "joint_basis"), ("axis_product", "product_basis"),
                       ("wrong_joint", "wrong_joint_basis"),
                       ("permuted_joint", "permuted_joint_basis")):
        out[label] = product_projection_residual(
            grouped, [matrices["time_basis"], matrices[key], None])
    nx, ny = data.shape[1], data.shape[2]
    split = data.values.permute(0, 1, 2, 3)
    out["axis_split"] = product_projection_residual(
        split, [matrices["time_basis"], matrices["axis_bases"][0],
                matrices["axis_bases"][1], None])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--couplings", default="0.0,0.3,0.6,0.9")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ratios", default=".02,.05,.10")
    parser.add_argument("--masks", default="random,spatial_sensors")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--shape", default="16,16,16,12")
    parser.add_argument("--ranks", default="4,9,4", help="time, space group, scenario")
    parser.add_argument("--axis-ranks", default="3,3",
                        help="x and y ranks of the split variant; their product "
                             "should match the space-group rank")
    parser.add_argument("--joint-cutoff", type=int, default=16)
    parser.add_argument("--axis-cutoff", type=int, default=8)
    parser.add_argument("--time-cutoff", type=int, default=8)
    parser.add_argument("--wrong-coupling", type=float, default=.9)
    parser.add_argument("--learner-coupling", type=float, default=None,
                        help="omit for the exact-operator tier")
    parser.add_argument("--neural-hidden", type=int, default=48)
    parser.add_argument("--scenario-kind", choices=["table", "neural"], default="table")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--reg", type=float, default=.002)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    shape = tuple(int(v) for v in args.shape.split(","))
    ranks = tuple(int(v) for v in args.ranks.split(","))
    axis_ranks = tuple(int(v) for v in args.axis_ranks.split(","))
    if axis_ranks[0] * axis_ranks[1] != ranks[1]:
        print(f"warning: split ranks {axis_ranks} do not match the space-group "
              f"rank {ranks[1]}; the spatial latent budgets are not matched")
    models = [m for m in args.models.split(",")]
    rows, datasets = [], {}

    for coupling in (float(v) for v in args.couplings.split(",")):
        data = joint_diffusion_2d_tensor(
            shape=shape, coupling=coupling, joint_cutoff=args.joint_cutoff,
            axis_cutoff=args.axis_cutoff, time_cutoff=args.time_cutoff,
            wrong_coupling=args.wrong_coupling,
            learner_coupling=args.learner_coupling)
        residuals = projection_residuals(data)
        cost = basis_construction_cost(data)
        datasets[f"{coupling:.2f}"] = {
            "name": data.name, "shape": list(data.shape),
            "metadata": data.metadata, "projection_residuals": residuals,
            "basis_cost": cost}
        truth = data.values.flatten()
        index_cache = {}

        for mask in args.masks.split(","):
            for ratio in (float(v) for v in args.ratios.split(",")):
                for seed in (int(v) for v in args.seeds.split(",")):
                    split = make_observation_split(data, ratio, mask, seed,
                                                   sensor_axes=(1, 2))
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
                        specs, groups = build_specs(name, data, ranks, axis_ranks,
                                                    args.neural_hidden, args.power,
                                                    args.scenario_kind)
                        if groups not in index_cache:
                            index_cache[groups] = grouped_indices(data.shape, groups)
                        index = index_cache[groups]
                        model = GroupedOperatorTucker(specs, power=args.power,
                                                      device=args.device)
                        model.fit(index[observed], y, steps=args.steps,
                                  reg_weight=args.reg, seed=seed)
                        prediction = model.predict(index)
                        mean = prediction.mean * scale + center
                        error = mean[split.held_out] - truth[split.held_out]
                        held_std = truth[split.held_out].std().clamp_min(1e-8)
                        rmse = float(error.square().mean().sqrt())
                        row = {
                            "coupling": coupling,
                            "separability_residual":
                                data.metadata["operator_separability_residual"],
                            "subspace_residual":
                                data.metadata["low_frequency_subspace_residual"],
                            "projection_residual": residuals.get(
                                {"joint_operator": "joint", "wrong_joint": "wrong_joint",
                                 "permuted_joint": "permuted_joint"}.get(name, name)),
                            "model": name, "mask": mask, "ratio": ratio,
                            "ratio_actual": split.ratio_actual,
                            "n_observed": int(len(observed)), "seed": seed,
                            "groups": [list(g) for g in groups],
                            "metrics": {
                                "rmse": rmse, "nrmse": float(rmse / held_std),
                                "mae": float(error.abs().mean())},
                            "observed_fit": {
                                "nrmse": float(
                                    (mean[observed] - noisy[observed]).square().mean().sqrt()
                                    / noisy[observed].std().clamp_min(1e-8))},
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "spatial_latent_dimension": model.spatial_latent_dimension(),
                            "core_size": math.prod(model.ranks),
                            "elapsed_seconds": time.perf_counter() - started,
                            "metadata": prediction.metadata,
                        }
                        rows.append(row)
                        print(f"eta={coupling:.2f} {mask} {ratio:g} s{seed} {name:16s} "
                              f"NRMSE={row['metrics']['nrmse']:.3f} "
                              f"params={row['parameters']}", flush=True)

    (args.output / "results.json").write_text(json.dumps(
        {"datasets": datasets, "arguments": vars(args), "results": rows},
        indent=2, default=str))


if __name__ == "__main__":
    main()
