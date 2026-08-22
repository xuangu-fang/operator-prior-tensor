#!/usr/bin/env python3
"""The main experiment: does knowing the geometry reconstruct the field better?

One claim, four geometry families, one table.  A spatiotemporal field is observed
at a few percent of its entries; the question is whether defining the spatial
factor's function space from the known geometry beats not doing so.

The comparison is deliberately narrow.  ``geometry_operator`` and
``blind_operator`` are the *same* model with the *same* decoder on the *same*
node set, differing only in whether the operator that defines the basis knows
the geometry.  Everything else is a standard baseline a tensor reader would ask
for: CP and Tucker fitted by alternating least squares from TensorLy, and their
functional counterparts whose factors are coordinate networks.

Two protocols, because they ask different questions.  Under random entries every
node appears in many observed entries, so a classical factorization is
well-posed and the comparison is a fair fight.  Under spatial sensors an
unobserved node appears in *no* observed entry, so its factor row is not
identifiable at all -- and a basis that ties nodes together through the geometry
is the only thing that makes the reconstruction defined.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from geoaware.als_baselines import cp_als, tucker_als
from functools import partial

from geoaware.benchmark import (FAMILIES, build_family, build_operator_variant)
from geoaware.grouped_operator_tucker import (GroupFactorSpec,
                                              GroupedOperatorTucker,
                                              grouped_indices)
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import (observable_modes,
                                           product_projection_residual)

SPECTRAL = {"geometry_operator": "geometry_operator",
            "blind_operator": "blind_operator",
            "flat_chart": "flat_chart",
            "permuted": "permuted"}
MODELS = ("geometry_operator", "blind_operator", "flat_chart", "neural_tucker",
          "neural_cp", "cp_als", "tucker_als", "permuted")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_specs(name, data, ranks, hidden, seen_nodes):
    """Same decoder for every learned model; only the node factor changes."""
    matrices = data.operator_matrices
    diagonal = name.endswith("_cp")
    node_rank = ranks[0] if diagonal else ranks[2]
    specs = [GroupFactorSpec("table", ranks[0], data.shape[0], name="scenario"),
             GroupFactorSpec("operator", ranks[0] if diagonal else ranks[1],
                             data.shape[1], basis=matrices["time_basis"],
                             eigenvalues=matrices["time_eigenvalues"], name="time")]
    if name in SPECTRAL:
        key = SPECTRAL[name]
        basis = matrices[f"{key}_basis"]
        eigenvalues = matrices[f"{key}_eigenvalues"]
        # A basis column that is not excited anywhere a sensor sits leaves its
        # coefficient unconstrained, and the model then extrapolates by whatever
        # that coefficient happens to be.  The screen uses only the basis and the
        # mask -- never the data -- and is applied identically to every spectral
        # model, so it cannot favour one of them.
        keep = observable_modes(basis, seen_nodes)
        basis, eigenvalues = basis[:, keep], eigenvalues[keep]
        specs.append(GroupFactorSpec("operator", min(node_rank, basis.shape[1]),
                                     data.shape[2], basis=basis,
                                     eigenvalues=eigenvalues, name="node"))
    else:
        specs.append(GroupFactorSpec("neural", node_rank, data.shape[2],
                                     coordinates=matrices["coordinates"],
                                     hidden=hidden, name="node"))
    if diagonal:
        for spec in specs:
            spec.rank = ranks[0]
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ratios", default=".05,.10")
    parser.add_argument("--masks", default="spatial_sensors,random")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--n-scenarios", type=int, default=20)
    parser.add_argument("--n-time", type=int, default=16)
    # Omitted by default: each family carries the cutoff that equalizes its
    # geometry-aware approximation floor, fixed before any fitting.
    parser.add_argument("--basis-cutoff", type=int, default=None)
    # Same geometry, different equation.  A wave field keeps energy across the
    # spectrum instead of collapsing onto the slowest modes, so it needs more
    # columns for the same approximation quality -- cutoff 64 rather than 16 --
    # and both bases receive them.
    parser.add_argument("--dynamics", choices=["diffusion", "wave"], default=None)
    parser.add_argument("--truth-modes", type=int, default=60)
    # Large enough that the fit reaches its own approximation floor rather than
    # a rank ceiling; see Iteration 14.  The cost of a big core is now small
    # because the model contracts mode by mode instead of forming the design.
    parser.add_argument("--ranks", default="12,10,16")
    parser.add_argument("--cp-rank", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=48)
    # 1500 is where the operator models stop moving; the coordinate networks
    # gain at most another 0.012 by 4000, which is recorded rather than spent,
    # so the budget does not quietly favour the proposed model.
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--reg", type=float, default=.002)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--als-iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ranks = tuple(int(v) for v in args.ranks.split(","))
    rows, geometries = [], {}
    for family in args.families.split(","):
        for layout in FAMILIES[family].layouts:
            builder = (partial(build_operator_variant, dynamics=args.dynamics)
                       if args.dynamics else build_family)
            data = builder(family, layout, n_scenarios=args.n_scenarios,
                           n_time=args.n_time, basis_cutoff=args.basis_cutoff,
                           truth_modes=args.truth_modes)
            matrices = data.operator_matrices
            residuals = {name: product_projection_residual(
                            data.values, [None, None, matrices[f"{key}_basis"]])
                         for name, key in SPECTRAL.items()}
            key = f"{family}/{layout}"
            geometries[key] = {"name": data.name, "shape": list(data.shape),
                               "metadata": data.metadata,
                               "projection_residuals": residuals}
            print(f"[{key}] nodes={data.shape[2]} "
                  + " ".join(f"{k}={v:.3f}" for k, v in residuals.items()),
                  flush=True)

            index = grouped_indices(data.shape, ((0,), (1,), (2,)))
            truth = data.values.flatten()
            for mask in args.masks.split(","):
                for ratio in (float(v) for v in args.ratios.split(",")):
                    for seed in (int(v) for v in args.seeds.split(",")):
                        split = make_observation_split(data, ratio, mask, seed,
                                                       sensor_axes=(2,))
                        observed = torch.where(split.observed)[0]
                        seen = split.observed.reshape(data.shape).reshape(
                            -1, data.shape[2]).any(0)
                        generator = torch.Generator().manual_seed(seed + 4401)
                        noisy = truth.clone()
                        noisy[observed] += (
                            torch.randn(len(observed), generator=generator)
                            * args.noise * truth[observed].std())
                        centre = float(noisy[observed].mean())
                        scale = float(noisy[observed].std().clamp_min(1e-6))
                        y = (noisy[observed] - centre) / scale
                        held = split.held_out

                        for name in args.models.split(","):
                            seed_all(seed)
                            started = time.perf_counter()
                            if name in ("cp_als", "tucker_als"):
                                noisy_cube = noisy.reshape(data.shape)
                                predicted = (
                                    cp_als(noisy_cube, split.observed,
                                           args.cp_rank, seed=seed,
                                           n_iter_max=args.als_iters)
                                    if name == "cp_als" else
                                    tucker_als(noisy_cube, split.observed, ranks,
                                               seed=seed,
                                               n_iter_max=args.als_iters))
                                mean = predicted.flatten()
                                parameters = (args.cp_rank * sum(data.shape)
                                              if name == "cp_als" else
                                              int(np.prod(ranks))
                                              + sum(r * s for r, s
                                                    in zip(ranks, data.shape)))
                                kept = None
                            else:
                                specs = build_specs(name, data, ranks,
                                                    args.hidden, seen)
                                model = GroupedOperatorTucker(
                                    specs, power=args.power, device=args.device,
                                    core="diagonal" if name.endswith("_cp")
                                    else "dense")
                                model.fit(index[observed], y, steps=args.steps,
                                          reg_weight=args.reg, seed=seed)
                                mean = model.predict(index).mean * scale + centre
                                parameters = sum(p.numel()
                                                 for p in model.parameters())
                                kept = (int(specs[2].basis.shape[1])
                                        if specs[2].kind == "operator" else None)
                            error = mean[held] - truth[held]
                            rmse = float(error.square().mean().sqrt())
                            # How much of its own basis a model actually cashed
                            # in.  Far above one means the fit is rank-limited,
                            # and then a comparison between function spaces
                            # measures the rank ceiling instead: every model is
                            # held back by the same thing and the geometry never
                            # gets a chance to show.  This is cheap and it is
                            # reported next to every number precisely because
                            # its absence once hid exactly that.
                            own_floor = residuals.get(name)
                            attained = float(rmse / truth[held].std().clamp_min(1e-8))
                            headroom = (attained / own_floor
                                        if own_floor else None)
                            rows.append({
                                "family": family, "layout": layout,
                                "model": name, "mask": mask, "ratio": ratio,
                                "seed": seed, "n_observed": int(len(observed)),
                                "n_nodes": int(data.shape[2]),
                                "basis_columns_kept": kept,
                                "parameters": int(parameters),
                                "own_projection_floor": own_floor,
                                "attained_over_floor": headroom,
                                "metrics": {
                                    "rmse": rmse,
                                    "nrmse": attained,
                                    "mae": float(error.abs().mean())},
                                "elapsed_seconds": time.perf_counter() - started})
                            print(f"{key} {mask} {ratio} s{seed} {name:18s} "
                                  f"NRMSE={attained:.3f} "
                                  + (f"over_floor={headroom:5.1f}x "
                                     if headroom else "")
                                  + f"params={parameters}", flush=True)

    (args.output / "results.json").write_text(json.dumps(
        {"arguments": vars(args), "geometries": geometries, "results": rows},
        indent=2, default=str))


if __name__ == "__main__":
    main()
