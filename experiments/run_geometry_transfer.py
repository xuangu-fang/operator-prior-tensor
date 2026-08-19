#!/usr/bin/env python3
"""Can a model trained on one barrier layout be moved to another?

Every layout in this benchmark shares one mesh and one node set, so a transfer
experiment changes exactly one thing: the operator.  That makes the question
sharp.  A geometry-aware factor is ``F = Phi_g W``; carrying ``W`` and the core
to a new layout and recomputing ``Phi_{g'}`` yields a factor that satisfies the
new barriers for free.  A coordinate network or a free table carries the *shape
of the field it saw*, which is the wrong field on the new geometry.

Three regimes are reported:

``source``    fit and evaluate on the source layout — the reference point.
``zero_shot`` fit on the source, evaluate on the target with no target data.
``few_shot``  as above, then refit only the small Tucker core on a few target
              observations, leaving every factor untouched.

Few-shot is the honest headline: zero-shot asks the transferred coefficients to
be correct in absolute terms, while few-shot asks only that the *function space*
transfer, which is what an operator prior actually claims.
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
                                              grouped_indices)
from geoaware.irregular_green_data import WALL_LAYOUTS, wall_field_tensor
from geoaware.masks import make_observation_split

MODELS = ("fem_operator", "topology_erased", "neural_coords", "discrete_table")
BASIS_KEY = {"fem_operator": "fem_correct", "topology_erased": "topology_erased"}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_specs(name, data, ranks, hidden):
    matrices = data.operator_matrices
    specs = [GroupFactorSpec("table", ranks[0], data.shape[0], name="scenario"),
             GroupFactorSpec("operator", ranks[1], data.shape[1],
                             basis=matrices["time_basis"],
                             eigenvalues=matrices["time_eigenvalues"], name="time")]
    if name in BASIS_KEY:
        key = BASIS_KEY[name]
        specs.append(GroupFactorSpec("operator", ranks[2], data.shape[2],
                                     basis=matrices[f"{key}_basis"],
                                     eigenvalues=matrices[f"{key}_eigenvalues"],
                                     name="node"))
    elif name == "neural_coords":
        specs.append(GroupFactorSpec("neural", ranks[2], data.shape[2],
                                     coordinates=matrices["coordinates"],
                                     hidden=hidden, name="node"))
    else:
        specs.append(GroupFactorSpec("table", ranks[2], data.shape[2], name="node"))
    return specs


def retarget(model, name, target):
    """Point the fitted model at the target geometry's operator.

    Only the spatial basis is replaced.  The learned spectral coefficients, the
    time factor, the scenario table and the core all stay exactly as fitted on
    the source, which is what makes this a test of the function space rather
    than of re-optimization.
    """
    if name not in BASIS_KEY:
        return model
    key = BASIS_KEY[name]
    matrices = target.operator_matrices
    device = next(model.parameters()).device
    spec = model.specs[2]
    spec.basis = matrices[f"{key}_basis"].float().to(device)
    spec.eigenvalues = matrices[f"{key}_eigenvalues"].float().to(device)
    return model


def normalized_observations(data, ratio, mask, seed, noise):
    truth = data.values.flatten()
    split = make_observation_split(data, ratio, mask, seed, sensor_axes=(2,))
    observed = torch.where(split.observed)[0]
    generator = torch.Generator().manual_seed(seed + 4401)
    noisy = truth.clone()
    noisy[observed] += (torch.randn(len(observed), generator=generator)
                        * noise * truth[observed].std())
    center = float(noisy[observed].mean())
    scale = float(noisy[observed].std().clamp_min(1e-6))
    return truth, split, observed, (noisy[observed] - center) / scale, center, scale


def score(mean, truth, held):
    error = mean[held] - truth[held]
    rmse = float(error.square().mean().sqrt())
    return {"rmse": rmse,
            "nrmse": float(rmse / truth[held].std().clamp_min(1e-8)),
            "mae": float(error.abs().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", default="chamber>sealed_4,sealed_4>chamber,"
                                           "arc>chamber,open>sealed_4")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ratio", type=float, default=.10)
    parser.add_argument("--target-ratio", type=float, default=.02)
    parser.add_argument("--mask", default="spatial_sensors")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--resolution", type=int, default=18)
    parser.add_argument("--n-scenarios", type=int, default=20)
    parser.add_argument("--n-time", type=int, default=16)
    parser.add_argument("--basis-cutoff", type=int, default=10)
    parser.add_argument("--ranks", default="4,4,6")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--core-steps", type=int, default=200)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ranks = tuple(int(v) for v in args.ranks.split(","))
    build = dict(resolution=args.resolution, n_scenarios=args.n_scenarios,
                 n_time=args.n_time, basis_cutoff=args.basis_cutoff)
    cache = {}
    rows = []
    for pair in args.pairs.split(","):
        source_name, target_name = pair.split(">")
        for label in (source_name, target_name):
            if label not in cache:
                cache[label] = wall_field_tensor(WALL_LAYOUTS[label], **build)
        source, target = cache[source_name], cache[target_name]
        index = grouped_indices(source.shape, ((0,), (1,), (2,)))

        for seed in (int(v) for v in args.seeds.split(",")):
            truth_s, split_s, obs_s, y_s, _, _ = normalized_observations(
                source, args.ratio, args.mask, seed, args.noise)
            truth_t, split_t, obs_t, y_t, center_t, scale_t = normalized_observations(
                target, args.target_ratio, args.mask, seed + 500, args.noise)
            for name in args.models.split(","):
                seed_all(seed)
                started = time.perf_counter()
                model = GroupedOperatorTucker(
                    build_specs(name, source, ranks, args.hidden),
                    device=args.device).fit(index[obs_s], y_s, steps=args.steps,
                                            seed=seed)
                # Source reference uses its own observed-only normalization.
                _, _, _, _, cs, ss = normalized_observations(
                    source, args.ratio, args.mask, seed, args.noise)
                source_mean = model.predict(index).mean * ss + cs
                record = {"pair": pair, "source": source_name,
                          "target": target_name, "model": name, "seed": seed,
                          "source_nrmse": score(source_mean, truth_s,
                                                split_s.held_out)["nrmse"]}

                retarget(model, name, target)
                zero = model.predict(index).mean * scale_t + center_t
                record["zero_shot"] = score(zero, truth_t, split_t.held_out)

                # Few-shot: only the core moves, so the factors — and therefore
                # the function space — are entirely inherited.
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.core.requires_grad_(True)
                optimizer = torch.optim.AdamW([model.core], lr=3e-3)
                device = next(model.parameters()).device
                ix, yy = index[obs_t].to(device), y_t.to(device)
                for _ in range(args.core_steps):
                    loss = (model(ix) - yy).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                history = model._posterior.get("history", [])
                objective = model._posterior.get("best_observed_objective")
                model._fit_core_posterior(ix, yy)
                model._posterior["history"] = history
                model._posterior["best_observed_objective"] = objective
                few = model.predict(index).mean * scale_t + center_t
                record["few_shot"] = score(few, truth_t, split_t.held_out)
                record["elapsed_seconds"] = time.perf_counter() - started
                rows.append(record)
                print(f"{pair} s{seed} {name:16s} source={record['source_nrmse']:.3f} "
                      f"zero={record['zero_shot']['nrmse']:.3f} "
                      f"few={record['few_shot']['nrmse']:.3f}", flush=True)

    (args.output / "results.json").write_text(json.dumps(
        {"arguments": vars(args), "results": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
