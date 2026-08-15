#!/usr/bin/env python3
"""Conditional core-IV acquisition for operator Bayesian Tucker.

The acquisition score is the exact reduction in integrated posterior variance
on a fixed evaluation set, conditional on the learned Tucker factors.  Every
strategy is evaluated by the same final correct-operator Tucker model, so the
experiment isolates acquisition rather than reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from geoaware.bayes_models import ExactFeatureBayes
from geoaware.tensor_bayes import OperatorBayesianTucker
from geoaware.tensor_data import (
    explicit_mode_bases,
    flat_product_features,
    operator_tucker_tensor,
)


def fit_tucker(data, noisy, observed, kind, ranks, steps, reg, seed, device):
    center = float(noisy[observed].mean())
    scale = float(noisy[observed].std().clamp_min(1e-6))
    y = (noisy[observed]-center)/scale
    flat, flat_eigenvalues = flat_product_features(data, 512, kind)
    initializer = ExactFeatureBayes(
        flat, flat_eigenvalues, "loo", False, device).fit(observed, y).predict()
    basis, eigenvalues = explicit_mode_bases(data, kind)
    model = OperatorBayesianTucker(
        basis, eigenvalues, ranks=ranks, power=1.5, device=device)
    model.fit(data.flat_indices()[observed], y, steps=steps, reg_weight=reg,
              seed=seed, initial_tensor=initializer.mean.reshape(data.shape))
    return model, center, scale


@torch.no_grad()
def integrated_variance_acquisition(model, all_indices, candidates,
                                    evaluation, budget):
    """Greedy exact conditional IV reduction with rank-one covariance updates."""
    device = next(model.parameters()).device
    factors = model.factor_tables()
    candidate_design = model.tucker_design(
        all_indices[candidates].to(device), factors).double()
    evaluation_design = model.tucker_design(
        all_indices[evaluation].to(device), factors).double()
    covariance = model._posterior["cov"].to(device).double().clone()
    gram = evaluation_design.T @ evaluation_design / len(evaluation_design)
    noise_variance = float(model._posterior["noise"])**2
    available = torch.ones(len(candidates), dtype=torch.bool, device=device)
    selected_local, score_history = [], []
    for _ in range(budget):
        covariance_gram = covariance @ gram @ covariance
        numerator = ((candidate_design @ covariance_gram)*candidate_design).sum(1)
        denominator = noise_variance + (
            (candidate_design @ covariance)*candidate_design).sum(1)
        scores = numerator/denominator.clamp_min(1e-12)
        scores[~available] = -torch.inf
        local = int(torch.argmax(scores))
        z = candidate_design[local]
        covariance_z = covariance @ z
        covariance -= torch.outer(covariance_z, covariance_z)/denominator[local]
        available[local] = False
        selected_local.append(local); score_history.append(float(scores[local]))
    return candidates[torch.tensor(selected_local)], score_history


@torch.no_grad()
def evaluate(model, center, scale, data, truth, evaluation):
    prediction = model.predict(data.flat_indices()[evaluation])
    mean = prediction.mean*scale+center
    target = truth[evaluation]
    error = mean-target
    return {
        "nrmse": float(error.square().mean().sqrt()/target.std().clamp_min(1e-8)),
        "rmse": float(error.square().mean().sqrt()),
        "coverage95": float((error.abs() <= 1.96*prediction.std*scale).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--initial-ratio", type=float, default=.01)
    parser.add_argument("--acquire-ratio", type=float, default=.01)
    parser.add_argument("--evaluation-ratio", type=float, default=.2)
    parser.add_argument("--noise", type=float, default=.1)
    parser.add_argument("--initial-steps", type=int, default=100)
    parser.add_argument("--final-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = operator_tucker_tensor(); truth = data.values.flatten(); n = len(truth)
    all_indices = data.flat_indices(); rows = []
    for seed in map(int, args.seeds.split(",")):
        generator = torch.Generator().manual_seed(seed+81001)
        permutation = torch.randperm(n, generator=generator)
        n_evaluation = round(args.evaluation_ratio*n)
        n_initial = round(args.initial_ratio*n)
        n_acquire = round(args.acquire_ratio*n)
        evaluation = permutation[:n_evaluation]
        initial = permutation[n_evaluation:n_evaluation+n_initial]
        candidates = permutation[n_evaluation+n_initial:]
        noise_generator = torch.Generator().manual_seed(seed+82001)
        noisy = truth + torch.randn(n, generator=noise_generator)*args.noise*truth.std()
        started = time.perf_counter()
        proposal_sets = {}
        acquisition_meta = {}
        for name, kind in (("correct_core_iv", "correct"),
                           ("wrong_core_iv", "permuted")):
            acquisition_model, _, _ = fit_tucker(
                data, noisy, initial, kind, (3, 4, 4), args.initial_steps,
                .05, seed, args.device)
            selected, scores = integrated_variance_acquisition(
                acquisition_model, all_indices, candidates, evaluation, n_acquire)
            proposal_sets[name] = selected
            acquisition_meta[name] = {
                "first_score": scores[0], "last_score": scores[-1],
                "mean_score": float(np.mean(scores)),
            }
            del acquisition_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        random_generator = torch.Generator().manual_seed(seed+83001)
        proposal_sets["random"] = candidates[
            torch.randperm(len(candidates), generator=random_generator)[:n_acquire]]
        for strategy, selected in proposal_sets.items():
            observed = torch.cat([initial, selected])
            final_model, center, scale = fit_tucker(
                data, noisy, observed, "correct", (4, 5, 5),
                args.final_steps, .002, seed, args.device)
            row = {
                "seed": seed, "strategy": strategy,
                "initial_observations": len(initial),
                "acquired_observations": len(selected),
                "total_ratio": len(observed)/n,
                "fixed_evaluation_size": len(evaluation),
                "metrics": evaluate(final_model, center, scale, data, truth, evaluation),
                "acquisition": acquisition_meta.get(strategy),
            }
            rows.append(row)
            print(f"seed={seed} {strategy} NRMSE={row['metrics']['nrmse']:.6f}",
                  flush=True)
            del final_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"seed={seed} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    payload = {
        "experiment_id": "A-OPERATOR-TUCKER-CONDITIONAL-CORE-IV",
        "status": "EXPLORATORY_SELECTION",
        "protocol_note": (
            "The evaluation set is fixed and excluded from both initial observations "
            "and the acquisition pool. All strategies use the same final correct-operator model."),
        "dataset": {"name": data.name, "shape": data.shape,
                    "description": data.description},
        "arguments": vars(args), "results": rows,
    }
    (args.output/"results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
