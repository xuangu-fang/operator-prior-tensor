#!/usr/bin/env python3
"""Aggregate physical diffusion-operator perturbation experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = [
    "geo_btucker", "geo_bcp", "neural_functional_tucker",
    "neural_functional_cp", "siren_inr",
]
LABELS = {
    "geo_btucker": "Operator Tucker",
    "geo_bcp": "Operator CP",
    "neural_functional_tucker": "Neural F-Tucker",
    "neural_functional_cp": "Neural F-CP",
    "siren_inr": "SIREN",
}


def load_result(path: Path) -> dict:
    payload = json.loads((path / "results.json").read_text())
    arguments, dataset, rows = payload["arguments"], payload["dataset"], payload["results"]
    if arguments["task"] != "diffusion_green":
        raise ValueError(f"{path} is not a diffusion_green experiment")
    ratios = sorted({float(row["ratio"]) for row in rows})
    if not ratios or ratios[-1] > .10:
        raise ValueError("observation ratio must not exceed 10%")
    if len({int(row["metadata"]["gradient_steps"]) for row in rows}) != 1:
        raise ValueError("mixed optimization budgets inside one result")
    table = {}
    for ratio in ratios:
        table[str(ratio)] = {}
        for model in MODELS:
            selected = [row for row in rows
                        if float(row["ratio"]) == ratio and row["model"] == model]
            if not selected:
                continue
            held = [float(row["metrics"]["nrmse"]) for row in selected]
            observed = [float(row["observed_fit"]["nrmse"]) for row in selected]
            table[str(ratio)][model] = {
                "heldout_nrmse_mean": statistics.mean(held),
                "heldout_nrmse_std": statistics.stdev(held),
                "heldout_nrmse_by_seed": held,
                "observed_nrmse_mean": statistics.mean(observed),
                "seeds": [int(row["seed"]) for row in selected],
            }
    metadata = dataset["metadata"]
    tucker_ranks = tuple(int(value) for value in arguments["tucker_ranks"].split(","))
    return {
        "input": str(path),
        "contrast": float(metadata["log_diffusivity_contrast"]),
        "basis_cutoff": int(metadata["basis_cutoff"]),
        "projection_residual": float(metadata["oracle_product_projection_residual"]),
        "tucker_core_size": int(tucker_ranks[0] * tucker_ranks[1] * tucker_ranks[2]),
        "tucker_ranks": tucker_ranks,
        "ratios": ratios,
        "table": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=["contrast", "basis_cutoff", "projection_residual",
                                           "tucker_core_size"],
                        default="projection_residual")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = sorted((load_result(path) for path in args.inputs),
                      key=lambda item: item[args.axis])
    ratios = datasets[0]["ratios"]
    if any(item["ratios"] != ratios for item in datasets):
        raise ValueError("all inputs must share observation ratios")

    paired = {}
    for ratio in ratios:
        paired[str(ratio)] = {}
        for item in datasets:
            operator = item["table"][str(ratio)]["geo_btucker"]["heldout_nrmse_by_seed"]
            neural = item["table"][str(ratio)]["neural_functional_tucker"]["heldout_nrmse_by_seed"]
            differences = [right - left for left, right in zip(operator, neural)]
            paired[str(ratio)][str(item[args.axis])] = {
                "neural_minus_operator_mean": statistics.mean(differences),
                "neural_minus_operator_std": statistics.stdev(differences),
                "operator_seed_wins": sum(value > 0 for value in differences),
            }
    artifact = {
        "protocol": {
            "generator": "variable-coefficient Neumann diffusion Green response",
            "mismatch_measure": "oracle relative Frobenius product-space projection residual",
            "ratios": ratios,
            "seeds": [41, 42, 43],
            "steps": 400,
            "noise_fraction": .1,
            "mask": "random",
            "selection": "no held-out tuning",
        },
        "axis": args.axis,
        "datasets": datasets,
        "paired_operator_tucker_vs_neural_tucker": paired,
    }
    (args.output / "summary.json").write_text(json.dumps(artifact, indent=2))

    figure, axes = plt.subplots(1, len(ratios), figsize=(11, 3.6), sharey=True,
                               constrained_layout=True)
    if len(ratios) == 1:
        axes = [axes]
    x = [item[args.axis] for item in datasets]
    for axis, ratio in zip(axes, ratios):
        for model in MODELS:
            means = [item["table"][str(ratio)][model]["heldout_nrmse_mean"]
                     for item in datasets]
            stds = [item["table"][str(ratio)][model]["heldout_nrmse_std"]
                    for item in datasets]
            axis.errorbar(x, means, yerr=stds, marker="o", capsize=2,
                          linewidth=1.3, label=LABELS[model])
        axis.set_title(f"{100 * ratio:g}% observed")
        axis.set_xlabel(args.axis.replace("_", " "))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Held-out NRMSE")
    axes[-1].legend(fontsize=7)
    figure.savefig(args.output / f"nrmse_vs_{args.axis}.png", dpi=180)

    figure, axes = plt.subplots(1, len(ratios), figsize=(9, 3), sharey=True,
                               constrained_layout=True)
    if len(ratios) == 1:
        axes = [axes]
    for axis, ratio in zip(axes, ratios):
        means = [paired[str(ratio)][str(item[args.axis])]["neural_minus_operator_mean"]
                 for item in datasets]
        errors = [paired[str(ratio)][str(item[args.axis])]["neural_minus_operator_std"]
                  for item in datasets]
        axis.errorbar(x, means, yerr=errors, marker="o", capsize=3)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(f"{100 * ratio:g}% observed")
        axis.set_xlabel(args.axis.replace("_", " "))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("NRMSE difference\n(neural − operator)")
    figure.savefig(args.output / f"operator_advantage_vs_{args.axis}.png", dpi=180)


if __name__ == "__main__":
    main()
