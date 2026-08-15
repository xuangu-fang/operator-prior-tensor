#!/usr/bin/env python3
"""Aggregate the calibrated operator-mismatch × observation-ratio experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "geo_btucker", "geo_bcp", "neural_functional_tucker",
    "neural_functional_cp", "siren_inr",
]
MODEL_LABELS = {
    "geo_btucker": "Operator Tucker",
    "geo_bcp": "Operator CP",
    "neural_functional_tucker": "Neural F-Tucker",
    "neural_functional_cp": "Neural F-CP",
    "siren_inr": "SIREN",
}


def summarize(path: Path) -> dict:
    payload = json.loads((path / "results.json").read_text())
    rows = payload["results"]
    mismatch = float(payload["arguments"]["mismatch"])
    ratios = sorted({float(row["ratio"]) for row in rows})
    if not ratios or ratios[-1] > .10:
        raise ValueError("Track-1 protocol forbids ratios above 10%")
    if any(row["arguments"]["task"] != "basis_mismatch" for row in rows):
        raise ValueError("all inputs must use the calibrated basis_mismatch task")
    table = {}
    for ratio in ratios:
        table[str(ratio)] = {}
        for model in MODEL_ORDER:
            selected = [row for row in rows
                        if float(row["ratio"]) == ratio and row["model"] == model]
            if not selected:
                continue
            held = [row["metrics"]["nrmse"] for row in selected]
            observed = [row["observed_fit"]["nrmse"] for row in selected]
            table[str(ratio)][model] = {
                "heldout_nrmse_mean": statistics.mean(held),
                "heldout_nrmse_std": statistics.stdev(held),
                "observed_nrmse_mean": statistics.mean(observed),
                "observed_nrmse_std": statistics.stdev(observed),
                "heldout_nrmse_by_seed": held,
                "observed_nrmse_by_seed": observed,
                "seeds": [row["seed"] for row in selected],
            }
    return {"mismatch": mismatch, "ratios": ratios, "table": table,
            "dataset": payload["dataset"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = sorted((summarize(path) for path in args.inputs),
                      key=lambda item: item["mismatch"])
    mismatches = [item["mismatch"] for item in datasets]
    ratios = datasets[0]["ratios"]
    if any(item["ratios"] != ratios for item in datasets):
        raise ValueError("all mismatch levels must share the same ratios")

    comparisons = {}
    for ratio in ratios:
        comparisons[str(ratio)] = {}
        for item in datasets:
            entries = item["table"][str(ratio)]
            operator = entries["geo_btucker"]["heldout_nrmse_by_seed"]
            neural = entries["neural_functional_tucker"]["heldout_nrmse_by_seed"]
            differences = [n - o for o, n in zip(operator, neural)]
            comparisons[str(ratio)][str(item["mismatch"])] = {
                "operator_tucker_advantage_mean": statistics.mean(differences),
                "operator_tucker_advantage_std": statistics.stdev(differences),
                "operator_tucker_seed_wins": sum(value > 0 for value in differences),
            }
    artifact = {
        "protocol": {
            "mismatch_definition": "relative Frobenius error after oracle projection onto the learner three-mode operator product space",
            "mismatch_levels": mismatches, "ratios": ratios,
            "seeds": [41, 42, 43], "steps": 500, "mask": "random",
            "noise_fraction": .10, "initialization": "cold-start",
            "selection": "no held-out model selection",
        },
        "datasets": datasets,
        "operator_tucker_vs_neural_functional_tucker": comparisons,
    }
    (args.output / "summary.json").write_text(json.dumps(artifact, indent=2))

    figure, axes = plt.subplots(1, len(ratios), figsize=(15, 4), sharey=True)
    colors = plt.get_cmap("tab10")
    for axis, ratio in zip(axes, ratios):
        for index, model in enumerate(MODEL_ORDER):
            means = [item["table"][str(ratio)][model]["heldout_nrmse_mean"]
                     for item in datasets]
            stds = [item["table"][str(ratio)][model]["heldout_nrmse_std"]
                    for item in datasets]
            axis.errorbar(mismatches, means, yerr=stds, marker="o", capsize=2,
                          linewidth=1.4, color=colors(index), label=MODEL_LABELS[model])
        axis.axhline(1, color="black", linestyle="--", linewidth=.8)
        axis.set_title(f"{100 * ratio:g}% observed")
        axis.set_xlabel("Oracle operator-space mismatch")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Held-out NRMSE")
    axes[-1].legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    figure.savefig(args.output / "mismatch_ratio_curves.png", dpi=180)

    advantage = np.array([
        [comparisons[str(ratio)][str(mismatch)]["operator_tucker_advantage_mean"]
         for mismatch in mismatches]
        for ratio in ratios
    ])
    limit = max(.05, float(np.abs(advantage).max()))
    figure, axis = plt.subplots(figsize=(9, 3.2))
    image = axis.imshow(advantage, cmap="RdBu", vmin=-limit, vmax=limit,
                        aspect="auto", origin="lower")
    axis.set_xticks(range(len(mismatches)), [f"{value:.2f}" for value in mismatches])
    axis.set_yticks(range(len(ratios)), [f"{100 * value:g}%" for value in ratios])
    axis.set_xlabel("Oracle operator-space mismatch")
    axis.set_ylabel("Observed entries")
    axis.set_title("Neural F-Tucker NRMSE − Operator Tucker NRMSE (positive: operator wins)")
    for row in range(len(ratios)):
        for column in range(len(mismatches)):
            axis.text(column, row, f"{advantage[row, column]:+.2f}",
                      ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Held-out NRMSE difference")
    figure.tight_layout()
    figure.savefig(args.output / "operator_advantage_phase_map.png", dpi=180)


if __name__ == "__main__":
    main()
