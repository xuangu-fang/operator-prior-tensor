#!/usr/bin/env python3
"""Audit the frozen fresh-seed and structured-fiber confirmation gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "geo_btucker": "Operator Tucker",
    "neural_functional_tucker": "Neural F-Tucker (wide)",
    "neural_functional_tucker_matched": "Neural F-Tucker (matched)",
    "wrong_btucker": "Wrong-operator Tucker",
}
ORDER = list(LABELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, payloads = [], []
    for directory in args.inputs:
        payload = json.loads((directory / "results.json").read_text())
        payloads.append(payload)
        rows.extend(payload["results"])
    first = payloads[0]
    frozen = first["arguments"]
    required = {
        "task": "diffusion_green", "mismatch": 1.0, "basis_cutoff": 8,
        "truth_modes": 14, "tucker_ranks": "4,5,5", "steps": 400,
        "power": 1.5, "reg": .002, "noise": .1, "init": "random",
    }
    for key, expected in required.items():
        if frozen[key] != expected:
            raise ValueError(f"frozen setting changed: {key}={frozen[key]!r}")
    if sorted({int(row["seed"]) for row in rows}) != [101, 102, 103, 104, 105]:
        raise ValueError("confirmation must use exactly five fresh seeds")
    if max(float(row["ratio"]) for row in rows) > .10:
        raise ValueError("confirmation ratio exceeds 10%")

    by_cell = defaultdict(dict)
    for row in rows:
        key = (row["mask"], float(row["ratio"]), row["model"])
        seed = int(row["seed"])
        if seed in by_cell[key]:
            raise ValueError(f"duplicate row for {key}, seed {seed}")
        by_cell[key][seed] = row

    masks = ["random", "source_fibers", "receiver_fibers"]
    ratios = [.02, .05, .10]
    summary = {}
    for mask in masks:
        summary[mask] = {}
        for ratio in ratios:
            cell = {}
            for model in ORDER:
                selected = by_cell[(mask, ratio, model)]
                if len(selected) != 5:
                    raise ValueError(f"missing fresh seeds for {mask}/{ratio}/{model}")
                values = [selected[seed]["metrics"]["nrmse"] for seed in sorted(selected)]
                parameters = sorted({selected[seed]["metadata"]["parameters"]
                                     for seed in selected})
                cell[model] = {
                    "nrmse_mean": statistics.mean(values),
                    "nrmse_std": statistics.stdev(values),
                    "nrmse_by_seed": dict(zip(map(str, sorted(selected)), values)),
                    "parameters": parameters[0] if len(parameters) == 1 else parameters,
                }
            operator = cell["geo_btucker"]["nrmse_by_seed"]
            cell["paired_vs_wide_neural"] = {
                "operator_seed_wins": sum(
                    operator[seed] < cell["neural_functional_tucker"]["nrmse_by_seed"][seed]
                    for seed in operator),
                "neural_minus_operator_mean": statistics.mean(
                    cell["neural_functional_tucker"]["nrmse_by_seed"][seed] - operator[seed]
                    for seed in operator),
            }
            cell["paired_vs_matched_neural"] = {
                "operator_seed_wins": sum(
                    operator[seed] < cell["neural_functional_tucker_matched"]["nrmse_by_seed"][seed]
                    for seed in operator),
                "neural_minus_operator_mean": statistics.mean(
                    cell["neural_functional_tucker_matched"]["nrmse_by_seed"][seed] - operator[seed]
                    for seed in operator),
            }
            summary[mask][str(ratio)] = cell

    residuals = {payload["dataset"]["metadata"]["oracle_product_projection_residual"]
                 for payload in payloads}
    if len(residuals) != 1:
        raise ValueError("projection residual changed across confirmation jobs")
    artifact = {
        "decision_rule": {
            "promotion": "Operator Tucker wins >=4/5 paired fresh seeds at one ratio <=10%, remains NRMSE<1, and survives a structured-fiber mask.",
            "hyperparameters_selected_on": [41, 42, 43],
            "confirmation_seeds": [101, 102, 103, 104, 105],
            "no_confirmation_tuning": True,
        },
        "frozen_protocol": required,
        "dataset": first["dataset"],
        "projection_residual": next(iter(residuals)),
        "summary": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(artifact, indent=2))

    figure, axes = plt.subplots(len(masks), len(ratios), figsize=(11.5, 8.4),
                               sharey=False, constrained_layout=True)
    for row_index, mask in enumerate(masks):
        for column_index, ratio in enumerate(ratios):
            axis = axes[row_index, column_index]
            cell = summary[mask][str(ratio)]
            models = ORDER
            means = [cell[model]["nrmse_mean"] for model in models]
            errors = [cell[model]["nrmse_std"] for model in models]
            x = np.arange(len(models))
            axis.bar(x, means, yerr=errors, capsize=3,
                     color=["#2962a3", "#d17a22", "#57a65a", "#9b4f96"])
            for position, model in zip(x, models):
                seed_values = list(cell[model]["nrmse_by_seed"].values())
                axis.scatter(np.full(5, position), seed_values, s=13, color="black", zorder=3)
            axis.axhline(1, color="gray", linestyle="--", linewidth=.8)
            axis.set_xticks(x, ["Operator", "Neural\nwide", "Neural\nmatched", "Wrong\noperator"],
                            fontsize=8)
            axis.set_title(f"{mask.replace('_', ' ')}, {100*ratio:g}%")
            axis.grid(axis="y", alpha=.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Held-out NRMSE")
    figure.savefig(args.output / "confirmation_nrmse.png", dpi=200)

    figure, axes = plt.subplots(1, len(masks), figsize=(11, 3.4), sharey=True,
                               constrained_layout=True)
    for axis, mask in zip(axes, masks):
        wide = [summary[mask][str(r)]["paired_vs_wide_neural"]["neural_minus_operator_mean"]
                for r in ratios]
        matched = [summary[mask][str(r)]["paired_vs_matched_neural"]["neural_minus_operator_mean"]
                   for r in ratios]
        axis.plot([2, 5, 10], wide, marker="o", label="wide neural")
        axis.plot([2, 5, 10], matched, marker="s", label="matched neural")
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(mask.replace("_", " "))
        axis.set_xlabel("Observed entries (%)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("NRMSE: neural − operator")
    axes[-1].legend(fontsize=8)
    figure.savefig(args.output / "paired_advantage.png", dpi=200)


if __name__ == "__main__":
    main()
