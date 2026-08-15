#!/usr/bin/env python3
"""Aggregate the locked 2/5/10% Track-1 observation-ratio experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_ORDER = [
    "geo_btucker", "geo_bcp", "neural_functional_tucker",
    "neural_functional_cp", "siren_inr",
]


def summarize(path: Path) -> dict:
    payload = json.loads((path / "results.json").read_text())
    rows = payload["results"]
    ratios = sorted({float(row["ratio"]) for row in rows})
    if not ratios or ratios[-1] > .10:
        raise ValueError("Track-1 early-stage protocol forbids ratios above 10%")
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
                "generalization_gap_mean": statistics.mean(held) - statistics.mean(observed),
                "seeds": [row["seed"] for row in selected],
                "heldout_nrmse_by_seed": held,
                "observed_nrmse_by_seed": observed,
            }
    return {
        "dataset": payload["dataset"],
        "ratios": ratios,
        "models": MODEL_ORDER,
        "table": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = [summarize(path) for path in args.inputs]
    artifact = {
        "protocol": {
            "ratios": [.02, .05, .10], "seeds": [41, 42, 43],
            "steps": 500, "mask": "random", "noise_fraction": .10,
            "selection": "cold-start; no held-out model selection",
            "normalization": "per-mask observed values only",
        },
        "datasets": datasets,
    }
    (args.output / "summary.json").write_text(json.dumps(artifact, indent=2))

    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4),
                                squeeze=False, sharey=False)
    for axis, dataset in zip(axes[0], datasets):
        for model in MODEL_ORDER:
            means, stds = [], []
            for ratio in dataset["ratios"]:
                entry = dataset["table"][str(ratio)][model]
                means.append(entry["heldout_nrmse_mean"])
                stds.append(entry["heldout_nrmse_std"])
            axis.errorbar([100 * ratio for ratio in dataset["ratios"]], means,
                          yerr=stds, marker="o", capsize=3, label=model)
        axis.axhline(1, color="black", linestyle="--", linewidth=.8)
        axis.set_title(dataset["dataset"]["name"])
        axis.set_xlabel("Observed entries (%)")
        axis.set_ylabel("Held-out NRMSE")
        axis.set_xticks([2, 5, 10])
        axis.grid(alpha=.2)
    axes[0, -1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / "heldout_nrmse_phase_curve.png", dpi=180)


if __name__ == "__main__":
    main()
