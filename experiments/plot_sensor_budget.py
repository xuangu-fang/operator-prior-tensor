#!/usr/bin/env python3
"""Accuracy against sensor count, which is the currency a user actually has.

The lines say something the error table cannot.  The proposed model's error
keeps falling as instruments are added; the baselines flatten early, because
what limits them is not how much they have measured but what their function
space is able to represent.  No sensor budget repairs a basis that cannot put a
discontinuity at a wall.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"geometry_operator": ("Geometry operator (ours)", "tab:red", "o", "-"),
         "blind_operator": ("Same model, geometry removed", "tab:blue", "s", "--"),
         "neural_tucker": ("Neural functional Tucker", "tab:green", "^", "-.")}
TITLES = {"apartment": "Apartment (floor plan)",
          "lab_suite": "Laboratory suite (floor plan)",
          "sealed_4": "Sealed quadrants (synthetic)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=.075)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = json.loads((args.input / "results.json").read_text())["results"]
    grouped = defaultdict(list)
    counts = {}
    for row in rows:
        grouped[(row["layout"], row["model"], row["ratio"])].append(
            row["metrics"]["nrmse"])
        counts[(row["layout"], row["ratio"])] = row["n_sensors"]

    layouts = [l for l in TITLES if any(k[0] == l for k in grouped)]
    fig, axes = plt.subplots(1, len(layouts), figsize=(4.6 * len(layouts), 3.8),
                             sharey=True)
    axes = [axes] if len(layouts) == 1 else list(axes)
    summary = {}
    for axis, layout in zip(axes, layouts):
        ratios = sorted({k[2] for k in grouped if k[0] == layout})
        for model, (label, colour, marker, dash) in STYLE.items():
            xs = [counts[(layout, r)] for r in ratios
                  if (layout, model, r) in grouped]
            ys = [st.mean(grouped[(layout, model, r)]) for r in ratios
                  if (layout, model, r) in grouped]
            if not xs:
                continue
            axis.plot(xs, ys, marker=marker, color=colour, linestyle=dash,
                      label=label, markersize=5)
            summary.setdefault(layout, {})[model] = dict(zip(xs, ys))
        axis.axhline(args.target, color="grey", lw=.8, linestyle=":")
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel("sensors installed")
        axis.set_title(TITLES[layout], fontsize=10)
        # Where each method crosses the target, and what that costs in sensors.
        reached = {}
        for model in STYLE:
            points = summary.get(layout, {}).get(model, {})
            crossing = [n for n, v in sorted(points.items()) if v <= args.target]
            reached[model] = crossing[0] if crossing else None
        ours, other = reached["geometry_operator"], reached["neural_tucker"]
        note = (f"to reach {args.target:.3f}:  ours {ours} sensors,\n"
                + (f"neural {other} — {other / ours:.0f}x more"
                   if ours and other else "neural never, at any budget here"))
        axis.text(.04, .04, note, transform=axis.transAxes, fontsize=8,
                  va="bottom", bbox=dict(boxstyle="round", fc="white", ec="grey",
                                         alpha=.85))
    axes[0].set_ylabel("held-out NRMSE")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("What the geometry is worth, in instruments", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output / "sensor_budget.png", dpi=170)
    (args.output / "sensor_budget.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.output}/sensor_budget.png")


if __name__ == "__main__":
    main()
