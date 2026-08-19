#!/usr/bin/env python3
"""Aggregate the geometry-family screen into a paper-facing table and figures.

Everything is computed from the raw per-seed JSON.  Nothing is selected after
seeing the numbers: the model order, the geometry order and the reported
statistics are fixed here, and the plain square is always shown because it is
the control that is *supposed* to show no advantage.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = ["fem_operator", "laplacian_geo", "topology_erased", "bounding_box",
               "laplacian_blind", "neural_coords", "neural_matched",
               "discrete_table", "laplacian_table", "permuted"]
MODEL_LABEL = {
    "fem_operator": "Geometry operator (ours)",
    "topology_erased": "Topology erased",
    "bounding_box": "Bounding-box product",
    "neural_coords": "Neural coordinates",
    "neural_matched": "Neural coords (matched)",
    "discrete_table": "Discrete Tucker",
    "laplacian_geo": "Laplacian, geometry-aware",
    "laplacian_blind": "Laplacian, geometry-blind",
    "laplacian_table": "Laplacian-regularized",
    "permuted": "Permuted (control)",
}
# Ordered by how much geometry there is to know, measured by the bias floor a
# geometry-blind basis pays.  "open" is first because it is the control where
# the advantage must vanish.
GEOMETRY_ORDER = ["open", "square", "labyrinth", "arc", "center_hole",
                  "two_holes", "L_shape", "U_shape", "chamber", "sealed_4"]


def load(paths):
    rows, geometries = [], {}
    for path in paths:
        payload = json.loads((path / "results.json").read_text())
        rows.extend(payload["results"])
        geometries.update(payload["geometries"])
    return rows, geometries


def select(rows, **filters):
    out = [r for r in rows
           if all(abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v
                  for k, v in filters.items())]
    return sorted(out, key=lambda r: r["seed"])


def summarize(rows, key="nrmse"):
    values = [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
    if not values:
        return None
    return {"mean": st.mean(values),
            "std": st.stdev(values) if len(values) > 1 else 0.,
            "by_seed": {r["seed"]: r["metrics"][key] for r in rows}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", default="random")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, geometries = load(args.inputs)
    layouts = [g for g in GEOMETRY_ORDER if g in geometries]
    ratios = sorted({r["ratio"] for r in rows})
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]

    summary = {"mask": args.mask, "layouts": {}, "ratios": ratios}
    for layout in layouts:
        entry = {"projection_residuals":
                 geometries[layout].get("projection_residuals", {}),
                 "n_nodes": geometries[layout]["shape"][1],
                 "metadata": geometries[layout]["metadata"], "cells": {}}
        for ratio in ratios:
            cell = {}
            reference = select(rows, layout=layout, ratio=ratio, mask=args.mask,
                               model="fem_operator")
            for model in models:
                picked = select(rows, layout=layout, ratio=ratio, mask=args.mask,
                                model=model)
                if not picked:
                    continue
                record = {"nrmse": summarize(picked),
                          "boundary_band_nrmse": summarize(picked, "boundary_band_nrmse"),
                          "parameters": picked[0]["parameters"]}
                if model != "fem_operator" and reference:
                    wins = sum(a["metrics"]["nrmse"] < b["metrics"]["nrmse"]
                               for a, b in zip(reference, picked))
                    record["ours_paired_wins"] = f"{wins}/{len(reference)}"
                cell[model] = record
            entry["cells"][f"{ratio:.2f}"] = cell
        summary["layouts"][layout] = entry
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    for ratio in ratios:
        print(f"\n=== held-out NRMSE, mask={args.mask}, observed={ratio:.0%} ===")
        header = f"{'model':26s}" + "".join(f"{g:>15s}" for g in layouts)
        print(header)
        for model in models:
            line = f"{MODEL_LABEL[model]:26s}"
            for layout in layouts:
                cell = summary["layouts"][layout]["cells"][f"{ratio:.2f}"].get(model)
                line += (f"{cell['nrmse']['mean']:.3f}±{cell['nrmse']['std']:.3f}".rjust(15)
                         if cell else " " * 15)
            print(line)

    fig, axes = plt.subplots(1, len(ratios), figsize=(5.2 * len(ratios), 4.2),
                             sharey=True)
    axes = [axes] if len(ratios) == 1 else list(axes)
    width = .8 / max(1, len(models))
    for axis, ratio in zip(axes, ratios):
        for index, model in enumerate(models):
            heights, errors = [], []
            for layout in layouts:
                cell = summary["layouts"][layout]["cells"][f"{ratio:.2f}"].get(model)
                heights.append(cell["nrmse"]["mean"] if cell else 0.)
                errors.append(cell["nrmse"]["std"] if cell else 0.)
            positions = [i + index * width - .4 for i in range(len(layouts))]
            axis.bar(positions, heights, width, yerr=errors, capsize=2,
                     label=MODEL_LABEL[model])
        axis.axhline(1., color="black", linestyle="--", linewidth=.8)
        axis.set_xticks(range(len(layouts)))
        axis.set_xticklabels(layouts, rotation=20, ha="right")
        axis.set_title(f"{ratio:.0%} observed")
        axis.set_ylabel("held-out NRMSE")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / "geometry_family_nrmse.png", dpi=150)

    # The mechanism figure: how much the geometry-aware subspace gains, against
    # how much geometry there is to know.  The control sits at the origin.
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    ratio = ratios[-1]
    for model in ("topology_erased", "bounding_box", "neural_coords",
                  "neural_matched"):
        xs, ys = [], []
        for layout in layouts:
            cells = summary["layouts"][layout]["cells"][f"{ratio:.2f}"]
            blind_floor = summary["layouts"][layout]["projection_residuals"].get(
                "topology_erased")
            if model not in cells or blind_floor is None:
                continue
            xs.append(blind_floor)
            ys.append(cells[model]["nrmse"]["mean"]
                      / max(cells["fem_operator"]["nrmse"]["mean"], 1e-9))
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            axis.plot([xs[i] for i in order], [ys[i] for i in order], "o-",
                      label=MODEL_LABEL[model])
    axis.axhline(1., color="black", linestyle="--", linewidth=.8)
    axis.set_xscale("log")
    axis.set_xlabel("bias floor of a geometry-blind basis  (how much geometry matters)")
    axis.set_ylabel(f"baseline NRMSE / ours, at {ratio:.0%}")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "advantage_vs_geometry_strength.png", dpi=150)
    print(f"\nwrote {args.output}/summary.json and two figures")


if __name__ == "__main__":
    main()
