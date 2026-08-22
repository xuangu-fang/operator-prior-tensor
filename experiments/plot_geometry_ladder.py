#!/usr/bin/env python3
"""The advantage against a number computable before any model is fitted.

Each point is one layout.  The horizontal axis is the projection residual a
geometry-blind basis pays on that layout -- available from the data and the two
candidate bases, with nothing trained -- and the vertical axis is the ratio of
held-out errors that actually resulted.  A method that comes with a diagnostic
saying in advance whether it is worth using is a different proposition from one
that has to be tried.
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

FAMILY_STYLE = {
    "plane_barrier": ("barriers in a plane", "tab:blue", "o"),
    "plane_domain": ("holes and corners", "tab:orange", "s"),
    "volume_barrier": ("partitions in a volume", "tab:green", "^"),
    "sphere": ("a curved surface", "tab:red", "*"),
    "floorplan": ("rooms and doorways", "tab:purple", "D"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", default="spatial_sensors")
    parser.add_argument("--ratio", type=float, default=.10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scores, floors = defaultdict(list), {}
    for path in args.inputs:
        payload = json.loads((path / "results.json").read_text())
        for key, info in payload["geometries"].items():
            floors[key] = info["projection_residuals"]["blind_operator"]
        for row in payload["results"]:
            if row["mask"] != args.mask or abs(row["ratio"] - args.ratio) > 1e-9:
                continue
            scores[(row["family"], row["layout"], row["model"])].append(
                row["metrics"]["nrmse"])

    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    points = []
    for (family, layout, model), values in scores.items():
        if model != "geometry_operator":
            continue
        blind = scores.get((family, layout, "blind_operator"))
        if not blind:
            continue
        points.append((family, layout, floors[f"{family}/{layout}"],
                       st.mean(blind) / st.mean(values)))
    for family, (label, colour, marker) in FAMILY_STYLE.items():
        chosen = sorted((p for p in points if p[0] == family), key=lambda p: p[2])
        if not chosen:
            continue
        axis.plot([p[2] for p in chosen], [p[3] for p in chosen], marker,
                  color=colour, markersize=9, linestyle="", label=label)
        for index, (_, layout, x, y) in enumerate(chosen):
            # Alternate the label side so the dense clusters stay legible.
            offset = (6, 3) if index % 2 == 0 else (-6, -10)
            axis.annotate(layout, (x, y), fontsize=7, xytext=offset,
                          textcoords="offset points", color=colour,
                          ha="left" if index % 2 == 0 else "right")
    def spearman(xs, ys):
        def rank(values):
            order = sorted(range(len(values)), key=lambda i: values[i])
            out = [0.] * len(values)
            for position, index in enumerate(order):
                out[index] = float(position)
            return out
        a, b = rank(xs), rank(ys)
        n = len(a)
        return (sum((x - st.mean(a)) * (y - st.mean(b)) for x, y in zip(a, b))
                / ((n - 1) * st.stdev(a) * st.stdev(b)))

    correlation = spearman([p[2] for p in points], [p[3] for p in points])
    planar = [p for p in points if p[0] != "volume_barrier"]
    planar_correlation = spearman([p[2] for p in planar], [p[3] for p in planar])
    axis.axhline(1., color="black", lw=.8, linestyle="--")
    axis.text(.03, .97,
              f"Spearman {correlation:+.2f} over {len(points)} layouts\n"
              f"{planar_correlation:+.2f} excluding the volume, where the\n"
              f"sensor budget cannot identify the modes",
              transform=axis.transAxes, va="top", ha="left", fontsize=8,
              bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=.9))
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("bias floor of a geometry-blind basis  (no fitting required)")
    axis.set_ylabel(f"geometry-blind NRMSE / ours  ({args.ratio:.0%}, {args.mask})")
    axis.set_title("What the geometry is worth, and when it is worth nothing")
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output / f"geometry_ladder_{args.mask}.png", dpi=170)
    (args.output / f"geometry_ladder_{args.mask}.json").write_text(
        json.dumps([{"family": f, "layout": l, "blind_floor": x, "ratio": y}
                    for f, l, x, y in sorted(points, key=lambda p: p[2])],
                   indent=2))
    print(f"wrote {args.output}/geometry_ladder_{args.mask}.png "
          f"({len(points)} layouts)")


if __name__ == "__main__":
    main()
