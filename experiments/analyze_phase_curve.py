#!/usr/bin/env python3
"""One curve for two unrelated geometry mechanisms.

Iteration 8b read a phase boundary off the barrier family: the geometry prior
pays off once the bias floor of ignoring the geometry becomes comparable to the
error the estimator could otherwise achieve.  That was fitted on one family, so
it was a description, not a prediction.

This script puts a second, mechanically unrelated family on the same axes --
polygonal domains with circular obstacles and reentrant corners, where geometry
enters through the *shape of the domain* rather than through internal barrier
material -- and asks whether the same scalar orders both.

The x axis is the blind bias floor itself -- a quantity computable from the data
and the two candidate bases *before any model is fitted*.  A dimensionless
version (floor divided by the error the proposed model attains) orders the
points better, but it shares a denominator with the reported advantage
``blind/ours``, and the control below shows that ``1/ours`` alone already
reproduces most of that ordering.  The normalized axis is therefore reported as
a diagnostic and the un-shared scalar is what the claim rests on.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_STYLE = {"barrier": ("o", "tab:blue"), "domain": ("s", "tab:red")}


def load(path: Path):
    payload = json.loads((path / "results.json").read_text())
    return payload["results"], payload["geometries"]


def mean_nrmse(rows, layout, model, mask, ratio):
    values = [r["metrics"]["nrmse"] for r in rows
              if r["layout"] == layout and r["model"] == model
              and r["mask"] == mask and abs(r["ratio"] - ratio) < 1e-9]
    return st.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", default="random")
    parser.add_argument("--ratio", type=float, default=.10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    points = []
    for family, path in (("barrier", args.barrier), ("domain", args.domain)):
        rows, geometries = load(path)
        for layout, info in geometries.items():
            ours = mean_nrmse(rows, layout, "fem_operator", args.mask, args.ratio)
            blind = mean_nrmse(rows, layout, "topology_erased", args.mask, args.ratio)
            neural = mean_nrmse(rows, layout, "neural_coords", args.mask, args.ratio)
            floor = info["projection_residuals"].get("topology_erased")
            if None in (ours, blind, floor):
                continue
            points.append({"family": family, "layout": layout,
                           "blind_bias_floor": floor, "ours": ours,
                           "blind": blind, "neural": neural,
                           "difficulty": floor / ours,
                           "blind_over_ours": blind / ours,
                           "neural_over_ours": neural / ours if neural else None})

    points.sort(key=lambda p: p["blind_bias_floor"])
    print(f"=== mask={args.mask}, observed={args.ratio:.0%} ===")
    print(f"{'family':9s}{'layout':14s}{'blind floor':>12s}{'ours':>8s}"
          f"{'floor/ours':>12s}{'blind/ours':>12s}{'neural/ours':>13s}")
    for p in points:
        neural = f"{p['neural_over_ours']:.2f}" if p["neural_over_ours"] else "-"
        print(f"{p['family']:9s}{p['layout']:14s}{p['blind_bias_floor']:12.3f}"
              f"{p['ours']:8.3f}{p['difficulty']:12.2f}{p['blind_over_ours']:12.2f}"
              f"{neural:>13s}")

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.] * len(values)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out

    def spearman(xs, ys):
        a, b = rank(xs), rank(ys)
        n = len(a)
        return (sum((x - st.mean(a)) * (y - st.mean(b)) for x, y in zip(a, b))
                / ((n - 1) * st.stdev(a) * st.stdev(b)))

    floors = [p["blind_bias_floor"] for p in points]
    ours = [p["ours"] for p in points]
    blind = [p["blind"] for p in points]
    advantage = [p["blind_over_ours"] for p in points]
    correlations = {
        # The claim: a scalar available before fitting orders the advantage.
        "floor_vs_advantage": spearman(floors, advantage),
        # No shared term on either side at all.
        "floor_vs_blind_error": spearman(floors, blind),
        # Reported, but not claimed: it shares ``ours`` with the advantage.
        "normalized_floor_vs_advantage": spearman(
            [f / o for f, o in zip(floors, ours)], advantage),
        # The control that says how much of the above is the shared denominator.
        "inverse_ours_vs_advantage": spearman([1 / o for o in ours], advantage),
    }
    n = len(points)
    print(f"\nSpearman over {n} geometries from two families:")
    for name, value in correlations.items():
        print(f"  {name:34s} {value:+.3f}")
    print("  the normalized axis is not claimed: 1/ours alone reaches "
          f"{correlations['inverse_ours_vs_advantage']:.3f}")

    fig, axis = plt.subplots(figsize=(6.8, 4.6))
    for family in ("barrier", "domain"):
        marker, color = FAMILY_STYLE[family]
        chosen = [p for p in points if p["family"] == family]
        axis.plot([p["blind_bias_floor"] for p in chosen],
                  [p["blind_over_ours"] for p in chosen], marker, color=color,
                  linestyle="-", label=f"{family} family")
        for p in chosen:
            axis.annotate(p["layout"],
                          (p["blind_bias_floor"], p["blind_over_ours"]),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.axhline(1., color="black", linestyle="--", linewidth=.8)
    axis.set_xlabel("bias floor of a geometry-blind basis (computable before fitting)")
    axis.set_ylabel(f"geometry-blind NRMSE / ours, at {args.ratio:.0%} {args.mask}")
    axis.set_title("One scalar orders two unrelated geometry mechanisms")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / f"phase_curve_{args.mask}.png", dpi=150)
    (args.output / f"phase_curve_{args.mask}.json").write_text(json.dumps(
        {"mask": args.mask, "ratio": args.ratio,
         "spearman": correlations, "points": points}, indent=2))
    print(f"wrote {args.output}/phase_curve_{args.mask}.png")


if __name__ == "__main__":
    main()
