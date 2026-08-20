#!/usr/bin/env python3
"""Does the geometry advantage survive when the truth is not solved by the
learner's own operator?

Each input directory is one truth refinement: at refinement one the truth and
the learner share a discretization (the inverse crime), above one the truth is
solved on an independently seeded finer mesh and interpolated.  The comparison
that matters is not whether the numbers move -- they must, because the learner's
operator now carries a discretization error it did not before -- but whether the
ordering and the margin survive.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

MODELS = ["fem_operator", "topology_erased", "bounding_box", "neural_coords",
          "discrete_table"]
LABEL = {"fem_operator": "ours", "topology_erased": "topology erased",
         "bounding_box": "bounding box", "neural_coords": "neural coords",
         "discrete_table": "discrete Tucker"}


def load(path: Path):
    payload = json.loads((path / "results.json").read_text())
    refinement = payload["arguments"].get("truth_refinement", 1)
    return int(refinement), payload["results"], payload["geometries"]


def cell(rows, layout, model, mask, ratio):
    picked = sorted((r for r in rows
                     if r["layout"] == layout and r["model"] == model
                     and r["mask"] == mask and abs(r["ratio"] - ratio) < 1e-9),
                    key=lambda r: r["seed"])
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", default="spatial_sensors")
    parser.add_argument("--ratio", type=float, default=.10)
    parser.add_argument("--layouts", default="open,labyrinth,arc,chamber,sealed_4")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = dict()
    for path in args.inputs:
        refinement, rows, geometries = load(path)
        runs[refinement] = (rows, geometries, str(path))
    layouts = [l for l in args.layouts.split(",")]

    summary = {"mask": args.mask, "ratio": args.ratio, "refinements": {}}
    for refinement in sorted(runs):
        rows, geometries, path = runs[refinement]
        entry = {"path": path, "layouts": {}}
        print(f"\n=== truth refinement {refinement}x "
              f"({'inverse crime present' if refinement == 1 else 'inverse crime avoided'})"
              f", mask={args.mask}, observed={args.ratio:.0%} ===")
        header = f"{'model':18s}" + "".join(f"{l:>16s}" for l in layouts)
        print(header)
        for model in MODELS:
            line = f"{LABEL[model]:18s}"
            for layout in layouts:
                picked = cell(rows, layout, model, args.mask, args.ratio)
                if not picked:
                    line += " " * 16
                    continue
                values = [r["metrics"]["nrmse"] for r in picked]
                mean = st.mean(values)
                std = st.stdev(values) if len(values) > 1 else 0.
                line += f"{mean:.3f}±{std:.3f}".rjust(16)
                slot = entry["layouts"].setdefault(layout, {})
                slot[model] = {"mean": mean, "std": std,
                               "by_seed": {r["seed"]: r["metrics"]["nrmse"]
                                           for r in picked}}
            print(line)
        for layout in layouts:
            slot = entry["layouts"].get(layout, {})
            ours = cell(rows, layout, "fem_operator", args.mask, args.ratio)
            for model in MODELS[1:]:
                other = cell(rows, layout, model, args.mask, args.ratio)
                if ours and other:
                    wins = sum(a["metrics"]["nrmse"] < b["metrics"]["nrmse"]
                               for a, b in zip(ours, other))
                    slot[model]["ours_paired_wins"] = f"{wins}/{len(ours)}"
            slot["blind_bias_floor"] = geometries[layout][
                "projection_residuals"].get("topology_erased")
            slot["aware_bias_floor"] = geometries[layout][
                "projection_residuals"].get("fem_operator")
        summary["refinements"][str(refinement)] = entry

    print(f"\n=== ours vs topology-erased, paired wins over five seeds ===")
    print(f"{'refinement':12s}" + "".join(f"{l:>16s}" for l in layouts))
    for refinement in sorted(runs):
        entry = summary["refinements"][str(refinement)]
        line = f"{refinement}x".ljust(12)
        for layout in layouts:
            slot = entry["layouts"].get(layout, {})
            record = slot.get("topology_erased", {})
            ratio = (slot.get("fem_operator", {}).get("mean") or 0.)
            margin = (record.get("mean", 0.) / ratio) if ratio else 0.
            line += f"{record.get('ours_paired_wins', '-')} ({margin:.2f}x)".rjust(16)
        print(line)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}/summary.json")


if __name__ == "__main__":
    main()
