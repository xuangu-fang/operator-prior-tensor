#!/usr/bin/env python3
"""The main table.

Rows are models, columns are geometries, cells are held-out NRMSE.  The two
numbers that carry the paper are the pair ``geometry_operator`` /
``blind_operator``, which differ in one thing only, and the tie between them on
the barrier-free control, which is what says the margin elsewhere is geometry
rather than capacity.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["geometry_operator", "blind_operator", "flat_chart", "neural_tucker",
         "neural_cp", "cp_als", "tucker_als", "permuted"]
LABEL = {"geometry_operator": "Geometry operator (ours)",
         "blind_operator": "Same model, geometry removed",
         "flat_chart": "Separable chart basis",
         "neural_tucker": "Neural functional Tucker",
         "neural_cp": "Neural functional CP",
         "cp_als": "CP-ALS (TensorLy)",
         "tucker_als": "Tucker-HOOI (TensorLy)",
         "permuted": "Permuted operator (control)"}
# Controls first in each family: the layout where there is nothing to know.
LAYOUTS = {
    "plane_barrier": ["open", "labyrinth", "arc", "chamber", "sealed_4"],
    "plane_domain": ["square", "center_hole", "two_holes", "L_shape", "U_shape"],
    "volume_barrier": ["open", "window", "chamber", "sealed_8"],
    "sphere": ["open_ocean"],
}


def load(paths, als_paths=()):
    """Learned-model rows, with the classical baselines taken from their own run.

    CP-ALS and Tucker-HOOI are fitted separately so they can sweep ranks on the
    CPU while the learned models occupy the GPU.  Where both sources contain a
    cell, the dedicated run wins: it is the one that gave the baseline an SVD
    start and its best rank.
    """
    rows, geometries = [], {}
    for path in paths:
        payload = json.loads((path / "results.json").read_text())
        rows.extend(payload["results"])
        geometries.update(payload.get("geometries", {}))
    replacements = []
    for path in als_paths:
        replacements.extend(json.loads((path / "results.json").read_text())["results"])
    if replacements:
        superseded = {(r["family"], r["layout"], r["model"], r["mask"],
                       r["ratio"], r["seed"]) for r in replacements}
        rows = [r for r in rows
                if (r["family"], r["layout"], r["model"], r["mask"],
                    r["ratio"], r["seed"]) not in superseded]
        rows.extend(replacements)
    return rows, geometries


def pick(rows, **filters):
    return sorted((r for r in rows if all(r[k] == v for k, v in filters.items())),
                  key=lambda r: r["seed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--als-inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=.10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, geometries = load(args.inputs, args.als_inputs)
    families = [f for f in LAYOUTS if any(r["family"] == f for r in rows)]
    summary = {"ratio": args.ratio, "masks": {}}

    for mask in ("spatial_sensors", "random"):
        if not any(r["mask"] == mask for r in rows):
            continue
        columns = [(f, l) for f in families for l in LAYOUTS[f]
                   if any(r["family"] == f and r["layout"] == l for r in rows)]
        print(f"\n=== held-out NRMSE, {mask}, {args.ratio:.0%} observed, "
              f"mean +- std over seeds ===")
        header = f"{'model':30s}" + "".join(f"{l[:13]:>15s}" for _, l in columns)
        print(header)
        print(f"{'':30s}" + "".join(f"{f[:13]:>15s}" for f, _ in columns))
        entry = {}
        for model in ORDER:
            line = f"{LABEL[model]:30s}"
            for family, layout in columns:
                chosen = pick(rows, family=family, layout=layout, model=model,
                              mask=mask, ratio=args.ratio)
                if not chosen:
                    line += " " * 15
                    continue
                values = [r["metrics"]["nrmse"] for r in chosen]
                cell = {"mean": st.mean(values),
                        "std": st.stdev(values) if len(values) > 1 else 0.,
                        "parameters": chosen[0]["parameters"],
                        "by_seed": {r["seed"]: r["metrics"]["nrmse"]
                                    for r in chosen}}
                if "selected_rank" in chosen[0]:
                    cell["selected_rank"] = [r["selected_rank"] for r in chosen]
                    cell["rank_selection"] = chosen[0]["rank_selection"]
                ours = pick(rows, family=family, layout=layout,
                            model="geometry_operator", mask=mask,
                            ratio=args.ratio)
                if model != "geometry_operator" and ours:
                    cell["ours_paired_wins"] = "{}/{}".format(
                        sum(a["metrics"]["nrmse"] < b["metrics"]["nrmse"]
                            for a, b in zip(ours, chosen)), len(ours))
                entry.setdefault(f"{family}/{layout}", {})[model] = cell
                line += f"{cell['mean']:.3f}±{cell['std']:.3f}".rjust(15)
            print(line)
        for family, layout in columns:
            key = f"{family}/{layout}"
            if key in geometries:
                entry[key]["projection_residuals"] = \
                    geometries[key]["projection_residuals"]
                entry[key]["n_nodes"] = geometries[key]["metadata"]["n_nodes"]
        summary["masks"][mask] = entry

        print(f"\n  ours vs the same model with geometry removed:")
        for family, layout in columns:
            cells = summary["masks"][mask][f"{family}/{layout}"]
            if "blind_operator" not in cells:
                continue
            ratio = cells["blind_operator"]["mean"] / max(
                cells["geometry_operator"]["mean"], 1e-9)
            print(f"    {family}/{layout:14s} {cells['geometry_operator']['mean']:.3f}"
                  f" vs {cells['blind_operator']['mean']:.3f}"
                  f"  = {ratio:5.2f}x  wins {cells['blind_operator'].get('ours_paired_wins','-')}")

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    for mask, entry in summary["masks"].items():
        keys = list(entry)
        models = [m for m in ORDER if any(m in entry[k] for k in keys)]
        fig, axis = plt.subplots(figsize=(1.5 * len(keys) + 4, 4.4))
        width = .8 / max(1, len(models))
        for index, model in enumerate(models):
            heights = [entry[k].get(model, {}).get("mean", 0.) for k in keys]
            errors = [entry[k].get(model, {}).get("std", 0.) for k in keys]
            axis.bar([i + index * width - .4 for i in range(len(keys))],
                     heights, width, yerr=errors, capsize=2, label=LABEL[model])
        axis.axhline(1., color="black", linestyle="--", linewidth=.8)
        axis.set_xticks(range(len(keys)))
        axis.set_xticklabels([k.split("/")[1] for k in keys], rotation=20,
                             ha="right", fontsize=8)
        axis.set_ylabel("held-out NRMSE")
        axis.set_title(f"{mask}, {args.ratio:.0%} observed")
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.output / f"main_{mask}.png", dpi=150)
    print(f"\nwrote {args.output}/summary.json and figures")


if __name__ == "__main__":
    main()
