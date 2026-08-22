#!/usr/bin/env python3
"""What the operator prior actually is, drawn rather than described.

The method's whole content is which functions the spatial factor is allowed to
be.  On a floor plan that is visible: the leading eigenfunctions of the
plan-aware operator are constant inside a room and change across a doorway,
because that is where the operator says the field can and cannot travel.  The
same operator with the walls removed produces smooth ramps that ignore them.

Nothing here is fitted.  These are the dictionaries, before any data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.tri import Triangulation

from geoaware.floorplan import LAYOUTS, floorplan_tensor
from geoaware.irregular_fem import build_mesh
from geoaware.simplex_fem import SimplexMesh


def triangulation(layout: str, resolution: int):
    outline, walls = LAYOUTS[layout]
    planar = build_mesh(resolution, (), polygon=outline, seed=0)
    mesh = SimplexMesh(planar.nodes, planar.triangles, "triangles")
    nodes = mesh.nodes.numpy()
    return Triangulation(nodes[:, 0], nodes[:, 1], mesh.cells.numpy()), walls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", default="apartment")
    parser.add_argument("--resolution", type=int, default=130)
    parser.add_argument("--modes", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = floorplan_tensor(args.layout, resolution=args.resolution,
                            n_scenarios=4, n_time=4, basis_cutoff=16,
                            truth_modes=40)
    tri, walls = triangulation(args.layout, args.resolution)
    matrices = data.operator_matrices

    rows = [("Plan-aware operator", "geometry_operator_basis"),
            ("Same operator, walls removed", "blind_operator_basis"),
            ("Separable chart of the bounding box", "flat_chart_basis")]
    fig, axes = plt.subplots(len(rows), args.modes,
                             figsize=(3.1 * args.modes, 2.3 * len(rows)))
    for row, (title, key) in enumerate(rows):
        basis = matrices[key].numpy()
        for column in range(args.modes):
            axis = axes[row, column]
            # Skip the constant mode: it is the same everywhere and says nothing.
            values = basis[:, column + 1]
            limit = np.abs(values).max()
            axis.tripcolor(tri, values, shading="gouraud", cmap="RdBu_r",
                           vmin=-limit, vmax=limit)
            for wall in walls:
                a = np.asarray(wall.start)
                b = np.asarray(wall.end)
                along = b - a
                length = np.linalg.norm(along)
                direction = along / max(length, 1e-12)
                cuts = [0.] + [t for door in wall.doors for t in door] + [length]
                for start, end in zip(cuts[0::2], cuts[1::2]):
                    p, q = a + direction * start, a + direction * end
                    axis.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=2.2)
            axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal")
            if column == 0:
                axis.set_ylabel(title, fontsize=8)
            if row == 0:
                axis.set_title(f"mode {column + 2}", fontsize=9)
    fig.suptitle(f"The dictionaries, before any data — {args.layout}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output / f"basis_{args.layout}.png", dpi=170)
    print(f"wrote {args.output}/basis_{args.layout}.png")


if __name__ == "__main__":
    main()
