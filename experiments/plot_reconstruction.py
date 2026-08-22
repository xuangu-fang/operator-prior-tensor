#!/usr/bin/env python3
"""What the reader should be able to see without reading a number.

A release disperses through a floor plan and is measured at a few sensors.  The
plan-aware model keeps the field inside the rooms it can reach; the same model
with the walls removed smears it straight through them, which is the error the
whole paper is about.  Sensor positions are drawn so the reader can check that
neither model was given more information than the other.
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

from geoaware.benchmark import build_family
from geoaware.floorplan import LAYOUTS
from geoaware.grouped_operator_tucker import (GroupedOperatorTucker,
                                              grouped_indices)
from geoaware.irregular_fem import build_mesh
from geoaware.masks import make_observation_split
from geoaware.simplex_fem import SimplexMesh

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_geometry_main import build_specs

PANELS = [("Truth", None),
          ("Plan-aware operator (ours)", "geometry_operator"),
          ("Same model, walls removed", "blind_operator"),
          ("Neural functional Tucker", "neural_tucker")]


def draw_walls(axis, walls):
    for wall in walls:
        a, b = np.asarray(wall.start), np.asarray(wall.end)
        along = b - a
        length = float(np.linalg.norm(along))
        direction = along / max(length, 1e-12)
        cuts = [0.] + [t for door in wall.doors for t in door] + [length]
        for start, end in zip(cuts[0::2], cuts[1::2]):
            p, q = a + direction * start, a + direction * end
            axis.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=2.4,
                      solid_capstyle="butt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", default="apartment")
    parser.add_argument("--ratio", type=float, default=.10)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--slice", type=int, default=2)
    parser.add_argument("--time", type=int, default=4)
    parser.add_argument("--ranks", default="12,10,16")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranks = tuple(int(v) for v in args.ranks.split(","))

    data = build_family("floorplan", args.layout, n_scenarios=12, n_time=12)
    outline, walls = LAYOUTS[args.layout]
    planar = build_mesh(data.metadata["resolution"], (), polygon=outline, seed=0)
    mesh = SimplexMesh(planar.nodes, planar.triangles, "triangles")
    nodes = mesh.nodes.numpy()
    tri = Triangulation(nodes[:, 0], nodes[:, 1], mesh.cells.numpy())

    index = grouped_indices(data.shape, ((0,), (1,), (2,)))
    truth = data.values.flatten()
    split = make_observation_split(data, args.ratio, "spatial_sensors",
                                   args.seed, sensor_axes=(2,))
    observed = torch.where(split.observed)[0]
    seen = split.observed.reshape(data.shape).reshape(-1, data.shape[2]).any(0)
    generator = torch.Generator().manual_seed(args.seed + 4401)
    noisy = truth.clone()
    noisy[observed] += (torch.randn(len(observed), generator=generator)
                        * .1 * truth[observed].std())
    centre = float(noisy[observed].mean())
    scale = float(noisy[observed].std().clamp_min(1e-6))
    y = (noisy[observed] - centre) / scale

    fields, errors = {}, {}
    for title, name in PANELS:
        if name is None:
            fields[title] = data.values[args.slice, args.time].numpy()
            continue
        torch.manual_seed(args.seed)
        model = GroupedOperatorTucker(
            build_specs(name, data, ranks, 48, seen), device="cuda")
        model.fit(index[observed], y, steps=args.steps, seed=args.seed)
        mean = (model.predict(index).mean * scale + centre).reshape(data.shape)
        fields[title] = mean[args.slice, args.time].numpy()
        held = split.held_out
        flat = mean.flatten()
        errors[title] = float((flat[held] - truth[held]).square().mean().sqrt()
                              / truth[held].std())
        print(f"{name:20s} NRMSE={errors[title]:.4f}", flush=True)

    limit = np.abs(fields["Truth"]).max()
    sensors = nodes[seen.numpy()]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.4 * len(PANELS), 2.9))
    for axis, (title, name) in zip(axes, PANELS):
        axis.tripcolor(tri, fields[title], shading="gouraud", cmap="magma",
                       vmin=-limit * .35, vmax=limit)
        draw_walls(axis, walls)
        # A thousand markers hide the field they are meant to explain, so a
        # fixed sample is drawn: enough to show where the instruments are and
        # that both panels got the same ones.
        shown = sensors[::max(1, len(sensors) // 120)]
        axis.scatter(shown[:, 0], shown[:, 1], s=5, c="#7CFC00",
                     edgecolors="black", linewidths=.25, alpha=.9)
        label = title if name is None else f"{title}\nheld-out NRMSE {errors[title]:.3f}"
        axis.set_title(label, fontsize=9)
        axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal")
    fig.suptitle(f"{args.layout}: one release at one instant, reconstructed from "
                 f"{int(seen.sum())} sensors of {data.shape[2]} nodes "
                 f"({args.ratio:.0%}); a sample of the sensors is marked",
                 fontsize=10)
    fig.tight_layout()
    target = args.output / f"reconstruction_{args.layout}.png"
    fig.savefig(target, dpi=170)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
