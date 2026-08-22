"""Rooms, walls and doorways: geometry that genuinely constrains a field.

The Peclet sweep in Iteration 13 said what a useful geometry has to do -- it has
to still constrain where the field can be.  A cylinder in an open channel does
not; a floor plan does.  Air, heat and contaminants move between rooms only
through the openings, so a wall is a real constraint on the field no matter what
drives it.

This also puts the sampling protocol on its own feet.  Elsewhere in this
repository "spatial sensors" is a stylized mask; in a building it is the actual
deployment: a handful of sensors are installed, and the field everywhere else
has to be inferred.  Reconstruction from a few fixed instruments is the problem
this setting exists to solve, not an abstraction of one.

The layouts are ordinary building shapes at ordinary dimensions -- an open-plan
floor, a corridor with rooms off it, a two-bedroom apartment, a laboratory
suite with an airlock -- specified in metres and meshed directly.  They are not
traced from a particular building, and the module says so rather than implying
provenance it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from .bases import BasisSpec
from .irregular_fem import UNIT_SQUARE, Polygon, build_mesh
from .joint_diffusion_2d import GroupedFieldDataset
from .operator_diagnostics import (mass_orthonormalize_columns,
                                   sparse_eigenpairs)
from .simplex_fem import SimplexMesh, assemble_sparse, to_torch_sparse

# A wall is a thin band of very low diffusivity.  The hundred-to-one contrast is
# the same one the synthetic barrier family uses, and for the same reason: push
# it further and the wall's own relaxation becomes slower than the air's.
WALL = 1e-2


@dataclass(frozen=True)
class Segment:
    """A straight interior wall of a given thickness, with optional doorways.

    ``doors`` are intervals along the wall's own axis that stay open, which is
    what makes a floor plan a floor plan rather than a set of sealed boxes.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float = .12
    doors: tuple[tuple[float, float], ...] = ()
    conductivity: float = WALL

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        a = torch.tensor(self.start, dtype=torch.float64)
        b = torch.tensor(self.end, dtype=torch.float64)
        along = b - a
        length = along.norm().clamp_min(1e-12)
        direction = along / length
        offset = points.double() - a
        travel = offset @ direction
        lateral = (offset - travel[:, None] * direction).norm(dim=1)
        inside = (lateral <= self.thickness / 2) & (travel >= 0) & (travel <= length)
        for low, high in self.doors:
            inside &= ~((travel >= low) & (travel <= high))
        return inside


def _plan(width: float, height: float, walls: tuple[Segment, ...]):
    outline = Polygon(((0., 0.), (width, 0.), (width, height), (0., height)))
    return outline, walls


# Dimensions in metres.  Doorway widths are 0.9 m, which is a standard door.
LAYOUTS = {
    # No interior walls: the control, where there is no plan to know.
    "open_plan": _plan(12., 8., ()),
    # A corridor down the middle with rooms opening off it.
    "corridor": _plan(12., 8., (
        Segment((0., 4.6), (12., 4.6), doors=((2.2, 3.1), (6.0, 6.9), (9.4, 10.3))),
        Segment((4., 4.6), (4., 8.), doors=((1.6, 2.5),)),
        Segment((8., 4.6), (8., 8.), doors=((1.6, 2.5),)),
    )),
    # A two-bedroom flat: living space, two bedrooms and a bathroom.
    "apartment": _plan(12., 8., (
        Segment((5.5, 0.), (5.5, 8.), doors=((1.3, 2.2), (5.4, 6.3))),
        Segment((5.5, 3.4), (12., 3.4), doors=((1.1, 2.0),)),
        Segment((8.8, 3.4), (8.8, 8.), doors=((2.6, 3.5),)),
    )),
    # A laboratory suite: an inner room reachable only through an airlock, which
    # is the strongest constraint a real plan tends to have.
    "lab_suite": _plan(12., 8., (
        Segment((4.5, 0.), (4.5, 8.), doors=((3.2, 4.1),)),
        Segment((8.5, 0.), (8.5, 8.), doors=((6.0, 6.9),)),
        Segment((4.5, 3.0), (8.5, 3.0), doors=()),
    )),
}


def floorplan_tensor(layout: str, *, resolution: int = 90, n_scenarios: int = 12,
                     n_time: int = 12, basis_cutoff: int = 16,
                     truth_modes: int = 60, contrast: float = .3,
                     reaction: float = .05,
                     time_span: tuple[float, float] = (.15, 3.),
                     mesh_seed: int = 0, scenario_seed: int = 7717,
                     permutation_seed: int = 9173) -> GroupedFieldDataset:
    """``Y(release, time, node)``: a scalar dispersing through a floor plan.

    Each scenario is a different release point -- a spill, a heat source, an open
    window.  The learner is given the plan, which any building has, and is not
    given the smooth variation in the air's effective diffusivity, which nobody
    measures.
    """
    outline, walls = LAYOUTS[layout]
    planar = build_mesh(resolution, (), polygon=outline, seed=mesh_seed)
    mesh = SimplexMesh(planar.nodes, planar.triangles, "triangles")
    centroids = mesh.centroids()

    def material(*, background: bool) -> torch.Tensor:
        values = torch.ones(len(centroids), dtype=torch.float64)
        if background:
            phase = (centroids * torch.tensor([.31, .47], dtype=torch.float64)).sum(1)
            values = torch.exp(contrast * torch.sin(2 * math.pi * phase) * .5)
        for wall in walls:
            values = torch.where(wall.contains(centroids),
                                 torch.full_like(values, wall.conductivity), values)
        return values

    truth_stiffness, mass = assemble_sparse(mesh, material(background=True))
    aware_stiffness, _ = assemble_sparse(mesh, material(background=False))
    open_stiffness, _ = assemble_sparse(mesh, 1.)

    coordinates = mesh.nodes
    truth_modes = min(truth_modes, mesh.n_nodes - 1)
    truth_values, truth_vectors = sparse_eigenpairs(truth_stiffness, mass,
                                                    truth_modes)
    span = coordinates.max(0).values - coordinates.min(0).values
    rates = truth_values * float(span.max()) ** 2 / math.pi ** 2

    generator = torch.Generator().manual_seed(scenario_seed)
    releases = []
    for _ in range(n_scenarios):
        centre = coordinates.min(0).values + span * torch.rand(2, generator=generator).double()
        width = float(span.max()) * (.06 + .04 * float(torch.rand(1, generator=generator)))
        field = torch.exp(-((coordinates - centre) ** 2).sum(1) / (2 * width ** 2))
        releases.append(field - field.mean())
    initial = torch.stack(releases)

    time = torch.linspace(time_span[0], time_span[1], n_time, dtype=torch.float64)
    decay = torch.exp(-time[:, None] * (reaction + rates[None, :]))
    amplitudes = initial @ torch.from_numpy(mass @ truth_vectors.numpy()).double()
    values = torch.einsum("sq,tq,nq->stn", amplitudes, decay, truth_vectors)
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    aware_eigen, aware_basis = sparse_eigenpairs(aware_stiffness, mass, basis_cutoff)
    blind_eigen, blind_basis = sparse_eigenpairs(open_stiffness, mass, basis_cutoff)
    scale = math.pi ** 2 / float(span.max()) ** 2

    low = coordinates.min(0).values.double()
    scaled = (coordinates.double() - low) / span.double().clamp_min(1e-9)
    limit = int(math.ceil(math.sqrt(basis_cutoff))) + 3
    pairs = torch.cartesian_prod(torch.arange(limit), torch.arange(limit))
    pairs = pairs[torch.argsort((pairs.double() ** 2).sum(1), stable=True)[:basis_cutoff]]
    chart = (torch.cos(math.pi * scaled[:, 0:1] * pairs[:, 0].double())
             * torch.cos(math.pi * scaled[:, 1:2] * pairs[:, 1].double()))
    chart_basis = mass_orthonormalize_columns(chart, mass)
    permutation = torch.randperm(
        mesh.n_nodes, generator=torch.Generator().manual_seed(permutation_seed))

    time_cutoff = min(basis_cutoff, n_time)
    reference = (aware_eigen * scale)[:time_cutoff]
    time_basis = torch.linalg.qr(
        torch.exp(-time[:, None] * (reaction + reference[None, :])),
        mode="reduced").Q[:, :time_cutoff]

    spatial = {
        "geometry_operator": (aware_basis, aware_eigen * scale),
        "blind_operator": (blind_basis, blind_eigen * scale),
        "flat_chart": (chart_basis, (pairs.double() ** 2).sum(1)[:chart_basis.shape[1]]),
        "permuted": (aware_basis[permutation], aware_eigen * scale),
    }
    matrices = {"coordinates": coordinates.float(),
                "time_basis": time_basis.float(),
                "time_eigenvalues": reference.float(),
                "mesh_nodes": mesh.nodes.float(),
                "mesh_cells": mesh.cells,
                "aware_stiffness": to_torch_sparse(aware_stiffness),
                "blind_stiffness": to_torch_sparse(open_stiffness),
                "mass": to_torch_sparse(mass)}
    for name, (basis, eigenvalues) in spatial.items():
        matrices[f"{name}_basis"] = basis.float()
        matrices[f"{name}_eigenvalues"] = eigenvalues[:basis.shape[1]].float()

    metadata = {
        "family": "floorplan", "layout": layout,
        "floor_size_metres": [outline.array[:, 0].max(), outline.array[:, 1].max()],
        "n_walls": len(walls), "doorway_width_metres": .9,
        "wall_thickness_metres": walls[0].thickness if walls else None,
        "wall_conductivity": WALL,
        "geometry_provenance":
            "ordinary building layouts at ordinary dimensions, specified here; "
            "not traced from a particular building",
        "pde": "d_t c + (-div(a grad c) + reaction c) = 0 (dispersion indoors)",
        "tensor_semantics": "release x time x node",
        "scenarios": "one localized release per slice",
        "mesh_hash": mesh.hash(), "n_nodes": int(mesh.n_nodes),
        "n_cells": int(len(mesh.cells)), "resolution": int(resolution),
        "operator_information_tier":
            "geometry (the plan is known; the air's effective diffusivity is not)",
        "basis_cutoff": int(basis_cutoff), "truth_modes": int(truth_modes),
        "n_scenarios": int(n_scenarios), "scenario_seed": int(scenario_seed),
        "time_span": [float(time_span[0]), float(time_span[1])],
        "permutation_seed": int(permutation_seed),
        "coordinate_groups": [[0], [1], [2]],
    }
    specs = tuple(BasisSpec("neumann", max(1, basis_cutoff - 1), name)
                  for name in ("release", "time", "node"))
    return GroupedFieldDataset(
        f"floorplan_{layout}_r{resolution}_k{basis_cutoff}", values.float(),
        ("release", "time", "node"), specs, (False, False, False),
        "generated:geoaware.floorplan.floorplan_tensor",
        "Scalar dispersion through a building floor plan; rooms communicate only "
        "through doorways, so the plan constrains where the field can be.",
        metadata=metadata, groups=((0,), (1,), (2,)), operator_matrices=matrices)
