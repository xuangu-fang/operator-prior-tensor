"""The barrier benchmark in three dimensions and on a curved surface.

The planar barrier family established the claim; its weakness as a main table is
that everything happens on one flat square under one PDE.  This module carries
the *same* experimental design — one fixed mesh, barriers as thin bands of
near-zero conductivity, every layout sharing an identical node set — into two
settings where the geometry is qualitatively different:

``box``
    Tetrahedra filling a cube, divided by planar partitions with apertures.  A
    partition in three dimensions separates volumes rather than areas, and the
    aperture is a two-dimensional window rather than a gap in a line.

``sphere``
    Triangles on a geodesic sphere, divided by bands along parallels and
    meridians.  Here the operator is the Laplace-Beltrami operator of a closed
    manifold with no boundary and no global chart, so the geometry-blind
    controls — a cosine product basis of the enclosing box, a network reading
    raw ``(x, y, z)`` — are wrong for a structural reason and not by a margin.

Keeping the design fixed across all three settings is the point: a control still
differs from the proposed model in exactly one respect, so the comparison needs
no new argument in each new dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .bases import BasisSpec
from .joint_diffusion_2d import GroupedFieldDataset
from .operator_diagnostics import generalized_eigenpairs
from .simplex_fem import (SimplexMesh, assemble, assemble_sparse,
                          build_box_mesh, build_sphere_mesh)


@dataclass(frozen=True)
class Partition:
    """A planar slab of near-zero conductivity, optionally with a window."""

    axis: int
    center: float
    half_width: float = .05
    window: tuple[tuple[float, float], ...] = ()
    conductivity: float = 1e-3

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        inside = (points[:, self.axis] - self.center).abs() < self.half_width
        if not self.window:
            return inside
        through = torch.ones(len(points), dtype=torch.bool)
        other = [d for d in range(points.shape[1]) if d != self.axis]
        for dimension, (low, high) in zip(other, self.window):
            through &= (points[:, dimension] >= low) & (points[:, dimension] <= high)
        return inside & ~through


@dataclass(frozen=True)
class Band:
    """A band of near-zero conductivity around a great circle of a sphere.

    ``axis`` names the pole the band is measured from, ``angle`` its polar angle
    (``pi/2`` is the equator) and ``gap`` an azimuthal aperture left open.
    """

    axis: int
    angle: float
    half_width: float = .10
    gap: tuple[float, float] = (0., 0.)
    conductivity: float = 1e-3

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        direction = points / points.norm(dim=1, keepdim=True).clamp_min(1e-12)
        polar = torch.arccos(direction[:, self.axis].clamp(-1., 1.))
        inside = (polar - self.angle).abs() < self.half_width
        low, high = self.gap
        if low == high:
            return inside
        other = [d for d in range(points.shape[1]) if d != self.axis]
        azimuth = torch.atan2(direction[:, other[1]],
                              direction[:, other[0]]) % (2 * math.pi)
        in_gap = ((azimuth >= low) & (azimuth <= high) if low <= high
                  else (azimuth >= low) | (azimuth <= high))
        return inside & ~in_gap


BOX_LAYOUTS = {
    # No barrier: the control where there is no geometry to know.
    "open": (),
    # One partition with a square window in the middle.
    "window": (Partition(0, .5, window=((.35, .65), (.35, .65))),),
    # Two staggered partitions, each with a window at an opposite corner.  The
    # windows sit in the corners of the cube where the mesh is coarsest, so the
    # aperture is barely resolved; the layout is kept out of the reported ladder
    # for that reason and the diagnostic in the iteration log records why.
    "labyrinth": (Partition(0, .34, window=((.02, .32), (.02, .32))),
                  Partition(0, .66, window=((.68, .98), (.68, .98)))),
    # Two chambers joined by a narrow slot.
    "chamber": (Partition(0, .5, window=((.44, .56), (.44, .56))),),
    # Three orthogonal partitions with no windows: eight sealed octants.
    "sealed_8": (Partition(0, .5), Partition(1, .5), Partition(2, .5)),
}

@dataclass(frozen=True)
class Cap:
    """A spherical cap of land: a region the wave equation cannot enter.

    Under the linearized shallow-water equations the operator is
    ``-div(g H grad)`` with ``H`` the resting depth, so land is simply ``H ~ 0``.
    A coastline is public geographic information, which makes "the learner knows
    the geometry but not the material" the natural reading rather than a
    contrived one: bathymetry in the open ocean stays unknown.
    """

    center: tuple[float, float, float]
    angular_radius: float
    conductivity: float = 1e-3

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        axis = torch.tensor(self.center, dtype=torch.float64)
        axis = axis / axis.norm()
        direction = points / points.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.arccos((direction @ axis).clamp(-1., 1.)) < self.angular_radius


# Ocean layouts.  ``open_ocean`` is the headline case rather than a control:
# on a bare sphere the two operator bases coincide, so what is being tested is
# not "does the operator know the obstacles" but "does it know the domain is a
# manifold at all" -- and the Euclidean and lat-lon baselines are wrong there
# for structural reasons.  Land is added afterwards, as a second axis.
SPHERE_LAYOUTS = {
    # A water world: no coastline, only the curvature of the domain itself.
    "open_ocean": (),
    # One large landmass: waves must travel around a hemisphere-scale obstacle.
    "continent": (Cap((0., 0., 1.), .95),),
    # Two landmasses leaving two narrow straits between them.
    "two_continents": (Cap((0., 0., 1.), 1.05), Cap((0., 0., -1.), 1.05)),
    # Small scattered islands.  Reported because it is a *negative* result: a
    # cap far smaller than the basis wavelength costs the geometry-aware
    # operator more capacity than it returns.
    "archipelago": (Cap((0., 0., 1.), .22), Cap((1., .3, .4), .20),
                    Cap((-.6, .7, .1), .18), Cap((.2, -.9, -.3), .20)),
}


def barrier_coefficient(centroids: torch.Tensor, barriers: tuple,
                        contrast: float, *, background: bool) -> torch.Tensor:
    """Material seen by the truth (``background=True``) or by the learner.

    The barrier layout is known metadata, so both agree there.  The smooth
    background variation is *not* known, which keeps this a geometry prior
    rather than an exact-physics prior.
    """
    material = torch.ones(len(centroids), dtype=torch.float64)
    if background and contrast:
        phase = (centroids * torch.tensor([1.7, 2.3, 1.1][:centroids.shape[1]],
                                          dtype=torch.float64)).sum(1)
        material = torch.exp(contrast * torch.sin(2 * math.pi * phase) * .5)
    for barrier in barriers:
        material = torch.where(barrier.contains(centroids),
                               torch.full_like(material, barrier.conductivity),
                               material)
    return material


def _mass_orthonormalize(basis: torch.Tensor, mass) -> torch.Tensor:
    """Orthonormalize columns in the mass inner product, dropping dependencies.

    Delegates to the matrix-free routine: the Gram matrix is ``r x r`` whatever
    the mesh, so the same code serves a six-hundred-node sphere and a
    hundred-thousand-node one, dense mass matrix or sparse.
    """
    from .operator_diagnostics import mass_orthonormalize_columns
    return mass_orthonormalize_columns(basis, mass)


def _bounding_box_product_basis(coordinates: torch.Tensor, mass: torch.Tensor,
                                count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Separable cosine modes of the enclosing box, evaluated at the nodes.

    This is exactly the basis a per-axis method picks when it believes the
    domain is a rectangle or a brick.  On the sphere it is also what a method
    picks when it does not know the domain is a manifold at all.
    """
    dimension = coordinates.shape[1]
    limit = int(math.ceil(count ** (1 / dimension))) + 3
    grid = torch.cartesian_prod(*[torch.arange(limit)] * dimension)
    order = torch.argsort((grid.double() ** 2).sum(1), stable=True)[:count]
    grid = grid[order]
    low = coordinates.min(0).values.double()
    span = (coordinates.max(0).values.double() - low).clamp_min(1e-9)
    scaled = (coordinates.double() - low) / span
    columns = torch.ones(len(coordinates), len(grid), dtype=torch.float64)
    for dim in range(dimension):
        columns = columns * torch.cos(math.pi * scaled[:, dim:dim + 1]
                                      * grid[:, dim].double())
    return _mass_orthonormalize(columns, mass), (grid.double() ** 2).sum(1)


def _lat_lon_product_basis(coordinates: torch.Tensor, mass: torch.Tensor,
                           count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Separable cosine modes of the ``(theta, phi)`` rectangle.

    This is the control that matters most on a sphere, because it is what almost
    everyone actually does with spherical data: pick a latitude-longitude grid
    and treat the two angles as independent axes.  It is a perfectly reasonable
    per-axis tensor factorization and it is wrong, because the chart is singular
    at the poles -- every longitude meets there, so a product basis cannot
    represent a field that is smooth across a pole.

    The failure is structural, not a matter of resolution, which is exactly the
    kind of control this paper needs: no amount of extra rank repairs it.
    """
    direction = coordinates.double()
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    polar = torch.arccos(direction[:, 2].clamp(-1., 1.))          # theta in [0, pi]
    azimuth = torch.atan2(direction[:, 1], direction[:, 0]) % (2 * math.pi)
    limit = int(math.ceil(math.sqrt(count))) + 3
    pairs = torch.cartesian_prod(torch.arange(limit), torch.arange(limit))
    order = torch.argsort((pairs.double() ** 2).sum(1), stable=True)[:count]
    pairs = pairs[order].double()
    columns = (torch.cos(polar[:, None] * pairs[:, 0][None, :])
               * torch.cos(azimuth[:, None] * pairs[:, 1][None, :]))
    return _mass_orthonormalize(columns, mass), (pairs ** 2).sum(1)


def _smooth_initial_states(coordinates: torch.Tensor, count: int, seed: int, *,
                           on_sphere: bool = False) -> torch.Tensor:
    """Localized smooth bumps, one field per scenario.

    Bumps rather than draws from the operator's own eigenbasis: a truncated
    learner basis must face energy it cannot represent, otherwise the projection
    residual would be an artifact of the generator.

    On a closed surface the bumps have to be built *intrinsically*.  Sampling
    centres in the ambient box puts most of them inside the ball, where the
    distance to the surface is nearly constant and the resulting field is a
    high-frequency ring rather than a smooth cap.  Centres are therefore drawn
    on the surface itself and the width is a geodesic angle.
    """
    generator = torch.Generator().manual_seed(seed)
    dimension = coordinates.shape[1]
    points = coordinates.double()
    low = points.min(0).values
    span = points.max(0).values - low
    states = []
    for _ in range(count):
        field = torch.zeros(len(points), dtype=torch.float64)
        for _ in range(3):
            if on_sphere:
                direction = torch.randn(dimension, generator=generator).double()
                center = direction / direction.norm().clamp_min(1e-12)
                unit = points / points.norm(dim=1, keepdim=True).clamp_min(1e-12)
                distance = torch.arccos((unit @ center).clamp(-1., 1.))
                width = .45 + .35 * float(torch.rand(1, generator=generator))
            else:
                center = low + span * torch.rand(dimension, generator=generator).double()
                distance = (points - center).norm(dim=1)
                width = .14 + .12 * float(torch.rand(1, generator=generator))
            amplitude = float(torch.randn(1, generator=generator))
            field = field + amplitude * torch.exp(-distance ** 2 / (2 * width ** 2))
        states.append(field - field.mean())
    return torch.stack(states)


def _normalized_rates(values: torch.Tensor) -> torch.Tensor:
    """Eigenvalues on a fixed geometric scale, never a data-dependent one."""
    return values / (math.pi ** 2)


def barrier_field_tensor(
        barriers: tuple = (), *, geometry: str = "box", resolution: int = 10,
        subdivisions: int = 3, n_scenarios: int = 20, n_time: int = 16,
        basis_cutoff: int = 10, truth_modes: int = 60, contrast: float = .3,
        reaction: float = .15, time_span: tuple[float, float] = (.15, 3.),
        mesh_seed: int = 0, scenario_seed: int = 7717,
        permutation_seed: int = 9173, dynamics: str = "diffusion",
        wave_speed: float = 1., damping: float = .05,
        sparse: bool = False) -> GroupedFieldDataset:
    """``Y(scenario, time, node)`` on a cube or a sphere divided by barriers.

    ``dynamics`` selects the time propagator applied to the *same* operator
    eigenpairs:

    ``diffusion``
        ``exp(-(reaction + lambda) t)`` — the parabolic problem used everywhere
        else in this repository.
    ``wave``
        ``exp(-damping t) cos(wave_speed sqrt(lambda) t)`` — the linearized
        shallow-water equation for free-surface height, whose operator is
        ``-div(g H grad)`` with ``H`` the resting depth.  Land is ``H ~ 0``.

    The wave case matters beyond breadth.  Diffusion drives every field towards
    the operator's slowest eigenvectors, which is the degeneracy R5a flagged as
    a NO-GO; an oscillatory propagator keeps energy in high modes for all time,
    so the benchmark stops being generous in that particular way.
    """
    if geometry == "box":
        mesh = build_box_mesh(resolution, seed=mesh_seed)
    elif geometry == "sphere":
        mesh = build_sphere_mesh(subdivisions)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

    centroids = mesh.centroids()
    truth_material = barrier_coefficient(centroids, barriers, contrast,
                                         background=True)
    learner_material = barrier_coefficient(centroids, barriers, 0.,
                                           background=False)
    if sparse:
        from .operator_diagnostics import sparse_eigenpairs
        truth_stiffness, mass = assemble_sparse(mesh, truth_material)
        nominal_stiffness, _ = assemble_sparse(mesh, learner_material)
        blind_stiffness, _ = assemble_sparse(mesh, 1.)
        eigenpairs = sparse_eigenpairs
        def apply_mass(x):
            return torch.from_numpy(mass @ x.double().numpy()).double()
    else:
        truth_stiffness, mass = assemble(mesh, truth_material)
        nominal_stiffness, _ = assemble(mesh, learner_material)
        blind_stiffness, _ = assemble(mesh, 1.)
        eigenpairs = generalized_eigenpairs
        def apply_mass(x):
            return mass.double() @ x.double()
    coordinates = mesh.nodes
    n_nodes = mesh.n_nodes
    truth_modes = min(truth_modes, n_nodes)

    truth_values, truth_vectors = eigenpairs(
        truth_stiffness, mass, truth_modes)
    rates = _normalized_rates(truth_values)
    initial = _smooth_initial_states(coordinates, n_scenarios, scenario_seed,
                                     on_sphere=geometry == "sphere")
    amplitudes = initial @ apply_mass(truth_vectors)
    time = torch.linspace(time_span[0], time_span[1], n_time, dtype=torch.float64)
    if dynamics == "diffusion":
        propagator = torch.exp(-time[:, None] * (reaction + rates[None, :]))
    elif dynamics == "wave":
        frequency = wave_speed * rates.clamp_min(0.).sqrt()
        propagator = (torch.exp(-damping * time[:, None])
                      * torch.cos(time[:, None] * frequency[None, :]))
    else:
        raise ValueError(f"unknown dynamics {dynamics!r}")
    values = torch.einsum("sq,tq,nq->stn", amplitudes, propagator, truth_vectors)
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    nominal_values, aware_basis = eigenpairs(
        nominal_stiffness, mass, basis_cutoff)
    blind_values, blind_basis = eigenpairs(
        blind_stiffness, mass, basis_cutoff)
    box_basis, box_values = _bounding_box_product_basis(
        coordinates, mass, basis_cutoff)
    permutation = torch.randperm(
        n_nodes, generator=torch.Generator().manual_seed(permutation_seed))
    time_cutoff = min(basis_cutoff, n_time)
    reference_rates = _normalized_rates(nominal_values)[:time_cutoff]
    if dynamics == "diffusion":
        reference_curves = torch.exp(-time[:, None]
                                     * (reaction + reference_rates[None, :]))
    else:
        reference_curves = (torch.exp(-damping * time[:, None])
                            * torch.cos(time[:, None]
                                        * (wave_speed
                                           * reference_rates.clamp_min(0.).sqrt())[None, :]))
    time_basis = torch.linalg.qr(reference_curves, mode="reduced").Q[:, :time_cutoff]

    spatial = {
        "fem_correct": (aware_basis, _normalized_rates(nominal_values)),
        "topology_erased": (blind_basis, _normalized_rates(blind_values)),
        "bounding_box_product": (box_basis, box_values[:box_basis.shape[1]]),
        "permuted": (aware_basis[permutation], _normalized_rates(nominal_values)),
    }
    if geometry == "sphere":
        # The per-axis basis a practitioner reaches for on spherical data.
        chart_basis, chart_values = _lat_lon_product_basis(
            coordinates, mass, basis_cutoff)
        spatial["lat_lon_product"] = (chart_basis,
                                      chart_values[:chart_basis.shape[1]])
    # The dense operators exist only for the penalized-table controls, the one
    # component that cannot be written matrix-free.  On a large mesh they are
    # omitted and the runner skips those controls rather than silently failing.
    matrices = ({} if sparse else {
        "mass": mass.float(), "nominal_stiffness": nominal_stiffness.float(),
        "blind_stiffness": blind_stiffness.float()}) | {
        "coordinates": coordinates.float(), "time_basis": time_basis.float(),
        "time_eigenvalues": _normalized_rates(nominal_values)[:time_cutoff].float(),
    }
    for name, (basis, eigenvalues) in spatial.items():
        matrices[f"{name}_basis"] = basis.float()
        matrices[f"{name}_eigenvalues"] = eigenvalues[:basis.shape[1]].float()

    metadata = {
        "geometry": geometry, "ambient_dimension": int(coordinates.shape[1]),
        "manifold_dimension": int(mesh.degree),
        "pde": ("d_t u + (-div(a grad u) + reaction I) u = 0"
                if dynamics == "diffusion" else
                "d_tt eta = div(g H grad eta) - damping d_t eta "
                "(linearized shallow water; land is H ~ 0)"),
        "dynamics": dynamics,
        "boundary_condition": "Neumann on the cube; closed manifold on the sphere",
        "tensor_semantics": "scenario x time x node",
        "mesh_hash": mesh.hash(), "n_cells": int(len(mesh.cells)),
        "domain_measure": mesh.volume(), "sparse_operators": bool(sparse),
        "barriers": [{k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in vars(b).items()} for b in barriers],
        "barrier_kinds": [type(b).__name__ for b in barriers],
        "operator_information_tier":
            "geometry (mesh and barrier layout known; background material unknown)",
        "log_diffusivity_contrast": float(contrast), "reaction": float(reaction),
        "basis_cutoff": int(basis_cutoff), "time_cutoff": int(time_cutoff),
        "truth_modes": int(truth_modes), "n_free_nodes": int(n_nodes),
        "n_scenarios": int(n_scenarios), "scenario_seed": int(scenario_seed),
        "time_span": [float(time_span[0]), float(time_span[1])],
        "permutation_seed": int(permutation_seed),
        "coordinate_groups": [[0], [1], [2]],
    }
    specs = tuple(BasisSpec("neumann", max(1, basis_cutoff - 1), name)
                  for name in ("scenario", "decay-time", "node"))
    tag = f"{geometry}_{dynamics}_b{len(barriers)}_k{basis_cutoff}"
    return GroupedFieldDataset(
        f"barrier_field_{tag}", values.float(), ("scenario", "time", "node"),
        specs, (False, False, False),
        "generated:geoaware.manifold_barrier_data.barrier_field_tensor",
        f"Diffusion field on a {geometry} divided by thin impermeable barriers; "
        "all layouts share one mesh and differ only in the barrier material.",
        metadata=metadata, groups=((0,), (1,), (2,)), operator_matrices=matrices)
