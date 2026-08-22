"""The main benchmark: one claim, four geometries, one code path.

The claim is a single sentence.  A spatiotemporal field is observed at a few
percent of its entries on a domain whose geometry is known; using that geometry
to define the function space the spatial factor lives in reconstructs the field
better than not using it.  Everything here exists to test that and nothing else.

Four families vary *how* geometry enters, and nothing else:

``plane_barrier``   impermeable walls inside a square -- geometry is an obstacle
``plane_domain``    circular holes and reentrant corners -- geometry is the shape
``volume_barrier``  partitions inside a cube -- the same obstacle in three dimensions
``sphere``          a closed curved surface carrying shallow water -- geometry is curvature

Each family defines its own *geometry-blind* operator, meaning the basis a
practitioner would use having ignored the geometry: the same operator with the
walls removed, a triangulation that connects straight across the holes, and on
the sphere the separable cosine basis of the latitude-longitude rectangle, which
is how spherical fields are usually factorized and which is singular at the
poles.  The proposed model and its blind counterpart then differ in exactly one
thing, on an identical node set, with an identical decoder.

Everything is assembled sparsely and solved with shift-invert Lanczos, so the
meshes are thousands of nodes rather than hundreds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.spatial import Delaunay

from .bases import BasisSpec
from .irregular_fem import (L_SHAPE, U_SHAPE, UNIT_SQUARE, Hole, Polygon,
                            build_mesh)
from .irregular_green_data import ArcWall, Wall
from .joint_diffusion_2d import GroupedFieldDataset
from .manifold_barrier_data import Cap, Partition
from .operator_diagnostics import (mass_orthonormalize_columns, sparse_eigenpairs)
from .simplex_fem import (SimplexMesh, assemble_advection_sparse,
                          assemble_sparse, build_box_mesh, build_sphere_mesh)


@dataclass(frozen=True)
class Family:
    """How one geometry family builds its mesh, its truth and its blind control.

    ``basis_cutoff`` is per family rather than global, chosen before any fitting
    so that the *geometry-aware* approximation floor is comparable everywhere --
    about ``0.02`` of the field's energy.  One global number would not do: at
    sixteen columns that floor is `0.022` on a plane, `0.101` on a sphere and
    `0.138` in a volume, because a three-dimensional or oscillatory field simply
    has more content to represent.  Holding the *truncation* fixed while the
    approximation quality varies sevenfold would compare truncation, not
    geometry.  Both bases in a comparison always receive the same number of
    columns.
    """

    name: str
    dimension: int
    dynamics: str
    resolution: int
    layouts: dict
    basis_cutoff: int = 16


# A hundred-to-one contrast between barrier and medium.  The choice is physical
# rather than numerical: push it further and the barrier's own relaxation becomes
# slower than the dynamics being observed, at which point the barrier interior is
# a separate slow subsystem and any truncated basis has to spend its leading
# modes describing it instead of the field outside.  That regime is measured and
# reported, but it is not the regime the method is for.
BARRIER = 1e-2

PLANE_BARRIERS = {
    "open": (),
    "labyrinth": (Wall((.34, .38), (0., .70), BARRIER),
                  Wall((.62, .66), (.30, 1.), BARRIER)),
    "arc": (ArcWall((.5, .5), .30, .022, gap=(1.25, 1.90),
                    conductivity=BARRIER),),
    "chamber": (Wall((.48, .52), (0., .42), BARRIER),
                Wall((.48, .52), (.58, 1.), BARRIER)),
    "sealed_4": (Wall((.48, .52), (0., 1.), BARRIER),
                 Wall((0., 1.), (.48, .52), BARRIER)),
}

PLANE_DOMAINS = {
    "square": (UNIT_SQUARE, ()),
    "center_hole": (UNIT_SQUARE, (Hole((.5, .5), .20),)),
    "two_holes": (UNIT_SQUARE, (Hole((.32, .62), .15), Hole((.68, .33), .13))),
    "L_shape": (L_SHAPE, ()),
    "U_shape": (U_SHAPE, ()),
}

VOLUME_BARRIERS = {
    "open": (),
    "window": (Partition(0, .5, window=((.35, .65), (.35, .65)),
                         conductivity=BARRIER),),
    "chamber": (Partition(0, .5, window=((.44, .56), (.44, .56)),
                          conductivity=BARRIER),),
    "sealed_8": (Partition(0, .5, conductivity=BARRIER),
                 Partition(1, .5, conductivity=BARRIER),
                 Partition(2, .5, conductivity=BARRIER)),
}

# A bare sphere, and only that.  Here the geometry under test is the manifold
# itself: there is no obstacle to know, so the proposed model's operator is
# simply the Laplace-Beltrami operator of the surface, and the control is the
# latitude-longitude product basis that spherical data is normally factorized
# with.  Adding land makes the comparison *worse*, not better -- a cap smaller
# than the basis wavelength costs the operator more capacity than it returns --
# and that negative result is recorded in the iteration log rather than tuned
# away.
SPHERE_LAYOUTS = {"open_ocean": ()}

# One resolution per family, chosen once so that the barrier is resolved by
# several elements and the mesh is large enough to be worth reporting, then left
# alone.  Sweeping it is not part of the claim.
def _floorplan_layouts():
    from .floorplan import LAYOUTS
    return LAYOUTS


FAMILIES = {
    "plane_barrier": Family("plane_barrier", 2, "diffusion", 80, PLANE_BARRIERS, 16),
    "plane_domain": Family("plane_domain", 2, "diffusion", 80, PLANE_DOMAINS, 16),
    "volume_barrier": Family("volume_barrier", 3, "diffusion", 20, VOLUME_BARRIERS, 48),
    "sphere": Family("sphere", 2, "wave", 5, SPHERE_LAYOUTS, 32),
    # Rooms and doorways: the family the Peclet sweep says should work, on a
    # geometry that looks like something rather than like a test case.
    # Resolution 130 puts the element size (0.093 m) below the wall thickness
    # (0.12 m).  Below that the walls leak and the advantage is *understated*
    # -- the opposite direction from the sub-element barriers of Iteration 11,
    # and the same lesson: an unresolved obstacle gives an untrustworthy number
    # whose sign you cannot even predict.  The ladder is converged from here:
    # 5.2/9.9/13.9 at 130 against 5.1/10.2/13.8 at 240.
    "floorplan": Family("floorplan", 2, "diffusion", 130, _floorplan_layouts(), 16),
}


def _material(centroids: torch.Tensor, obstacles: tuple, contrast: float, *,
              background: bool) -> torch.Tensor:
    """Diffusivity seen by the truth or by the learner.

    Both agree inside the obstacles, because their layout is the geometric
    information the method is allowed to use.  Only the truth sees the smooth
    background variation, which keeps this a geometry prior and not a
    known-physics prior.
    """
    material = torch.ones(len(centroids), dtype=torch.float64)
    if background and contrast:
        weights = torch.tensor([1.7, 2.3, 1.1][:centroids.shape[1]],
                               dtype=torch.float64)
        phase = (centroids.double() * weights).sum(1)
        material = torch.exp(contrast * torch.sin(2 * math.pi * phase) * .5)
    for obstacle in obstacles:
        material = torch.where(obstacle.contains(centroids),
                               torch.full_like(material, obstacle.conductivity),
                               material)
    return material


def _initial_states(coordinates: torch.Tensor, count: int, seed: int, *,
                    on_sphere: bool) -> torch.Tensor:
    """Smooth localized bumps, one per scenario.

    Bumps rather than draws from the operator's own eigenbasis, so a truncated
    learner basis has to face energy it cannot represent.  On the sphere the
    bumps are geodesic caps, because sampling centres in the ambient box puts
    most of them inside the ball and produces high-frequency rings instead.
    """
    generator = torch.Generator().manual_seed(seed)
    points = coordinates.double()
    dimension = points.shape[1]
    low, span = points.min(0).values, points.max(0).values - points.min(0).values
    states = []
    for _ in range(count):
        field = torch.zeros(len(points), dtype=torch.float64)
        for _ in range(3):
            if on_sphere:
                direction = torch.randn(dimension, generator=generator).double()
                centre = direction / direction.norm().clamp_min(1e-12)
                unit = points / points.norm(dim=1, keepdim=True).clamp_min(1e-12)
                distance = torch.arccos((unit @ centre).clamp(-1., 1.))
                width = .45 + .35 * float(torch.rand(1, generator=generator))
            else:
                centre = low + span * torch.rand(dimension, generator=generator).double()
                distance = (points - centre).norm(dim=1)
                width = .14 + .12 * float(torch.rand(1, generator=generator))
            field = field + float(torch.randn(1, generator=generator)) * torch.exp(
                -distance ** 2 / (2 * width ** 2))
        states.append(field - field.mean())
    return torch.stack(states)


def _lat_lon_basis(coordinates: torch.Tensor, mass, count: int):
    """Separable cosine modes of the ``(theta, phi)`` rectangle.

    The geometry-blind control on a sphere.  It is what spherical fields are
    usually factorized with, and it cannot represent a field that is smooth
    across a pole, because every longitude meets there.  No amount of extra rank
    repairs that, which is what makes it the right control.
    """
    direction = coordinates.double()
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    polar = torch.arccos(direction[:, 2].clamp(-1., 1.))
    azimuth = torch.atan2(direction[:, 1], direction[:, 0]) % (2 * math.pi)
    limit = int(math.ceil(math.sqrt(count))) + 3
    pairs = torch.cartesian_prod(torch.arange(limit), torch.arange(limit))
    pairs = pairs[torch.argsort((pairs.double() ** 2).sum(1), stable=True)[:count]]
    columns = (torch.cos(polar[:, None] * pairs[:, 0].double()[None, :])
               * torch.cos(azimuth[:, None] * pairs[:, 1].double()[None, :]))
    return mass_orthonormalize_columns(columns, mass), (pairs.double() ** 2).sum(1)


def _flat_chart_basis(coordinates: torch.Tensor, mass, count: int):
    """Separable cosine modes of the bounding box, in any dimension."""
    dimension = coordinates.shape[1]
    limit = int(math.ceil(count ** (1 / dimension))) + 3
    grid = torch.cartesian_prod(*[torch.arange(limit)] * dimension)
    grid = grid[torch.argsort((grid.double() ** 2).sum(1), stable=True)[:count]]
    low = coordinates.min(0).values.double()
    span = (coordinates.max(0).values.double() - low).clamp_min(1e-9)
    scaled = (coordinates.double() - low) / span
    columns = torch.ones(len(coordinates), len(grid), dtype=torch.float64)
    for dim in range(dimension):
        columns = columns * torch.cos(math.pi * scaled[:, dim:dim + 1]
                                      * grid[:, dim].double())
    return mass_orthonormalize_columns(columns, mass), (grid.double() ** 2).sum(1)


def _erased_triangulation(coordinates: torch.Tensor) -> SimplexMesh:
    """Retriangulate the same nodes as if no hole or notch were there.

    Restricting the full-square operator to a sub-block would not do: dropping
    the interactions that pass through a hole would quietly hand the control most
    of the geometry it is supposed to lack.  Delaunay over the free nodes keeps
    every triangle, including those spanning an obstacle.
    """
    points = coordinates.double().numpy()
    cells = Delaunay(points).simplices
    return SimplexMesh(torch.from_numpy(points).double(),
                       torch.from_numpy(cells.astype(np.int64)), "erased")


def build_family(family: str, layout: str, *, resolution: int | None = None,
                 n_scenarios: int = 20, n_time: int = 16,
                 basis_cutoff: int | None = None,
                 truth_modes: int = 60, contrast: float = .3,
                 reaction: float = .15, time_span: tuple[float, float] = (.15, 3.),
                 wave_speed: float = 1., damping: float = .05,
                 mesh_seed: int = 0, scenario_seed: int = 7717,
                 permutation_seed: int = 9173) -> GroupedFieldDataset:
    """``Y(scenario, time, node)`` for one layout of one geometry family."""
    spec = FAMILIES[family]
    resolution = spec.resolution if resolution is None else resolution
    basis_cutoff = spec.basis_cutoff if basis_cutoff is None else basis_cutoff
    if family == "floorplan":
        from .floorplan import floorplan_tensor
        return floorplan_tensor(layout, resolution=resolution,
                                n_scenarios=n_scenarios, n_time=n_time,
                                basis_cutoff=basis_cutoff,
                                truth_modes=truth_modes, contrast=contrast,
                                time_span=time_span, mesh_seed=mesh_seed,
                                scenario_seed=scenario_seed,
                                permutation_seed=permutation_seed)
    obstacles: tuple = ()
    polygon: Polygon | None = None

    if family == "plane_barrier":
        obstacles = spec.layouts[layout]
        planar = build_mesh(resolution, (), polygon=UNIT_SQUARE, seed=mesh_seed)
        mesh = SimplexMesh(planar.nodes, planar.triangles, "triangles")
        blind_mesh = mesh
    elif family == "plane_domain":
        polygon, holes = spec.layouts[layout]
        planar = build_mesh(resolution, holes, polygon=polygon, seed=mesh_seed)
        mesh = SimplexMesh(planar.nodes, planar.triangles, "triangles")
        blind_mesh = _erased_triangulation(planar.nodes)
    elif family == "volume_barrier":
        obstacles = spec.layouts[layout]
        mesh = build_box_mesh(resolution, seed=mesh_seed)
        blind_mesh = mesh
    elif family == "sphere":
        obstacles = spec.layouts[layout]
        mesh = build_sphere_mesh(resolution)
        blind_mesh = mesh
    else:
        raise ValueError(f"unknown family {family!r}")

    centroids = mesh.centroids()
    coordinates = mesh.nodes
    truth_stiffness, mass = assemble_sparse(
        mesh, _material(centroids, obstacles, contrast, background=True))
    aware_stiffness, _ = assemble_sparse(
        mesh, _material(centroids, obstacles, 0., background=False))

    def apply_mass(x):
        return torch.from_numpy(mass @ x.double().numpy()).double()

    truth_modes = min(truth_modes, mesh.n_nodes - 1)
    truth_values, truth_vectors = sparse_eigenpairs(truth_stiffness, mass,
                                                    truth_modes)
    rates = truth_values / math.pi ** 2
    initial = _initial_states(coordinates, n_scenarios, scenario_seed,
                              on_sphere=family == "sphere")
    time = torch.linspace(time_span[0], time_span[1], n_time, dtype=torch.float64)
    if spec.dynamics == "diffusion":
        propagator = torch.exp(-time[:, None] * (reaction + rates[None, :]))
    else:
        propagator = (torch.exp(-damping * time[:, None])
                      * torch.cos(time[:, None]
                                  * (wave_speed * rates.clamp_min(0.).sqrt())[None, :]))
    values = torch.einsum("sq,tq,nq->stn", initial @ apply_mass(truth_vectors),
                          propagator, truth_vectors)
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    aware_values, aware_basis = sparse_eigenpairs(aware_stiffness, mass,
                                                 basis_cutoff)
    if family == "sphere":
        # No barrier-free operator exists on a bare sphere -- the geometry *is*
        # the manifold -- so the blind control is the chart a per-axis method
        # would impose on it.
        blind_basis, blind_values = _lat_lon_basis(coordinates, mass, basis_cutoff)
    else:
        blind_stiffness, blind_mass = assemble_sparse(blind_mesh, 1.)
        blind_values, blind_basis = sparse_eigenpairs(blind_stiffness, blind_mass,
                                                     basis_cutoff)
        blind_values = blind_values / math.pi ** 2
    chart_basis, chart_values = _flat_chart_basis(coordinates, mass, basis_cutoff)
    permutation = torch.randperm(
        mesh.n_nodes, generator=torch.Generator().manual_seed(permutation_seed))

    time_cutoff = min(basis_cutoff, n_time)
    reference = (aware_values / math.pi ** 2)[:time_cutoff]
    if spec.dynamics == "diffusion":
        curves = torch.exp(-time[:, None] * (reaction + reference[None, :]))
    else:
        curves = (torch.exp(-damping * time[:, None])
                  * torch.cos(time[:, None]
                              * (wave_speed * reference.clamp_min(0.).sqrt())[None, :]))
    time_basis = torch.linalg.qr(curves, mode="reduced").Q[:, :time_cutoff]

    spatial = {
        "geometry_operator": (aware_basis, aware_values / math.pi ** 2),
        "blind_operator": (blind_basis, blind_values),
        "flat_chart": (chart_basis, chart_values[:chart_basis.shape[1]]),
        "permuted": (aware_basis[permutation], aware_values / math.pi ** 2),
    }
    matrices = {"coordinates": coordinates.float(),
                "time_basis": time_basis.float(),
                "time_eigenvalues": reference.float(),
                # The mesh the basis was assembled on.  Each basis column is a
                # P1 function on it, so keeping the mesh is what lets the fitted
                # factor be evaluated anywhere in the domain and not only at the
                # nodes (see GroupedOperatorTucker.predict_at).
                "mesh_nodes": mesh.nodes.float(),
                "mesh_cells": mesh.cells}
    for name, (basis, eigenvalues) in spatial.items():
        matrices[f"{name}_basis"] = basis.float()
        matrices[f"{name}_eigenvalues"] = eigenvalues[:basis.shape[1]].float()

    metadata = {
        "family": family, "layout": layout, "resolution": int(resolution),
        "ambient_dimension": int(coordinates.shape[1]),
        "manifold_dimension": int(mesh.degree), "dynamics": spec.dynamics,
        "pde": ("d_t u + (-div(a grad u) + reaction I) u = 0"
                if spec.dynamics == "diffusion" else
                "d_tt eta = div(g H grad eta) - damping d_t eta (shallow water)"),
        "tensor_semantics": "scenario x time x node",
        "mesh_hash": mesh.hash(), "n_nodes": int(mesh.n_nodes),
        "n_cells": int(len(mesh.cells)), "domain_measure": mesh.volume(),
        "blind_operator_is": ("lat-lon separable chart" if family == "sphere"
                              else "same nodes, geometry removed"),
        "obstacles": [{k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in vars(o).items()} for o in obstacles],
        "polygon_vertices": (list(polygon.vertices) if polygon else None),
        "operator_information_tier":
            "geometry (mesh and obstacle layout known; background material unknown)",
        "log_diffusivity_contrast": float(contrast), "reaction": float(reaction),
        "basis_cutoff": int(basis_cutoff), "time_cutoff": int(time_cutoff),
        "truth_modes": int(truth_modes), "n_scenarios": int(n_scenarios),
        "scenario_seed": int(scenario_seed),
        "time_span": [float(time_span[0]), float(time_span[1])],
        "permutation_seed": int(permutation_seed),
        "coordinate_groups": [[0], [1], [2]],
    }
    specs = tuple(BasisSpec("neumann", max(1, basis_cutoff - 1), name)
                  for name in ("scenario", "time", "node"))
    return GroupedFieldDataset(
        f"{family}_{layout}_r{resolution}_k{basis_cutoff}", values.float(),
        ("scenario", "time", "node"), specs, (False, False, False),
        "generated:geoaware.benchmark.build_family",
        f"{spec.dynamics} field on a {family} geometry; all layouts in a family "
        "share one node set and differ only in the geometry the operator sees.",
        metadata=metadata, groups=((0,), (1,), (2,)), operator_matrices=matrices)


def _cellular_velocity(centroids: torch.Tensor) -> torch.Tensor:
    """A divergence-free cellular flow, tangential to the outer boundary."""
    x, y = centroids[:, 0], centroids[:, 1]
    return torch.stack([torch.sin(math.pi * x) * torch.cos(math.pi * y),
                        -torch.cos(math.pi * x) * torch.sin(math.pi * y)], 1)


def build_advected_barrier(layout: str, peclet: float, *, resolution: int = 80,
                           n_scenarios: int = 12, n_time: int = 12,
                           basis_cutoff: int = 16, contrast: float = .3,
                           reaction: float = .15,
                           time_span: tuple[float, float] = (.15, 3.),
                           steps: int = 600, mesh_seed: int = 0,
                           scenario_seed: int = 7717,
                           permutation_seed: int = 9173) -> GroupedFieldDataset:
    """The barrier family, with transport added to the truth and not to the prior.

    The learner assembles the same geometry-aware Laplacian at every Peclet
    number; only the physics generating the data changes.  That isolates a
    question the rest of the benchmark cannot ask: the method is given the right
    geometry but an increasingly wrong *operator class*, and the point at which
    that stops being survivable is what two real datasets ran into.

    The velocity vanishes inside the barriers, because a wall has no flow
    through it.  Without that, transport would simply carry the field across the
    obstacles and the geometry would stop being real -- which is a different
    failure and would confound the measurement.

    The truth is time-stepped with Crank-Nicolson rather than expanded in
    eigenfunctions, since an advection-diffusion operator is not symmetric.
    """
    from scipy.sparse.linalg import splu

    planar = build_mesh(resolution, (), polygon=UNIT_SQUARE, seed=mesh_seed)
    mesh = SimplexMesh(planar.nodes, planar.triangles, "triangles")
    centroids = mesh.centroids()
    obstacles = PLANE_BARRIERS[layout]
    truth_stiffness, mass = assemble_sparse(
        mesh, _material(centroids, obstacles, contrast, background=True))
    aware_stiffness, _ = assemble_sparse(
        mesh, _material(centroids, obstacles, 0., background=False))
    blind_stiffness, _ = assemble_sparse(mesh, 1.)

    velocity = _cellular_velocity(centroids) * peclet
    solid = torch.zeros(len(centroids), dtype=torch.bool)
    for obstacle in obstacles:
        solid |= obstacle.contains(centroids)
    velocity[solid] = 0.
    advection = assemble_advection_sparse(mesh, velocity)

    time = torch.linspace(time_span[0], time_span[1], n_time, dtype=torch.float64)
    initial = _initial_states(mesh.nodes, n_scenarios, scenario_seed,
                              on_sphere=False)
    step = float(time_span[1]) / steps
    generator = truth_stiffness + advection + reaction * mass
    solver = splu((mass + .5 * step * generator).tocsc())
    explicit = (mass - .5 * step * generator).tocsc()
    state = initial.numpy().T.copy()
    wanted = np.clip((time.numpy() / step).round().astype(int), 1, steps)
    frames, cursor = [], 0
    for index in range(1, steps + 1):
        state = solver.solve(explicit @ state)
        while cursor < n_time and wanted[cursor] == index:
            frames.append(state.copy())
            cursor += 1
    values = torch.from_numpy(np.stack(frames, 0)).permute(2, 0, 1).double()
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    aware_values, aware_basis = sparse_eigenpairs(aware_stiffness, mass,
                                                 basis_cutoff)
    blind_values, blind_basis = sparse_eigenpairs(blind_stiffness, mass,
                                                 basis_cutoff)
    chart_basis, chart_values = _flat_chart_basis(mesh.nodes, mass, basis_cutoff)
    permutation = torch.randperm(
        mesh.n_nodes, generator=torch.Generator().manual_seed(permutation_seed))
    scale = math.pi ** 2
    # No operator is claimed for the time axis here either: with transport the
    # decay curves of the nominal operator are no longer the right family.
    time_basis = torch.linalg.qr(
        torch.stack([time ** k for k in range(min(basis_cutoff, n_time))], 1),
        mode="reduced").Q

    spatial = {
        "geometry_operator": (aware_basis, aware_values / scale),
        "blind_operator": (blind_basis, blind_values / scale),
        "flat_chart": (chart_basis, chart_values[:chart_basis.shape[1]]),
        "permuted": (aware_basis[permutation], aware_values / scale),
    }
    matrices = {"coordinates": mesh.nodes.float(),
                "time_basis": time_basis.float(),
                "time_eigenvalues": torch.arange(
                    time_basis.shape[1], dtype=torch.float32)}
    for name, (basis, eigenvalues) in spatial.items():
        matrices[f"{name}_basis"] = basis.float()
        matrices[f"{name}_eigenvalues"] = eigenvalues[:basis.shape[1]].float()

    metadata = {
        "family": "advected_barrier", "layout": layout,
        "peclet": float(peclet), "resolution": int(resolution),
        "pde": "d_t u + b . grad u - div(a grad u) + reaction u = 0",
        "advection": "divergence-free cellular flow, zero inside the barriers",
        "time_integration": f"Crank-Nicolson, {steps} steps",
        "tensor_semantics": "scenario x time x node",
        "mesh_hash": mesh.hash(), "n_nodes": int(mesh.n_nodes),
        "operator_information_tier":
            "geometry known; operator class increasingly wrong",
        "basis_cutoff": int(basis_cutoff), "n_scenarios": int(n_scenarios),
        "time_span": [float(time_span[0]), float(time_span[1])],
        "coordinate_groups": [[0], [1], [2]],
    }
    specs = tuple(BasisSpec("neumann", max(1, basis_cutoff - 1), name)
                  for name in ("scenario", "time", "node"))
    return GroupedFieldDataset(
        f"advected_{layout}_pe{peclet:g}_k{basis_cutoff}", values.float(),
        ("scenario", "time", "node"), specs, (False, False, False),
        "generated:geoaware.benchmark.build_advected_barrier",
        "Advection-diffusion past thin barriers; the learner's operator knows "
        "the geometry but not the transport.",
        metadata=metadata, groups=((0,), (1,), (2,)), operator_matrices=matrices)


def build_operator_variant(family: str, layout: str, dynamics: str, *,
                           resolution: int | None = None, n_scenarios: int = 12,
                           n_time: int = 12, basis_cutoff: int | None = None,
                           truth_modes: int = 60, contrast: float = .3,
                           reaction: float = .15, wave_speed: float = 1.,
                           damping: float = .05,
                           time_span: tuple[float, float] = (.15, 3.),
                           mesh_seed: int = 0, scenario_seed: int = 7717,
                           permutation_seed: int = 9173) -> GroupedFieldDataset:
    """The same geometry under a different equation.

    Every planar family in this benchmark relaxes.  The obvious question is
    whether the geometry prior is really a statement about geometry or a
    statement about diffusion, and the way to answer it is to keep the mesh, the
    barriers, the learner and its basis fixed and change only the propagator.

    ``dynamics="wave"`` replaces ``exp(-lambda t)`` with a damped
    ``cos(c sqrt(lambda) t)``.  That matters beyond breadth: a parabolic field
    collapses onto the operator's slowest eigenvectors, which is the degeneracy
    R5a flagged, while an oscillatory one keeps energy across the spectrum for
    all time and is strictly less generous to a truncated basis.
    """
    spec = FAMILIES[family]
    if family == "floorplan":
        raise ValueError("use the planar families; the floor plan carries its own"
                         " builder")
    original = spec.dynamics
    try:
        FAMILIES[family] = dataclasses.replace(spec, dynamics=dynamics)
        data = build_family(family, layout, resolution=resolution,
                            n_scenarios=n_scenarios, n_time=n_time,
                            basis_cutoff=basis_cutoff, truth_modes=truth_modes,
                            contrast=contrast, reaction=reaction,
                            wave_speed=wave_speed, damping=damping,
                            time_span=time_span, mesh_seed=mesh_seed,
                            scenario_seed=scenario_seed,
                            permutation_seed=permutation_seed)
    finally:
        FAMILIES[family] = dataclasses.replace(spec, dynamics=original)
    data.metadata["dynamics_variant"] = dynamics
    return data
