"""Green-response tensor on a square with holes, and the bases to compare on it.

The tensor has exactly the semantics of the frozen 1-D benchmark —
``Y(t, receiver, source)`` for a diffusion Green kernel — with one change: the
spatial coordinate is a 2-D mesh with an irregular boundary, so both spatial
axes index the *same* set of mesh nodes.  Readers therefore only have to learn
one setting for the whole paper.

Four spatial bases are produced on the **identical node set**, which is what
makes the comparison auditable without any interpolation:

``fem_correct``
    Generalized eigenvectors of the mesh operator that respects the outer
    boundary and the hole rims.  This is the proposed geometry-aware subspace.
``topology_erased``
    The same nodes and the same finite-element machinery, but triangulated
    across the holes and with the rim condition dropped, i.e. a model that
    knows the mesh and not the topology.
``bounding_box_product``
    Analytic separable cosine modes of the enclosing square, sampled at the
    node coordinates.  This is what an ordinary tensor method does: it treats
    the domain as a rectangle and never sees the obstacle.
``permuted``
    ``fem_correct`` with its rows shuffled.  Destructive control: same columns,
    same eigenvalues, same parameter count, no index-operator alignment.

The learner never sees the material coefficient.  Truth uses a variable
diffusivity; every learner basis is built from the constant-coefficient
operator, so geometry is known metadata while the material is not.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .bases import BasisSpec
from .irregular_fem import (UNIT_SQUARE, Hole, IrregularMesh, Polygon,
                            assemble_p1, build_mesh, free_nodes,
                            interpolate_p1, mesh_metadata, restrict,
                            triangle_centroids)
from .joint_diffusion_2d import GroupedFieldDataset
from .operator_diagnostics import generalized_eigenpairs

DEFAULT_HOLES = (Hole((.32, .62), .15), Hole((.68, .33), .13))


@dataclass(frozen=True)
class Wall:
    """A thin impermeable baffle: an axis-aligned rectangle of near-zero material.

    Walls are the sharpest way to make geometry matter.  Two points a millimetre
    apart on opposite sides of a baffle are far apart for the physics, so any
    model that reasons in Euclidean coordinates — a cosine basis of the bounding
    box, a coordinate network — smooths straight through the barrier, while an
    operator assembled with the wall in place does not.

    Walls stay inside one fixed mesh instead of being cut out of it, so every
    geometry variant shares an identical node set by construction and no control
    needs an interpolation step.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    conductivity: float = 1e-3

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        x, y = points[:, 0], points[:, 1]
        return ((x >= self.x_range[0]) & (x <= self.x_range[1])
                & (y >= self.y_range[0]) & (y <= self.y_range[1]))


@dataclass(frozen=True)
class ArcWall:
    """A curved baffle: an annular band with an angular aperture.

    An axis-aligned slab is the easiest possible barrier for a model that reads
    raw coordinates — a single tanh unit reproduces a step at ``x = c``.  A
    curved barrier is not separable in ``x`` and ``y`` and has no low-order
    coordinate description, so it separates "knows the geometry" from "has
    enough capacity to memorize where the jump is".
    """

    center: tuple[float, float]
    radius: float
    half_width: float
    gap: tuple[float, float] = (0., 0.)
    conductivity: float = 1e-3

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        offset = points - torch.tensor(self.center, dtype=points.dtype)
        distance = offset.norm(dim=1)
        angle = torch.atan2(offset[:, 1], offset[:, 0]) % (2 * math.pi)
        low, high = self.gap
        in_gap = ((angle >= low) & (angle <= high) if low <= high
                  else (angle >= low) | (angle <= high))
        return ((distance - self.radius).abs() < self.half_width) & ~in_gap


WALL_LAYOUTS = {
    # No barrier: the control where there is no geometry to know.
    "open": (),
    # One baffle rising from the floor, leaving a gap at the top.
    "single_baffle": (Wall((.48, .52), (0., .72)),),
    # Two staggered baffles: the field must snake around them.
    "labyrinth": (Wall((.34, .38), (0., .70)), Wall((.62, .66), (.30, 1.))),
    # Two chambers joined by a narrow aperture in the middle of the wall.
    "chamber": (Wall((.48, .52), (0., .42)), Wall((.48, .52), (.58, 1.))),
    # Fully sealed quadrants: the strongest case, where geometry decides which
    # regions can interact at all rather than merely how fast.
    "sealed_4": (Wall((.48, .52), (0., 1.)), Wall((0., 1.), (.48, .52))),
    # Curved barriers: no low-order description in raw coordinates.
    "arc": (ArcWall((.5, .5), .30, .022, gap=(1.25, 1.90)),),
    "double_arc": (ArcWall((.5, .5), .17, .022, gap=(.35, 1.00)),
                   ArcWall((.5, .5), .36, .022, gap=(3.50, 4.15))),
}


def wall_coefficient(centroids: torch.Tensor, walls: tuple,
                     contrast: float, background: bool = True) -> torch.Tensor:
    """Material seen by the truth (``background``) or by the learner.

    The learner is told where the walls are — that is the geometry it is
    supposed to exploit — but not the smooth background variation, which is what
    keeps this a geometry prior rather than an exact-physics prior.
    """
    values = (_variable_diffusivity(centroids, contrast) if background
              else torch.ones(len(centroids), dtype=torch.float64))
    for wall in walls:
        values = torch.where(wall.contains(centroids),
                             torch.full_like(values, wall.conductivity), values)
    return values


def _variable_diffusivity(centroids: torch.Tensor, contrast: float) -> torch.Tensor:
    """Material the truth uses and the learner never reads."""
    x, y = centroids[:, 0], centroids[:, 1]
    return torch.exp(contrast * (torch.cos(2 * math.pi * x)
                                 + .35 * torch.sin(3 * math.pi * y + .37)))


def _mass_orthonormalize(basis: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Orthonormalize columns in the mass inner product, dropping dependencies."""
    from .operator_diagnostics import inverse_sqrt, matrix_sqrt

    whitened = matrix_sqrt(mass) @ basis.double()
    q, r = torch.linalg.qr(whitened, mode="reduced")
    keep = r.diagonal().abs() > 1e-9 * r.diagonal().abs().max().clamp_min(1e-30)
    return inverse_sqrt(mass) @ q[:, keep]


def _bounding_box_product_basis(coordinates: torch.Tensor, mass: torch.Tensor,
                                count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Separable Neumann cosine modes of the enclosing square at node positions.

    Modes are ranked by ``k^2 + l^2``, the eigenvalue of the square's Laplacian,
    so this is exactly the basis a per-axis method would pick if it believed the
    domain were a rectangle.
    """
    limit = int(math.ceil(math.sqrt(count))) + 3
    pairs = torch.cartesian_prod(torch.arange(limit), torch.arange(limit))
    order = torch.argsort((pairs.double() ** 2).sum(1), stable=True)[:count]
    pairs = pairs[order]
    x, y = coordinates[:, 0:1].double(), coordinates[:, 1:2].double()
    columns = (torch.cos(math.pi * x * pairs[:, 0].double())
               * torch.cos(math.pi * y * pairs[:, 1].double()))
    eigenvalues = (pairs.double() ** 2).sum(1)
    return _mass_orthonormalize(columns, mass), eigenvalues


def _topology_erased_operator(coordinates: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """P1 operator on the same nodes, triangulated as if no hole existed."""
    from scipy.spatial import Delaunay

    triangulation = Delaunay(coordinates.double().numpy())
    mesh = IrregularMesh(
        coordinates.double(),
        torch.from_numpy(triangulation.simplices.astype("int64")),
        torch.zeros(len(coordinates), dtype=torch.bool),
        torch.zeros(len(coordinates), dtype=torch.bool),
        -torch.ones(len(coordinates), dtype=torch.long), (), 0)
    return assemble_p1(mesh, 1.)


def _farthest_point_sources(coordinates: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic spread-out source nodes; no randomness to record."""
    if count > len(coordinates):
        raise ValueError("more sources requested than free nodes")
    chosen = [int(torch.argmin(coordinates.sum(1)))]
    distance = (coordinates - coordinates[chosen[0]]).norm(dim=1)
    while len(chosen) < count:
        nxt = int(torch.argmax(distance))
        chosen.append(nxt)
        distance = torch.minimum(distance, (coordinates - coordinates[nxt]).norm(dim=1))
    return torch.tensor(sorted(chosen), dtype=torch.long)


def _normalized_rates(values: torch.Tensor) -> torch.Tensor:
    """Eigenvalues on a fixed geometric scale, not a per-geometry one.

    Dividing by ``pi^2`` — the fundamental scale of the unit square — keeps
    decay rates comparable across hole layouts.  Rescaling by a per-geometry
    quantity such as ``lambda_2 - lambda_1`` would silently change how much
    high-frequency content each geometry's field carries, which would make the
    task difficulty, not the basis, responsible for any difference.
    """
    return values / (math.pi ** 2)


def _smooth_initial_states(coordinates: torch.Tensor, count: int,
                           seed: int) -> torch.Tensor:
    """Localized smooth initial conditions, one per scenario.

    Gaussian bumps rather than draws from the operator's own eigenbasis: a
    truncated learner basis has to face energy it cannot represent, otherwise
    the projection residual would be an artifact of the generator.
    """
    generator = torch.Generator().manual_seed(seed)
    low = coordinates.min(0).values
    span = (coordinates.max(0).values - low)
    states = []
    for _ in range(count):
        field = torch.zeros(len(coordinates), dtype=torch.float64)
        for _ in range(3):
            center = low + span * torch.rand(2, generator=generator).double()
            width = .10 + .10 * float(torch.rand(1, generator=generator))
            amplitude = float(torch.randn(1, generator=generator))
            field = field + amplitude * torch.exp(
                -((coordinates.double() - center) ** 2).sum(1) / (2 * width ** 2))
        states.append(field - field.mean())
    return torch.stack(states)


def irregular_field_tensor(
        holes: tuple[Hole, ...] = DEFAULT_HOLES, *,
        polygon: Polygon = UNIT_SQUARE, resolution: int = 16,
        n_scenarios: int = 20, n_time: int = 16, basis_cutoff: int = 32,
        truth_modes: int = 60, contrast: float = .3, reaction: float = .15,
        time_span: tuple[float, float] = (.15, 3.), mesh_seed: int = 0,
        scenario_seed: int = 7717, hole_condition: str = "neumann",
        permutation_seed: int = 9173,
        truth_refinement: int = 1) -> GroupedFieldDataset:
    """``Y(scenario, time, node)``: the same PDE from different initial states.

    This is the plainest spatiotemporal setting there is — a family of
    simulations on one domain, observed at a few percent of its entries — and it
    needs no source/receiver semantics to explain.  It shares the mesh, the
    operator, the four spatial bases and the leakage rules with
    :func:`irregular_green_tensor`, so the two settings differ only in what the
    non-spatial modes mean.
    """
    green = irregular_green_tensor(
        holes, polygon=polygon, resolution=resolution, n_time=n_time,
        n_sources=min(n_scenarios, 8), basis_cutoff=basis_cutoff,
        truth_modes=truth_modes, contrast=contrast, reaction=reaction,
        time_span=time_span, mesh_seed=mesh_seed, hole_condition=hole_condition,
        permutation_seed=permutation_seed)
    matrices = dict(green.operator_matrices)
    coordinates = matrices["coordinates"]
    truth_values = matrices["truth_eigenvalues"]
    truth_vectors = matrices["truth_eigenvectors"]
    mass = matrices["mass"].double()

    time = torch.linspace(time_span[0], time_span[1], n_time, dtype=torch.float64)
    if truth_refinement > 1:
        # Solve the truth on an independently seeded finer mesh of the *same*
        # domain and interpolate onto the learner's nodes, so the learner's
        # operator is a discretization of the continuum problem rather than the
        # generator of the data itself.  Both the geometry-aware and the
        # geometry-blind basis pay the identical interpolation error.
        fine = build_mesh((resolution - 1) * truth_refinement + 1, holes,
                          polygon=polygon, seed=mesh_seed + 977 * truth_refinement)
        fine_stiffness, fine_mass_full = assemble_p1(
            fine, _variable_diffusivity(triangle_centroids(fine), contrast))
        fine_keep = free_nodes(fine, hole_condition, "neumann")
        fine_mass = restrict(fine_mass_full, fine_keep)
        fine_values, fine_vectors = generalized_eigenpairs(
            restrict(fine_stiffness, fine_keep), fine_mass,
            min(truth_modes, len(fine_keep)))
        fine_coordinates = fine.nodes[fine_keep]
        initial = _smooth_initial_states(fine_coordinates, n_scenarios,
                                         scenario_seed)
        amplitudes = initial @ fine_mass @ fine_vectors
        decay = torch.exp(-time[:, None]
                          * (reaction + _normalized_rates(fine_values)[None, :]))
        on_free = torch.einsum("sq,tq,nq->stn", amplitudes, decay, fine_vectors)
        # Constrained rim nodes carry the Dirichlet value, so the field is
        # complete on the fine mesh before it is interpolated.
        full = torch.zeros(n_scenarios, n_time, fine.n_nodes, dtype=torch.float64)
        full[..., fine_keep] = on_free
        values = interpolate_p1(fine, full, coordinates.double())
    else:
        fine = None
        initial = _smooth_initial_states(coordinates, n_scenarios, scenario_seed)
        amplitudes = initial @ mass @ truth_vectors.double()
        decay = torch.exp(-time[:, None]
                          * (reaction + truth_values.double()[None, :]))
        values = torch.einsum("sq,tq,nq->stn", amplitudes, decay,
                              truth_vectors.double())
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    matrices.pop("source_nodes", None)
    metadata = dict(green.metadata)
    metadata.update({"tensor_semantics": "scenario x time x node",
                     "n_scenarios": int(n_scenarios),
                     "scenario_seed": int(scenario_seed),
                     "truth_refinement": int(truth_refinement),
                     "truth_mesh_hash": fine.hash() if fine is not None
                                        else metadata.get("mesh_hash"),
                     "truth_mesh_nodes": int(fine.n_nodes) if fine is not None
                                         else int(len(coordinates)),
                     "inverse_crime":
                         "avoided: truth solved on an independent finer mesh "
                         "and interpolated" if truth_refinement > 1 else
                         "present: truth and learner operators share one "
                         "discretization",
                     "coordinate_groups": [[0], [1], [2]]})
    metadata.pop("source_node_indices", None)
    specs = tuple(green.basis_specs[:1] * 3)
    return GroupedFieldDataset(
        green.name.replace("irregular_green", "irregular_field"), values.float(),
        ("scenario", "time", "node"), specs, (False, False, False),
        "generated:geoaware.irregular_green_data.irregular_field_tensor",
        "Diffusion field on a polygonal domain with obstacles, one slice per "
        "initial condition; the spatial mode indexes mesh nodes.",
        metadata=metadata, groups=((0,), (1,), (2,)),
        operator_matrices=matrices)


def wall_field_tensor(
        walls: tuple[Wall, ...] = (), *, resolution: int = 18,
        n_scenarios: int = 20, n_time: int = 16, basis_cutoff: int = 32,
        truth_modes: int = 60, contrast: float = .3, reaction: float = .15,
        time_span: tuple[float, float] = (.15, 3.), mesh_seed: int = 0,
        scenario_seed: int = 7717, permutation_seed: int = 9173,
        truth_refinement: int = 1) -> GroupedFieldDataset:
    """``Y(scenario, time, node)`` on one square mesh divided by thin baffles.

    Every wall layout uses the *same* mesh and the *same* nodes; only the
    material inside the baffles changes.  A control therefore differs from the
    proposed model in exactly one respect — whether its operator knows the
    barriers — with no confound from meshing, node ordering or interpolation.

    ``truth_refinement`` above one solves the truth on an independently seeded
    mesh that many times finer and interpolates it onto the learner's nodes.
    The learner's operator is then a discretization of the same continuum
    problem rather than the very object that generated the data, which is what
    an inverse-crime objection asks for.  Every basis — geometry-aware and
    geometry-blind alike — pays the same discretization error, so the
    comparison is unchanged.  The default of one reproduces the frozen tensors
    bit for bit.
    """
    mesh = build_mesh(resolution, (), polygon=UNIT_SQUARE, seed=mesh_seed)
    centroids = triangle_centroids(mesh)
    truth_material = wall_coefficient(centroids, walls, contrast, background=True)
    learner_material = wall_coefficient(centroids, walls, 0., background=False)
    truth_stiffness, mass = assemble_p1(mesh, truth_material)
    nominal_stiffness, _ = assemble_p1(mesh, learner_material)
    blind_stiffness, _ = assemble_p1(mesh, 1.)
    coordinates = mesh.nodes
    n_nodes = mesh.n_nodes
    truth_modes = min(truth_modes, n_nodes)

    if truth_refinement > 1:
        # A different mesh *and* a different jitter seed: refinement alone would
        # leave the fine nodes nested inside the coarse ones on the structured
        # part of the grid.
        truth_mesh = build_mesh((resolution - 1) * truth_refinement + 1, (),
                                polygon=UNIT_SQUARE,
                                seed=mesh_seed + 977 * truth_refinement)
        truth_operator, truth_mass = assemble_p1(
            truth_mesh, wall_coefficient(triangle_centroids(truth_mesh), walls,
                                         contrast, background=True))
    else:
        truth_mesh, truth_operator, truth_mass = mesh, truth_stiffness, mass
    truth_modes = min(truth_modes, truth_mesh.n_nodes)

    truth_values, truth_vectors = generalized_eigenpairs(
        truth_operator, truth_mass, truth_modes)
    rates = _normalized_rates(truth_values)
    initial = _smooth_initial_states(truth_mesh.nodes, n_scenarios, scenario_seed)
    amplitudes = initial @ truth_mass @ truth_vectors
    time = torch.linspace(time_span[0], time_span[1], n_time, dtype=torch.float64)
    decay = torch.exp(-time[:, None] * (reaction + rates[None, :]))
    values = torch.einsum("sq,tq,nq->stn", amplitudes, decay, truth_vectors)
    if truth_mesh is not mesh:
        values = interpolate_p1(truth_mesh, values, coordinates)
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    nominal_values, wall_basis = generalized_eigenpairs(
        nominal_stiffness, mass, basis_cutoff)
    blind_values, blind_basis = generalized_eigenpairs(
        blind_stiffness, mass, basis_cutoff)
    box_basis, box_values = _bounding_box_product_basis(
        coordinates, mass, basis_cutoff)
    permutation = torch.randperm(
        n_nodes, generator=torch.Generator().manual_seed(permutation_seed))
    time_cutoff = min(basis_cutoff, n_time)
    time_basis = torch.linalg.qr(
        torch.exp(-time[:, None] * (reaction
                                    + _normalized_rates(nominal_values)[:time_cutoff][None, :])),
        mode="reduced").Q[:, :time_cutoff]

    spatial = {
        "fem_correct": (wall_basis, _normalized_rates(nominal_values)),
        "topology_erased": (blind_basis, _normalized_rates(blind_values)),
        "bounding_box_product": (box_basis, box_values[:box_basis.shape[1]]),
        "permuted": (wall_basis[permutation], _normalized_rates(nominal_values)),
    }
    matrices = {
        "mass": mass.float(), "nominal_stiffness": nominal_stiffness.float(),
        # The same operator with the barriers removed.  Pairing it with the
        # geometry-aware one turns the comparison into a clean 2x2: geometry
        # known or not, crossed with spectral truncation or a free table under a
        # smoothness penalty.
        "blind_stiffness": blind_stiffness.float(),
        "coordinates": coordinates.float(), "time_basis": time_basis.float(),
        "time_eigenvalues": _normalized_rates(nominal_values)[:time_cutoff].float(),
    }
    for name, (basis, eigenvalues) in spatial.items():
        matrices[f"{name}_basis"] = basis.float()
        matrices[f"{name}_eigenvalues"] = eigenvalues[:basis.shape[1]].float()

    metadata = mesh_metadata(mesh, truth_stiffness, mass) | {
        "pde": "d_t u + (-div(a grad u) + reaction I) u = 0",
        "boundary_condition": "Neumann on the outer square",
        "tensor_semantics": "scenario x time x node",
        "walls": [{k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(w).items()} for w in walls],
        "wall_kinds": [type(w).__name__ for w in walls],
        "operator_information_tier":
            "geometry (mesh, boundary and barrier layout known; background material unknown)",
        "log_diffusivity_contrast": float(contrast), "reaction": float(reaction),
        "basis_cutoff": int(basis_cutoff), "time_cutoff": int(time_cutoff),
        "truth_modes": int(truth_modes), "n_free_nodes": int(n_nodes),
        "truth_refinement": int(truth_refinement),
        "truth_mesh_hash": truth_mesh.hash(),
        "truth_mesh_nodes": int(truth_mesh.n_nodes),
        "inverse_crime":
            "avoided: truth solved on an independent finer mesh and interpolated"
            if truth_refinement > 1 else
            "present: truth and learner operators share one discretization",
        "n_scenarios": int(n_scenarios), "scenario_seed": int(scenario_seed),
        "time_span": [float(time_span[0]), float(time_span[1])],
        "permutation_seed": int(permutation_seed),
        "coordinate_groups": [[0], [1], [2]],
    }
    specs = tuple(BasisSpec("neumann", max(1, basis_cutoff - 1), name)
                  for name in ("scenario", "decay-time", "node"))
    return GroupedFieldDataset(
        f"wall_field_w{len(walls)}_r{resolution}_k{basis_cutoff}", values.float(),
        ("scenario", "time", "node"), specs, (False, False, False),
        "generated:geoaware.irregular_green_data.wall_field_tensor",
        "Diffusion field on a square divided by thin impermeable baffles; all "
        "layouts share one mesh and differ only in the barrier material.",
        metadata=metadata, groups=((0,), (1,), (2,)), operator_matrices=matrices)


def irregular_green_tensor(
        holes: tuple[Hole, ...] = DEFAULT_HOLES, *,
        polygon: Polygon = UNIT_SQUARE, resolution: int = 20,
        n_time: int = 16, n_sources: int = 24, basis_cutoff: int = 16,
        truth_modes: int = 60, contrast: float = 1., reaction: float = .15,
        time_span: tuple[float, float] = (.025, .55), mesh_seed: int = 0,
        hole_condition: str = "neumann",
        permutation_seed: int = 9173) -> GroupedFieldDataset:
    """``Y(t, receiver-node, source-node)`` on a holed square."""
    mesh = build_mesh(resolution, holes, polygon=polygon, seed=mesh_seed)
    centroids = triangle_centroids(mesh)
    truth_stiffness, mass_full = assemble_p1(
        mesh, _variable_diffusivity(centroids, contrast))
    nominal_stiffness, _ = assemble_p1(mesh, 1.)
    keep = free_nodes(mesh, hole_condition, "neumann")
    mass = restrict(mass_full, keep)
    truth_operator = restrict(truth_stiffness, keep)
    nominal_operator = restrict(nominal_stiffness, keep)
    coordinates = mesh.nodes[keep]
    n_nodes = len(keep)
    if truth_modes > n_nodes:
        truth_modes = n_nodes

    # The topology-erased operator must be assembled on exactly these nodes and
    # allowed to connect straight across the holes.  Restricting the full-square
    # operator to a sub-block would not do: dropping the interactions that pass
    # through the hole interior would quietly hand the control most of the
    # geometry it is supposed to lack.
    erased_operator, erased_mass_block = _topology_erased_operator(coordinates)

    truth_values, truth_vectors = generalized_eigenpairs(
        truth_operator, mass, truth_modes)
    sources = _farthest_point_sources(coordinates, n_sources)
    time = torch.linspace(time_span[0], time_span[1], n_time, dtype=torch.float64)
    rates = _normalized_rates(truth_values)
    decay = torch.exp(-time[:, None] * (reaction + rates[None, :]))
    weight = (1 + rates).pow(-.18)
    values = torch.einsum("tq,rq,sq,q->trs", decay, truth_vectors,
                          truth_vectors[sources], weight)
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    nominal_values, fem_basis = generalized_eigenpairs(
        nominal_operator, mass, basis_cutoff)
    erased_values, erased_basis = generalized_eigenpairs(
        erased_operator, erased_mass_block, basis_cutoff)
    box_basis, box_values = _bounding_box_product_basis(
        coordinates, mass, basis_cutoff)
    permutation = torch.randperm(
        n_nodes, generator=torch.Generator().manual_seed(permutation_seed))

    # QR of the reference decay functions can only return as many independent
    # columns as there are time samples, so the temporal cutoff is capped there
    # and the eigenvalues are sliced to match rather than assumed.
    time_cutoff = min(basis_cutoff, n_time)
    time_basis = torch.linalg.qr(
        torch.exp(-time[:, None] * (reaction + _normalized_rates(
            nominal_values)[:time_cutoff][None, :])), mode="reduced").Q
    time_basis = time_basis[:, :time_cutoff]

    spatial = {
        "fem_correct": (fem_basis, _normalized_rates(nominal_values)),
        "topology_erased": (erased_basis, _normalized_rates(erased_values)),
        "bounding_box_product": (box_basis, box_values[:box_basis.shape[1]]),
        "permuted": (fem_basis[permutation], _normalized_rates(nominal_values)),
    }
    matrices = {
        "mass": mass.float(),
        "blind_stiffness": erased_operator.float(),
        "truth_eigenvalues": rates.float(),
        "truth_eigenvectors": truth_vectors.float(),
        "nominal_stiffness": nominal_operator.float(),
        "coordinates": coordinates.float(),
        "source_nodes": sources,
        "time_basis": time_basis.float(),
        "time_eigenvalues": _normalized_rates(nominal_values)[:time_cutoff].float(),
    }
    for name, (basis, eigenvalues) in spatial.items():
        matrices[f"{name}_basis"] = basis.float()
        matrices[f"{name}_eigenvalues"] = eigenvalues[:basis.shape[1]].float()

    metadata = mesh_metadata(mesh, truth_operator, mass) | {
        "pde": "d_t u + (-div(a grad u) + reaction I) u = 0",
        "boundary_condition":
            f"Neumann on the outer boundary, {hole_condition} on hole rims",
        "discretization": "P1 finite elements on a Delaunay mesh",
        "operator_information_tier": "geometry (mesh and boundary known, material unknown)",
        "log_diffusivity_contrast": float(contrast),
        "reaction": float(reaction),
        "basis_cutoff": int(basis_cutoff),
        "time_cutoff": int(time_cutoff),
        "truth_modes": int(truth_modes),
        "n_free_nodes": int(n_nodes),
        "n_sources": int(n_sources),
        "time_span": [float(time_span[0]), float(time_span[1])],
        "source_node_indices": sources.tolist(),
        "source_selection": "deterministic farthest-point sampling",
        "permutation_seed": int(permutation_seed),
        "coordinate_groups": [[0], [1], [2]],
        "spatial_basis_columns": {name: int(basis.shape[1])
                                  for name, (basis, _) in spatial.items()},
    }
    specs = tuple(BasisSpec("neumann", max(1, basis_cutoff - 1), name)
                  for name in ("decay-time", "receiver-node", "source-node"))
    n_holes = len(holes)
    tag = f"v{len(polygon.vertices)}h{n_holes}"
    return GroupedFieldDataset(
        f"irregular_green_{tag}_r{resolution}_k{basis_cutoff}", values.float(),
        ("time", "receiver", "source"), specs, (False, False, False),
        "generated:geoaware.irregular_green_data.irregular_green_tensor",
        "Diffusion Green response on a square with circular holes; both spatial "
        "axes index the same mesh nodes.",
        metadata=metadata, groups=((0,), (1,), (2,)), operator_matrices=matrices)
