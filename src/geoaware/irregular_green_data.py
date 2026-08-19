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

import math

import torch

from .bases import BasisSpec
from .irregular_fem import (UNIT_SQUARE, Hole, IrregularMesh, Polygon,
                            assemble_p1, build_mesh, free_nodes, mesh_metadata,
                            restrict, triangle_centroids)
from .joint_diffusion_2d import GroupedFieldDataset
from .operator_diagnostics import generalized_eigenpairs

DEFAULT_HOLES = (Hole((.32, .62), .15), Hole((.68, .33), .13))


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
