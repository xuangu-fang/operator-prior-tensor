"""P1 finite elements on simplices of any dimension, flat or curved.

One formula covers every scenario the paper needs.  For a ``k``-simplex with
vertices ``p_0 ... p_k`` embedded in ``R^n``, write ``E`` for the ``k x n``
matrix of edge vectors from ``p_0``.  Then the volume is
``sqrt(det(E E^T)) / k!``, the barycentric gradients are the rows of
``(E E^T)^{-1} E`` together with their negated sum, and the consistent mass
matrix is ``vol / ((k+1)(k+2)) * (ones + I)``.

Taking ``k = n = 2`` gives the planar triangles of :mod:`geoaware.irregular_fem`,
``k = n = 3`` gives tetrahedra in a volume, and ``k = 2, n = 3`` gives triangles
on a curved surface — where the same expression is the Laplace-Beltrami operator
rather than a Euclidean Laplacian, because ``E E^T`` is the induced metric.  A
sphere therefore needs no special code, only a different mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from scipy.spatial import Delaunay


@dataclass
class SimplexMesh:
    """Nodes in ``R^n`` and the ``k``-simplices connecting them."""

    nodes: torch.Tensor              # (N, n)
    cells: torch.Tensor              # (T, k+1)
    kind: str = ""

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def degree(self) -> int:
        return int(self.cells.shape[1]) - 1

    def centroids(self) -> torch.Tensor:
        return self.nodes[self.cells].mean(dim=1)

    def volume(self) -> float:
        return float(cell_volumes(self).sum())

    def hash(self) -> str:
        payload = (self.nodes.double().numpy().tobytes()
                   + self.cells.int().numpy().tobytes())
        return hashlib.sha256(payload).hexdigest()[:16]


def _edges(mesh: SimplexMesh) -> torch.Tensor:
    corners = mesh.nodes[mesh.cells].double()          # (T, k+1, n)
    return corners[:, 1:] - corners[:, :1]             # (T, k, n)


def cell_volumes(mesh: SimplexMesh) -> torch.Tensor:
    edges = _edges(mesh)
    metric = edges @ edges.transpose(1, 2)             # (T, k, k) induced metric
    determinant = torch.linalg.det(metric).clamp_min(0.)
    return determinant.sqrt() / math.factorial(mesh.degree)


def barycentric_gradients(mesh: SimplexMesh) -> torch.Tensor:
    """Gradients of the ``k+1`` hat functions, tangent to the simplex."""
    edges = _edges(mesh)
    metric = edges @ edges.transpose(1, 2)
    inverse = torch.linalg.inv(metric)
    upper = inverse @ edges                            # (T, k, n)
    first = -upper.sum(dim=1, keepdim=True)            # (T, 1, n)
    return torch.cat([first, upper], dim=1)            # (T, k+1, n)


def assemble(mesh: SimplexMesh, coefficient: torch.Tensor | float = 1.
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stiffness for ``-div(a grad u)`` and the consistent mass matrix.

    ``coefficient`` is a scalar or one value per cell, so the truth operator and
    the operator a learner is allowed to see can differ in material while
    sharing every node.
    """
    volumes = cell_volumes(mesh)
    if float(volumes.min()) <= 0:
        raise ValueError("degenerate simplex in the mesh")
    gradients = barycentric_gradients(mesh)
    if isinstance(coefficient, torch.Tensor):
        if len(coefficient) != len(mesh.cells):
            raise ValueError("one coefficient per cell is required")
        material = coefficient.double()
    else:
        material = torch.full((len(mesh.cells),), float(coefficient),
                              dtype=torch.float64)

    local_stiffness = ((gradients @ gradients.transpose(1, 2))
                       * (volumes * material)[:, None, None])
    corners = mesh.cells.shape[1]
    pattern = torch.eye(corners, dtype=torch.float64) + 1.
    local_mass = (volumes / (corners * (corners + 1)))[:, None, None] * pattern

    n = mesh.n_nodes
    rows = mesh.cells[:, :, None].expand(-1, -1, corners).reshape(-1)
    columns = mesh.cells[:, None, :].expand(-1, corners, -1).reshape(-1)
    flat = rows * n + columns
    stiffness = torch.zeros(n * n, dtype=torch.float64).index_add_(
        0, flat, local_stiffness.reshape(-1))
    mass = torch.zeros(n * n, dtype=torch.float64).index_add_(
        0, flat, local_mass.reshape(-1))
    return stiffness.reshape(n, n), mass.reshape(n, n)


def build_box_mesh(resolution: int = 10, *, jitter: float = .18, seed: int = 0,
                   low: tuple[float, ...] = (0., 0., 0.),
                   high: tuple[float, ...] = (1., 1., 1.)) -> SimplexMesh:
    """Tetrahedra filling an axis-aligned box.

    Interior points are jittered for the same reason as in two dimensions: an
    unjittered grid is a disguised tensor product, which would flatter any
    "treat the domain as a box" comparison.  Boundary faces are left on the
    lattice so the box is filled exactly.
    """
    lower, upper = np.asarray(low, float), np.asarray(high, float)
    rng = np.random.default_rng(seed)
    axes = [np.linspace(lower[d], upper[d], resolution) for d in range(len(lower))]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(lower))
    step = float((upper - lower).max()) / (resolution - 1)
    interior = np.all((grid > lower + 1e-12) & (grid < upper - 1e-12), axis=1)
    grid[interior] += rng.uniform(-jitter, jitter, size=grid[interior].shape) * step
    cells = Delaunay(grid).simplices
    mesh = SimplexMesh(torch.from_numpy(grid).double(),
                       torch.from_numpy(cells.astype(np.int64)), "box")
    # Delaunay on a lattice leaves slivers of essentially zero volume among the
    # coplanar boundary points.  They carry no energy and no mass but do make
    # the barycentric metric singular, so they are dropped rather than tolerated.
    volumes = cell_volumes(mesh)
    keep = volumes > 1e-9 * float(volumes.max())
    mesh.cells = mesh.cells[keep]
    used, remapped = torch.unique(mesh.cells, return_inverse=True)
    mesh.nodes = mesh.nodes[used]
    mesh.cells = remapped.reshape(mesh.cells.shape)
    return mesh


def build_sphere_mesh(subdivisions: int = 3, radius: float = 1.) -> SimplexMesh:
    """A geodesic sphere: an icosahedron subdivided and projected outwards.

    The surface is a two-dimensional manifold with no boundary and no global
    coordinate chart, so a bounding-box product basis or a network reading raw
    ``(x, y, z)`` is wrong for a structural reason rather than by a margin.
    """
    phi = (1 + math.sqrt(5)) / 2
    vertices = []
    for a, b in ((1., phi), (-1., phi), (1., -phi), (-1., -phi)):
        vertices += [(0., a, b), (a, b, 0.), (b, 0., a)]
    nodes = np.asarray(vertices, dtype=float)
    nodes /= np.linalg.norm(nodes, axis=1, keepdims=True)
    faces = _convex_hull_faces(nodes)

    for _ in range(subdivisions):
        midpoints, new_faces = {}, []

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                point = nodes[i] + nodes[j]
                midpoints[key] = len(nodes) + len(extra)
                extra.append(point / np.linalg.norm(point))
            return midpoints[key]

        extra: list[np.ndarray] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        nodes = np.concatenate([nodes, np.asarray(extra)])
        faces = np.asarray(new_faces, dtype=np.int64)

    return SimplexMesh(torch.from_numpy(nodes * radius).double(),
                       torch.from_numpy(faces), "sphere")


def _convex_hull_faces(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    faces = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        a, b, c = points[simplex]
        normal = np.cross(b - a, c - a)
        # Orient every face outwards so subdivision keeps a consistent winding.
        faces.append(simplex if normal @ equation[:3] > 0 else simplex[::-1])
    return np.asarray(faces, dtype=np.int64)
