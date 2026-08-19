"""P1 finite elements on polygonal domains with circular holes.

Domains are described by an outer polygon — a square, an L, a U — optionally
minus circular obstacles.  These are the classic shapes for showing that a
boundary matters: in a U the two arms are connected only around the bottom, so
anything that treats the domain as its bounding rectangle links them directly
and gets the physics wrong.

Everything here is self-contained: the environment has no meshing library, and
a hand-rolled mesh is in any case easier to audit than a black-box generator.
The module records what a reproducibility claim needs — node coordinates,
elements, boundary tags, matrix checksums and the mesh hash — so a later run can
prove it used the same geometry rather than a similar one.

The holes matter physically, not decoratively.  With Dirichlet conditions on the
hole rims the constant mode disappears and the leading eigenfunctions bend
around the obstacles, so a basis built on the wrong hole layout — or on the
bounding box, ignoring holes entirely — is a genuinely different function space
rather than a mild perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from matplotlib.path import Path as _MplPath
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class Hole:
    center: tuple[float, float]
    radius: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Positive outside the hole, negative inside."""
        offset = points - np.asarray(self.center)[None, :]
        return np.linalg.norm(offset, axis=1) - self.radius


@dataclass(frozen=True)
class Polygon:
    """Outer boundary of a simply-connected domain, given counter-clockwise.

    A polygon covers the shapes this paper actually needs — a plain square, an
    L, a U — without pulling in a meshing library, and its boundary is exact
    rather than approximated, so boundary tags can be re-derived from geometry
    during an audit.
    """

    vertices: tuple[tuple[float, float], ...]

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.vertices, dtype=float)

    def contains(self, points: np.ndarray) -> np.ndarray:
        return _MplPath(np.concatenate([self.array, self.array[:1]])).contains_points(points)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Unsigned distance to the boundary polyline."""
        corners = self.array
        starts = corners
        ends = np.roll(corners, -1, axis=0)
        segment = ends - starts
        length = (segment ** 2).sum(1).clip(1e-30)
        offset = points[:, None, :] - starts[None, :, :]
        t = ((offset * segment[None]).sum(2) / length[None]).clip(0., 1.)
        closest = starts[None] + t[:, :, None] * segment[None]
        return np.linalg.norm(points[:, None, :] - closest, axis=2).min(1)

    def boundary_points(self, spacing: float) -> np.ndarray:
        corners = self.array
        out = []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            steps = max(1, int(round(np.linalg.norm(end - start) / spacing)))
            for k in range(steps):
                out.append(start + (end - start) * (k / steps))
        return np.asarray(out)


UNIT_SQUARE = Polygon(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
L_SHAPE = Polygon(((0., 0.), (1., 0.), (1., .5), (.5, .5), (.5, 1.), (0., 1.)))
U_SHAPE = Polygon(((0., 0.), (1., 0.), (1., 1.), (.68, 1.), (.68, .38),
                   (.32, .38), (.32, 1.), (0., 1.)))


@dataclass
class IrregularMesh:
    nodes: torch.Tensor              # (N, 2) coordinates
    triangles: torch.Tensor          # (T, 3) vertex indices
    outer_boundary: torch.Tensor     # bool (N,)
    hole_boundary: torch.Tensor      # bool (N,)
    hole_index: torch.Tensor         # long (N,), -1 away from any hole rim
    holes: tuple[Hole, ...]
    resolution: int
    polygon: Polygon = UNIT_SQUARE

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    def area(self) -> float:
        p = self.nodes[self.triangles]
        cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                 - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        return float(.5 * cross.abs().sum())

    def hash(self) -> str:
        payload = (self.nodes.double().numpy().tobytes()
                   + self.triangles.int().numpy().tobytes())
        return hashlib.sha256(payload).hexdigest()[:16]


def build_mesh(resolution: int = 20, holes: tuple[Hole, ...] = (),
               *, polygon: Polygon = UNIT_SQUARE, jitter: float = .18,
               seed: int = 0, rim_points: int = 28,
               margin: float = .04) -> IrregularMesh:
    """Interior points minus obstacles, plus explicit boundary points.

    Interior points are jittered so the mesh is not a disguised regular grid;
    otherwise a "treat the domain as a rectangle" comparison would be flattered
    by node positions that happen to form a tensor product.  Boundary points sit
    exactly on the polygon and on the hole rims, so every boundary tag can be
    recovered from the geometry alone.
    """
    if resolution < 4:
        raise ValueError("resolution must be at least four")
    rng = np.random.default_rng(seed)
    corners = polygon.array
    low, high = corners.min(0), corners.max(0)
    step = float((high - low).max()) / (resolution - 1)
    axes = [np.arange(low[d], high[d] + .5 * step, step) for d in range(2)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    grid = grid + rng.uniform(-jitter, jitter, size=grid.shape) * step

    keep = polygon.contains(grid) & (polygon.distance(grid) > margin)
    for hole in holes:
        keep &= hole.signed_distance(grid) > margin
    points = grid[keep]

    boundary = polygon.boundary_points(step)
    keep_boundary = np.ones(len(boundary), dtype=bool)
    for hole in holes:
        keep_boundary &= hole.signed_distance(boundary) > margin
    boundary = boundary[keep_boundary]
    points = np.concatenate([points, boundary])
    rim_owner = [-1] * len(points)

    for index, hole in enumerate(holes):
        angles = np.arange(rim_points) * (2 * math.pi / rim_points)
        rim = np.stack([hole.center[0] + hole.radius * np.cos(angles),
                        hole.center[1] + hole.radius * np.sin(angles)], -1)
        rim = rim[polygon.contains(rim) & (polygon.distance(rim) > margin)]
        points = np.concatenate([points, rim])
        rim_owner.extend([index] * len(rim))

    triangulation = Delaunay(points)
    centroids = points[triangulation.simplices].mean(axis=1)
    valid = polygon.contains(centroids)
    for hole in holes:
        valid &= hole.signed_distance(centroids) > 0
    simplices = triangulation.simplices[valid]

    used = np.unique(simplices)
    remap = -np.ones(len(points), dtype=np.int64)
    remap[used] = np.arange(len(used))
    points, simplices = points[used], remap[simplices]
    owner = np.asarray(rim_owner)[used]

    outer = polygon.distance(points) < 1e-9
    rim_flag = owner >= 0
    return IrregularMesh(
        torch.from_numpy(points).double(),
        torch.from_numpy(simplices.astype(np.int64)),
        torch.from_numpy(outer), torch.from_numpy(rim_flag),
        torch.from_numpy(owner), tuple(holes), resolution, polygon)


def assemble_p1(mesh: IrregularMesh,
                coefficient: torch.Tensor | float = 1.
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stiffness and consistent mass for ``-div(a grad u)`` on a P1 mesh.

    ``coefficient`` is either a scalar or one value per triangle, so a variable
    material can differ between the truth operator and the nominal operator a
    learner is allowed to see.
    """
    nodes, triangles = mesh.nodes.double(), mesh.triangles
    p = nodes[triangles]                                   # (T, 3, 2)
    edge1, edge2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    cross = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    area = .5 * cross.abs()
    if float(area.min()) <= 0:
        raise ValueError("degenerate triangle in the mesh")
    # Gradients of the barycentric hat functions, from the rotated opposite edge.
    opposite = torch.stack([p[:, 2] - p[:, 1], p[:, 0] - p[:, 2],
                            p[:, 1] - p[:, 0]], dim=1)
    gradients = torch.stack([opposite[:, :, 1], -opposite[:, :, 0]], dim=2)
    gradients = gradients / (cross[:, None, None])
    if isinstance(coefficient, torch.Tensor):
        if len(coefficient) != len(triangles):
            raise ValueError("one coefficient per triangle is required")
        a = coefficient.double()
    else:
        a = torch.full((len(triangles),), float(coefficient), dtype=torch.float64)

    n = mesh.n_nodes
    stiffness = torch.zeros(n * n, dtype=torch.float64)
    mass = torch.zeros(n * n, dtype=torch.float64)
    mass_template = (torch.ones(3, 3, dtype=torch.float64) + torch.eye(3, dtype=torch.float64)) / 12
    for i in range(3):
        for j in range(3):
            flat = triangles[:, i] * n + triangles[:, j]
            stiffness.scatter_add_(
                0, flat, area * a * (gradients[:, i] * gradients[:, j]).sum(1))
            mass.scatter_add_(0, flat, area * mass_template[i, j])
    stiffness = stiffness.reshape(n, n)
    mass = mass.reshape(n, n)
    return .5 * (stiffness + stiffness.T), .5 * (mass + mass.T)


def free_nodes(mesh: IrregularMesh, hole_condition: str = "dirichlet",
               outer_condition: str = "neumann") -> torch.Tensor:
    """Indices left after eliminating Dirichlet-constrained nodes.

    Dirichlet rims are what make holes change the spectrum sharply; insulating
    (Neumann) rims are also supported, but then the hole only alters
    connectivity and the constant mode survives.
    """
    constrained = torch.zeros(mesh.n_nodes, dtype=torch.bool)
    if hole_condition == "dirichlet":
        constrained |= mesh.hole_boundary
    elif hole_condition != "neumann":
        raise ValueError(f"unknown hole condition: {hole_condition}")
    if outer_condition == "dirichlet":
        constrained |= mesh.outer_boundary
    elif outer_condition != "neumann":
        raise ValueError(f"unknown outer condition: {outer_condition}")
    return torch.where(~constrained)[0]


def restrict(matrix: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return matrix[keep][:, keep]


def triangle_centroids(mesh: IrregularMesh) -> torch.Tensor:
    return mesh.nodes[mesh.triangles].mean(dim=1)


def mesh_metadata(mesh: IrregularMesh, stiffness: torch.Tensor,
                  mass: torch.Tensor) -> dict:
    def checksum(matrix: torch.Tensor) -> str:
        return hashlib.sha256(
            matrix.double().contiguous().numpy().tobytes()).hexdigest()[:16]

    return {
        "mesh_hash": mesh.hash(),
        "resolution": mesh.resolution,
        "n_nodes": mesh.n_nodes,
        "n_triangles": int(mesh.triangles.shape[0]),
        "domain_area": mesh.area(),
        "polygon_vertices": [list(v) for v in mesh.polygon.vertices],
        "holes": [{"center": list(hole.center), "radius": hole.radius}
                  for hole in mesh.holes],
        "n_outer_boundary_nodes": int(mesh.outer_boundary.sum()),
        "n_hole_boundary_nodes": int(mesh.hole_boundary.sum()),
        "stiffness_checksum": checksum(stiffness),
        "mass_checksum": checksum(mass),
    }
