"""Controlled two-dimensional operator family with tunable separability.

The point of this benchmark is a single knob.  On a fixed square grid with
fixed boundary conditions we discretize

``d_t u + [ -div(A_eta grad u) + kappa I ] u = 0``,

with the anisotropic diffusion tensor

``A_eta(x, y) = [[a_x(x), eta c(x, y)], [eta c(x, y), a_y(y)]]``,
``c(x, y) = sqrt(a_x(x) a_y(y)) s(x, y)``, ``|s| <= 1``.

At ``eta = 0`` the diffusion tensor is diagonal with axis-wise coefficients, so
the Q1 stiffness matrix is *exactly* the Kronecker sum ``K_x (x) M_y + M_x (x)
K_y``: a per-axis factorization loses nothing, and both diagnostics vanish to
machine precision.  Increasing ``eta`` keeps the same PDE family, boundary
condition, mesh and noise while continuously destroying that structure, and
``|eta s| <= 1`` keeps ``A_eta`` positive semidefinite so the problem stays a
well-posed diffusion.

This is deliberately *not* an operator-misspecification benchmark.  The learner
may see the exact discrete operator (information tier 1 in the report); the
question under test is whether a joint operator on the coordinate group
``(x, y)`` beats the per-axis approximation, and whether the gap tracks the
measured separability.
"""

from __future__ import annotations

import hashlib
import math

import torch

from .bases import BasisSpec
from .data import FieldDataset
from .operator_diagnostics import (generalized_eigenpairs, ranked_product_basis,
                                   separability_residual, subspace_residual)


# Two-point Gauss rule on the unit interval, exact for cubics.  The 2-D rule is
# its tensor product, so 1-D and 2-D assembly evaluate coefficients at exactly
# the same physical points and the separable case agrees to machine precision.
_GAUSS = ((1 - 1 / math.sqrt(3)) / 2, (1 + 1 / math.sqrt(3)) / 2)
_LOCAL = ((0, 0), (0, 1), (1, 0), (1, 1))


class GroupedFieldDataset(FieldDataset):
    """A field plus an explicit coordinate partition and its operators.

    ``groups`` records which raw axes share a physical operator domain.  Every
    basis a learner is allowed to use is stored in ``operator_matrices`` so an
    audit can check that no variant silently reads a different operator than it
    claims.
    """

    def __init__(self, *args, groups=None, operator_matrices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.operator_matrices = operator_matrices

    def grouped_shape(self) -> tuple[int, ...]:
        return tuple(math.prod(self.shape[axis] for axis in group)
                     for group in self.groups)

    def grouped_values(self) -> torch.Tensor:
        """Reshape into one axis per group without moving any entry."""
        order = [axis for group in self.groups for axis in group]
        if sorted(order) != list(range(len(self.shape))):
            raise ValueError("groups must partition every axis exactly once")
        return self.values.permute(*order).reshape(self.grouped_shape())


def _axis_coefficient(x: torch.Tensor, amplitude: float, frequency: float,
                      phase: float) -> torch.Tensor:
    return torch.exp(amplitude * torch.cos(frequency * math.pi * x + phase))


def _coupling_shape(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Normalized correlation pattern ``s(x, y)`` with ``|s| <= 1``.

    It is deliberately non-separable and sign-changing: a constant or a product
    ``f(x)g(y)`` would let a per-axis model absorb part of the coupling and
    would blur what the separability axis measures.
    """
    return (torch.sin(2 * math.pi * x + .3) * torch.cos(2 * math.pi * y - .4)
            + .35 * torch.sin(math.pi * (x + y))) / 1.35


def assemble_axis_operator(size: int, amplitude: float, frequency: float,
                           phase: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """P1 stiffness/mass for ``-d/dx(a(x) d/dx)`` with natural Neumann BC."""
    if size < 3:
        raise ValueError("axis needs at least three nodes")
    h = 1. / (size - 1)
    stiffness = torch.zeros(size, size, dtype=torch.float64)
    mass = torch.zeros(size, size, dtype=torch.float64)
    left = torch.arange(size - 1)
    coefficient_samples = []
    for point in _GAUSS:
        x = (left.double() + point) * h
        a = _axis_coefficient(x, amplitude, frequency, phase)
        coefficient_samples.append(a)
        shape = torch.tensor([1 - point, point], dtype=torch.float64)
        gradient = torch.tensor([-1. / h, 1. / h], dtype=torch.float64)
        weight = h / 2
        for p, (sp, gp) in enumerate(zip(shape, gradient)):
            for q, (sq, gq) in enumerate(zip(shape, gradient)):
                stiffness[left + p, left + q] += weight * a * gp * gq
                mass[left + p, left + q] += weight * sp * sq
    coefficient = torch.cat(coefficient_samples)
    return stiffness, mass, coefficient


def assemble_joint_operator(nx: int, ny: int, coupling: float, *,
                            x_amplitude: float, x_frequency: float, x_phase: float,
                            y_amplitude: float, y_frequency: float, y_phase: float
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Q1 stiffness/mass for ``-div(A_eta grad u)`` on a uniform rectangle.

    Nodes are numbered ``ix * ny + iy`` so that ``values.reshape(nx * ny)``
    matches ``torch.kron`` ordering, which is what makes the Kronecker-sum
    comparison meaningful rather than a permutation artifact.
    """
    if abs(coupling) > 1:
        raise ValueError("|coupling| > 1 would make the diffusion tensor indefinite")
    hx, hy = 1. / (nx - 1), 1. / (ny - 1)
    cells = torch.cartesian_prod(torch.arange(nx - 1), torch.arange(ny - 1))
    base = cells[:, 0] * ny + cells[:, 1]
    nodes = torch.stack([base + dx * ny + dy for dx, dy in _LOCAL], dim=1)

    n = nx * ny
    stiffness = torch.zeros(n * n, dtype=torch.float64)
    mass = torch.zeros(n * n, dtype=torch.float64)
    for xi in _GAUSS:
        for eta in _GAUSS:
            x = (cells[:, 0].double() + xi) * hx
            y = (cells[:, 1].double() + eta) * hy
            ax = _axis_coefficient(x, x_amplitude, x_frequency, x_phase)
            ay = _axis_coefficient(y, y_amplitude, y_frequency, y_phase)
            axy = coupling * (ax * ay).sqrt() * _coupling_shape(x, y)
            weight = hx * hy / 4
            shape, grad_x, grad_y = [], [], []
            for dx, dy in _LOCAL:
                sx, sy = (xi if dx else 1 - xi), (eta if dy else 1 - eta)
                shape.append(sx * sy)
                grad_x.append((1. if dx else -1.) * sy / hx)
                grad_y.append(sx * (1. if dy else -1.) / hy)
            for p in range(4):
                for q in range(4):
                    flat = nodes[:, p] * n + nodes[:, q]
                    local = weight * (ax * grad_x[p] * grad_x[q]
                                      + ay * grad_y[p] * grad_y[q]
                                      + axy * (grad_x[p] * grad_y[q]
                                               + grad_y[p] * grad_x[q]))
                    stiffness.scatter_add_(0, flat, local)
                    mass.scatter_add_(0, flat,
                                      torch.full_like(local, weight * shape[p] * shape[q]))
    stiffness = stiffness.reshape(n, n)
    mass = mass.reshape(n, n)
    return .5 * (stiffness + stiffness.T), .5 * (mass + mass.T)


def _checksum(matrix: torch.Tensor) -> str:
    return hashlib.sha256(
        matrix.double().contiguous().numpy().tobytes()).hexdigest()[:16]


def _scenario_initial_conditions(nx: int, ny: int, count: int,
                                 seed: int) -> torch.Tensor:
    """Localized smooth initial states, one per scenario.

    Gaussian bumps are used instead of low-frequency mode draws on purpose: a
    truncated learner basis must face genuine unresolved energy, otherwise the
    projection residual would be an artifact of the generator.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, 1, nx, dtype=torch.float64)[:, None, None]
    y = torch.linspace(0, 1, ny, dtype=torch.float64)[None, :, None]
    centers_x = .15 + .7 * torch.rand(count, 3, generator=generator).double()
    centers_y = .15 + .7 * torch.rand(count, 3, generator=generator).double()
    widths = .08 + .10 * torch.rand(count, 3, generator=generator).double()
    weights = torch.randn(count, 3, generator=generator).double()
    fields = []
    for s in range(count):
        field = torch.zeros(nx, ny, dtype=torch.float64)
        for b in range(3):
            field = field + weights[s, b] * torch.exp(
                -(((x[:, :, 0] - centers_x[s, b]) ** 2
                   + (y[:, :, 0] - centers_y[s, b]) ** 2)
                  / (2 * widths[s, b] ** 2)))
        fields.append(field - field.mean())
    return torch.stack(fields)


def joint_diffusion_2d_tensor(
        shape: tuple[int, int, int, int] = (16, 16, 16, 12),
        coupling: float = 0., *, reaction: float = .25,
        joint_cutoff: int = 16, axis_cutoff: int = 8, time_cutoff: int = 8,
        learner_coupling: float | None = None, wrong_coupling: float = .9,
        truth_modes: int | None = None, time_span: tuple[float, float] = (.01, .35),
        seed: int = 5101) -> GroupedFieldDataset:
    """Spatiotemporal tensor ``Y(t, x, y, s)`` for the group-wise POC.

    ``learner_coupling`` defaults to ``coupling``, i.e. the exact-operator tier:
    the learner knows the discrete operator but still only keeps its leading
    ``joint_cutoff`` modes.  Passing ``0`` instead gives the nominal tier, where
    the learner assumes a separable operator that the truth does not obey; the
    two tiers must never be reported in the same column.
    """
    nt, nx, ny, ns = shape
    if learner_coupling is None:
        learner_coupling = coupling
    axis_kwargs = {"x_amplitude": .55, "x_frequency": 2., "x_phase": .21,
                   "y_amplitude": .40, "y_frequency": 3., "y_phase": -.37}

    stiffness_x, mass_x, _ = assemble_axis_operator(
        nx, axis_kwargs["x_amplitude"], axis_kwargs["x_frequency"], axis_kwargs["x_phase"])
    stiffness_y, mass_y, _ = assemble_axis_operator(
        ny, axis_kwargs["y_amplitude"], axis_kwargs["y_frequency"], axis_kwargs["y_phase"])
    truth_stiffness, joint_mass = assemble_joint_operator(nx, ny, coupling, **axis_kwargs)
    truth_stiffness = truth_stiffness + reaction * joint_mass
    stiffness_x = stiffness_x + .5 * reaction * mass_x
    stiffness_y = stiffness_y + .5 * reaction * mass_y

    n_space = nx * ny
    modes = n_space if truth_modes is None else int(truth_modes)
    truth_values, truth_vectors = generalized_eigenpairs(truth_stiffness, joint_mass, modes)

    time = torch.linspace(time_span[0], time_span[1], nt, dtype=torch.float64)
    initial = _scenario_initial_conditions(nx, ny, ns, seed).reshape(ns, n_space)
    amplitudes = initial @ joint_mass @ truth_vectors                    # (ns, modes)
    decay = torch.exp(-time[:, None] * truth_values[None, :])            # (nt, modes)
    values = torch.einsum("tq,xq,sq->txs", decay, truth_vectors, amplitudes)
    values = values.reshape(nt, nx, ny, ns)
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    # Learner-visible operators.  The wrong control keeps the same family, mesh,
    # cutoff and eigenvalue magnitudes and only changes the coupling, so it is a
    # misspecified operator rather than a broken one.
    learner_stiffness = (truth_stiffness if learner_coupling == coupling else
                         assemble_joint_operator(nx, ny, learner_coupling, **axis_kwargs)[0]
                         + reaction * joint_mass)
    wrong_stiffness = (assemble_joint_operator(nx, ny, wrong_coupling, **axis_kwargs)[0]
                       + reaction * joint_mass)
    joint_eigenvalues, joint_basis = generalized_eigenpairs(
        learner_stiffness, joint_mass, joint_cutoff)
    wrong_eigenvalues, wrong_basis = generalized_eigenpairs(
        wrong_stiffness, joint_mass, joint_cutoff)
    # Two different controls, never to be reported as one.  ``wrong_joint`` is a
    # *misspecified* operator from the same family, which may still carry a
    # usable low-frequency space; ``permuted_joint`` destroys the node-operator
    # alignment outright and is the destructive negative control matching the
    # frozen 1-D protocol.
    permutation = torch.randperm(
        n_space, generator=torch.Generator().manual_seed(seed + 9173))
    permuted_basis = joint_basis[permutation]
    axis_x_eigenvalues, axis_x_basis = generalized_eigenpairs(stiffness_x, mass_x, axis_cutoff)
    axis_y_eigenvalues, axis_y_basis = generalized_eigenpairs(stiffness_y, mass_y, axis_cutoff)
    product_basis, product_eigenvalues = ranked_product_basis(
        axis_x_basis, axis_x_eigenvalues, axis_y_basis, axis_y_eigenvalues, joint_cutoff)

    # The time group inherits the semigroup induced by the same operator.
    time_basis = torch.linalg.qr(
        torch.exp(-time[:, None] * joint_eigenvalues[:time_cutoff][None, :]),
        mode="reduced").Q
    time_eigenvalues = joint_eigenvalues[:time_cutoff].clone()

    def normalize(eigenvalues: torch.Tensor) -> torch.Tensor:
        span = (eigenvalues[1] - eigenvalues[0]).clamp_min(1e-12)
        return ((eigenvalues - eigenvalues[0]) / span).float()

    epsilon_sep = separability_residual(truth_stiffness, joint_mass, stiffness_x,
                                        mass_x, stiffness_y, mass_y, reaction=0.)
    epsilon_sub = subspace_residual(joint_basis, product_basis, joint_mass)
    metadata = {
        "pde": "d_t u + (-div(A_eta grad u) + kappa I) u = 0",
        "boundary_condition": "natural Neumann (zero flux) on the unit square",
        "discretization": "Q1 finite elements, 2x2 Gauss quadrature",
        "coupling": float(coupling),
        "learner_coupling": float(learner_coupling),
        "wrong_coupling": float(wrong_coupling),
        "reaction": float(reaction),
        "operator_information_tier": ("exact" if learner_coupling == coupling
                                      else "nominal"),
        "joint_cutoff": int(joint_cutoff),
        "axis_cutoff": int(axis_cutoff),
        "time_cutoff": int(time_cutoff),
        "truth_modes": int(modes),
        "time_span": [float(time_span[0]), float(time_span[1])],
        "coordinate_groups": [[0], [1, 2], [3]],
        "operator_separability_residual": epsilon_sep,
        "low_frequency_subspace_residual": epsilon_sub,
        "wrong_operator_subspace_residual": subspace_residual(
            joint_basis, wrong_basis, joint_mass),
        "node_permutation_seed": int(seed + 9173),
        "joint_stiffness_checksum": _checksum(truth_stiffness),
        "joint_mass_checksum": _checksum(joint_mass),
        "learner_stiffness_checksum": _checksum(learner_stiffness),
        "axis_stiffness_checksums": [_checksum(stiffness_x), _checksum(stiffness_y)],
        "scenario_seed": int(seed),
    }
    matrices = {
        "joint_stiffness": learner_stiffness.float(),
        "joint_mass": joint_mass.float(),
        "axis_stiffness": [stiffness_x.float(), stiffness_y.float()],
        "axis_mass": [mass_x.float(), mass_y.float()],
        "joint_basis": joint_basis.float(),
        "joint_eigenvalues": normalize(joint_eigenvalues),
        "product_basis": product_basis.float(),
        "product_eigenvalues": normalize(product_eigenvalues),
        "wrong_joint_basis": wrong_basis.float(),
        "wrong_joint_eigenvalues": normalize(wrong_eigenvalues),
        "permuted_joint_basis": permuted_basis.float(),
        "permuted_joint_eigenvalues": normalize(joint_eigenvalues),
        "axis_bases": [axis_x_basis.float(), axis_y_basis.float()],
        "axis_eigenvalues": [normalize(axis_x_eigenvalues), normalize(axis_y_eigenvalues)],
        "time_basis": time_basis.float(),
        "time_eigenvalues": normalize(time_eigenvalues),
    }
    specs = (BasisSpec("neumann", max(1, time_cutoff - 1), "decay-time"),
             BasisSpec("neumann", max(1, axis_cutoff - 1), "x"),
             BasisSpec("neumann", max(1, axis_cutoff - 1), "y"),
             BasisSpec("neumann", 1, "scenario"))
    return GroupedFieldDataset(
        f"joint_diffusion_2d_eta{coupling:.2f}_k{joint_cutoff}", values.float(),
        ("time", "x", "y", "scenario"), specs, (False, False, False, False),
        "generated:geoaware.joint_diffusion_2d.joint_diffusion_2d_tensor",
        "Anisotropic 2-D diffusion with tunable non-separable coupling; "
        "axes are time x x x y x scenario with coordinate groups {t},{x,y},{s}.",
        metadata=metadata, groups=((0,), (1, 2), (3,)), operator_matrices=matrices)
