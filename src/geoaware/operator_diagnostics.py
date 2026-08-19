"""Auditable residuals for group-wise operator priors.

Three quantities separate claims that are easy to confuse:

``epsilon_sep``
    How far the *joint* discrete operator is from the Kronecker-sum operator
    implied by per-axis operators.  It is a property of the PDE discretization
    alone and never reads the field.

``epsilon_sub``
    How far the *finite low-frequency subspace* actually used by a learner
    differs between the joint eigenbasis and the per-axis product basis.  Two
    operators can be quite non-separable while their leading eigenspaces still
    agree, so this is the quantity that bounds what completion can feel.

``product_projection_residual``
    The relative energy of the target tensor outside the learner's grouped
    product space.  This is the bias floor of any method restricted to those
    bases.

All three are computed from learner-visible operators.  None may be tuned by
inspecting held-out entries.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return .5 * (matrix + matrix.transpose(-1, -2))


def inverse_sqrt(matrix: torch.Tensor, floor: float = 1e-12) -> torch.Tensor:
    """Symmetric inverse square root of a positive-definite matrix."""
    values, vectors = torch.linalg.eigh(_symmetrize(matrix.double()))
    return vectors @ torch.diag(values.clamp_min(floor).rsqrt()) @ vectors.T


def matrix_sqrt(matrix: torch.Tensor, floor: float = 0.) -> torch.Tensor:
    """Symmetric square root of a positive-semidefinite matrix."""
    values, vectors = torch.linalg.eigh(_symmetrize(matrix.double()))
    return vectors @ torch.diag(values.clamp_min(floor).sqrt()) @ vectors.T


def generalized_eigenpairs(stiffness: torch.Tensor, mass: torch.Tensor,
                           count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest ``count`` solutions of ``K phi = lambda M phi``.

    Eigenvectors are returned mass-orthonormal (``Phi^T M Phi = I``), which is
    the correct normalization on non-uniform meshes: plain Euclidean
    orthogonality would let node density decide what counts as low frequency.
    """
    if count < 1 or count > stiffness.shape[0]:
        raise ValueError("count must lie in [1, matrix size]")
    whitener = inverse_sqrt(mass)
    whitened = _symmetrize(whitener @ stiffness.double() @ whitener)
    values, vectors = torch.linalg.eigh(whitened)
    return values[:count].clone(), (whitener @ vectors[:, :count]).clone()


def kronecker_sum_operator(stiffness_x: torch.Tensor, mass_x: torch.Tensor,
                           stiffness_y: torch.Tensor, mass_y: torch.Tensor,
                           reaction: float = 0.) -> torch.Tensor:
    """``K_x (x) M_y + M_x (x) K_y + reaction M_x (x) M_y``.

    This is the operator a per-axis factorization implicitly assumes.  The
    Kronecker order matches ``values.reshape(nx * ny)`` with the x axis varying
    slowest, i.e. ``torch.kron``.
    """
    separable = (torch.kron(stiffness_x.double(), mass_y.double())
                 + torch.kron(mass_x.double(), stiffness_y.double()))
    if reaction:
        separable = separable + reaction * torch.kron(mass_x.double(), mass_y.double())
    return _symmetrize(separable)


def separability_residual(joint_stiffness: torch.Tensor, joint_mass: torch.Tensor,
                          stiffness_x: torch.Tensor, mass_x: torch.Tensor,
                          stiffness_y: torch.Tensor, mass_y: torch.Tensor,
                          reaction: float = 0.) -> float:
    """Mass-whitened relative distance to the best per-axis Kronecker sum.

    Whitening first is essential: without it the residual would change under a
    pure re-scaling of the mesh or of the mass matrix, and the number could be
    made small or large without touching the physics.
    """
    whitener = inverse_sqrt(joint_mass)
    joint = _symmetrize(whitener @ joint_stiffness.double() @ whitener)
    separable = kronecker_sum_operator(stiffness_x, mass_x, stiffness_y, mass_y,
                                       reaction)
    separable = _symmetrize(whitener @ separable @ whitener)
    return float((joint - separable).norm() / joint.norm().clamp_min(1e-12))


def mass_orthonormal_projector(basis: torch.Tensor,
                               mass: torch.Tensor) -> torch.Tensor:
    """Projector onto ``span(basis)`` in whitened coordinates."""
    whitened = matrix_sqrt(mass) @ basis.double()
    q = torch.linalg.qr(whitened, mode="reduced").Q
    return q @ q.T


def subspace_residual(joint_basis: torch.Tensor, product_basis: torch.Tensor,
                      mass: torch.Tensor) -> float:
    """Relative projector distance between two finite low-frequency spaces.

    Both bases must have the same number of columns; comparing spaces of
    different dimension would confound subspace disagreement with a difference
    in latent budget, which is exactly the fairness question this experiment
    has to keep separate.
    """
    if joint_basis.shape[1] != product_basis.shape[1]:
        raise ValueError("subspace residual requires a matched latent dimension")
    joint = mass_orthonormal_projector(joint_basis, mass)
    product = mass_orthonormal_projector(product_basis, mass)
    return float((joint - product).norm() / joint.norm().clamp_min(1e-12))


def ranked_product_basis(basis_x: torch.Tensor, eigenvalues_x: torch.Tensor,
                         basis_y: torch.Tensor, eigenvalues_y: torch.Tensor,
                         count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis product basis truncated to ``count`` lowest summed eigenvalues.

    Selecting by summed eigenvalue rather than by a rectangular ``K_x x K_y``
    block is what makes the comparison honest: when the joint operator is an
    exact Kronecker sum, the joint eigenvalues *are* these sums, so the two
    bases then span the same space and ``epsilon_sub`` is zero by construction
    instead of by luck.  Ties are broken deterministically by index.
    """
    kx, ky = basis_x.shape[1], basis_y.shape[1]
    if count > kx * ky:
        raise ValueError("requested more product modes than the axis cutoffs allow")
    pairs = torch.cartesian_prod(torch.arange(kx), torch.arange(ky))
    summed = eigenvalues_x.double()[pairs[:, 0]] + eigenvalues_y.double()[pairs[:, 1]]
    order = torch.argsort(summed, stable=True)[:count]
    pairs, summed = pairs[order], summed[order]
    columns = (basis_x.double()[:, pairs[:, 0]][:, None, :]
               * basis_y.double()[:, pairs[:, 1]][None, :, :])
    return columns.reshape(-1, count).contiguous(), summed.contiguous()


def product_projection_residual(values: torch.Tensor,
                                bases: Sequence[torch.Tensor | None]) -> float:
    """Relative energy of ``values`` outside the grouped product space.

    ``bases[m] is None`` marks a group with no operator prior, which is left
    unprojected.  ``values`` must already be reshaped into grouped form, one
    axis per entry of ``bases``.
    """
    if values.ndim != len(bases):
        raise ValueError("values must have one axis per group")
    projected = values.double()
    for mode, basis in enumerate(bases):
        if basis is None:
            continue
        q = torch.linalg.qr(basis.double(), mode="reduced").Q
        projected = torch.tensordot(q @ q.T, projected, dims=([1], [mode]))
        projected = projected.movedim(0, mode)
    return float((values.double() - projected).norm() /
                 values.double().norm().clamp_min(1e-12))
