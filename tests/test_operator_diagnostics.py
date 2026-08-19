import math

import torch

from geoaware.joint_diffusion_2d import (assemble_axis_operator,
                                         assemble_joint_operator,
                                         joint_diffusion_2d_tensor)
from geoaware.operator_diagnostics import (generalized_eigenpairs,
                                           kronecker_sum_operator,
                                           ranked_product_basis,
                                           separability_residual,
                                           subspace_residual)


AXIS = {"x_amplitude": .55, "x_frequency": 2., "x_phase": .21,
        "y_amplitude": .40, "y_frequency": 3., "y_phase": -.37}


def _axis_pair(size: int, axis: str):
    return assemble_axis_operator(size, AXIS[f"{axis}_amplitude"],
                                  AXIS[f"{axis}_frequency"], AXIS[f"{axis}_phase"])


def test_uncoupled_joint_operator_is_exactly_a_kronecker_sum():
    nx, ny = 9, 7
    joint, mass = assemble_joint_operator(nx, ny, 0., **AXIS)
    stiffness_x, mass_x, _ = _axis_pair(nx, "x")
    stiffness_y, mass_y, _ = _axis_pair(ny, "y")
    separable = kronecker_sum_operator(stiffness_x, mass_x, stiffness_y, mass_y)
    assert torch.allclose(joint, separable, atol=1e-10)
    # The consistent mass matrix must factor as well, otherwise the whitened
    # comparison would silently mix a mass mismatch into the operator residual.
    assert torch.allclose(mass, torch.kron(mass_x, mass_y), atol=1e-12)
    assert separability_residual(joint, mass, stiffness_x, mass_x,
                                 stiffness_y, mass_y) < 1e-10


def test_separability_residual_grows_with_coupling():
    nx, ny = 9, 7
    stiffness_x, mass_x, _ = _axis_pair(nx, "x")
    stiffness_y, mass_y, _ = _axis_pair(ny, "y")
    residuals = []
    for coupling in (0., .25, .5, .75):
        joint, mass = assemble_joint_operator(nx, ny, coupling, **AXIS)
        residuals.append(separability_residual(joint, mass, stiffness_x, mass_x,
                                               stiffness_y, mass_y))
    assert residuals[0] < 1e-10
    assert all(later > earlier + 1e-3
               for earlier, later in zip(residuals, residuals[1:]))


def test_generalized_eigenpairs_are_mass_orthonormal():
    nx, ny = 8, 8
    joint, mass = assemble_joint_operator(nx, ny, .4, **AXIS)
    values, vectors = generalized_eigenpairs(joint, mass, 10)
    identity = vectors.T @ mass @ vectors
    assert torch.allclose(identity, torch.eye(10, dtype=identity.dtype), atol=1e-8)
    residual = joint @ vectors - mass @ vectors @ torch.diag(values)
    assert float(residual.norm()) < 1e-8
    assert torch.all(values[1:] >= values[:-1] - 1e-12)


def test_product_basis_matches_joint_space_only_when_separable():
    nx, ny = 9, 9
    stiffness_x, mass_x, _ = _axis_pair(nx, "x")
    stiffness_y, mass_y, _ = _axis_pair(ny, "y")
    eig_x, basis_x = generalized_eigenpairs(stiffness_x, mass_x, 6)
    eig_y, basis_y = generalized_eigenpairs(stiffness_y, mass_y, 6)
    product, _ = ranked_product_basis(basis_x, eig_x, basis_y, eig_y, 12)

    aligned, mass_a = assemble_joint_operator(nx, ny, 0., **AXIS)
    _, joint_basis = generalized_eigenpairs(aligned, mass_a, 12)
    assert subspace_residual(joint_basis, product, mass_a) < 1e-6

    coupled, mass_c = assemble_joint_operator(nx, ny, .6, **AXIS)
    _, coupled_basis = generalized_eigenpairs(coupled, mass_c, 12)
    assert subspace_residual(coupled_basis, product, mass_c) > .05


def test_dataset_records_learner_visible_diagnostics_and_is_reproducible():
    kwargs = dict(shape=(8, 9, 9, 5), joint_cutoff=10, axis_cutoff=5)
    first = joint_diffusion_2d_tensor(coupling=.5, **kwargs)
    repeat = joint_diffusion_2d_tensor(coupling=.5, **kwargs)
    assert torch.equal(first.values, repeat.values)
    assert (first.metadata["joint_stiffness_checksum"]
            == repeat.metadata["joint_stiffness_checksum"])
    assert first.metadata["operator_information_tier"] == "exact"
    assert first.metadata["operator_separability_residual"] > 0
    assert 0 < first.metadata["low_frequency_subspace_residual"] < 1

    # The wrong-operator control must differ from the learner basis while
    # keeping the same shape and eigenvalue budget.
    matrices = first.operator_matrices
    assert matrices["wrong_joint_basis"].shape == matrices["joint_basis"].shape
    assert not torch.allclose(matrices["wrong_joint_basis"], matrices["joint_basis"])

    nominal = joint_diffusion_2d_tensor(coupling=.5, learner_coupling=0., **kwargs)
    assert nominal.metadata["operator_information_tier"] == "nominal"
    assert torch.equal(nominal.values, first.values)
    assert not torch.allclose(nominal.operator_matrices["joint_basis"],
                              matrices["joint_basis"])


def test_grouped_reshape_preserves_every_entry():
    data = joint_diffusion_2d_tensor(shape=(6, 7, 7, 4), joint_cutoff=8,
                                     axis_cutoff=4, coupling=.3)
    grouped = data.grouped_values()
    assert grouped.shape == (6, 49, 4)
    assert torch.equal(grouped.reshape(data.shape), data.values)
    assert math.prod(grouped.shape) == math.prod(data.shape)
