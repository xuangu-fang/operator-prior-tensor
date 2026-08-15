import torch

from geoaware.tensor_data import (diffusion_green_tensor,
                                  explicit_mode_bases)


def test_diffusion_green_is_finite_and_auditable():
    data = diffusion_green_tensor(shape=(12, 18, 18), contrast=.4,
                                  basis_cutoff=6, truth_modes=10)
    assert data.shape == (12, 18, 18)
    assert torch.isfinite(data.values).all()
    assert abs(float(data.values.mean())) < 1e-5
    assert .999 < float(data.values.std()) < 1.001
    residual = data.metadata["oracle_product_projection_residual"]
    assert 0 < residual < 1
    assert data.metadata["diffusivity_max"] > data.metadata["diffusivity_min"]


def test_attached_reference_basis_and_wrong_operator_are_distinct():
    data = diffusion_green_tensor(shape=(12, 18, 18), contrast=.2,
                                  basis_cutoff=6, truth_modes=10)
    correct, eigenvalues = explicit_mode_bases(data, "correct")
    wrong, wrong_eigenvalues = explicit_mode_bases(data, "permuted")
    assert [tuple(item.shape) for item in correct] == [(12, 6), (18, 6), (18, 6)]
    assert all(torch.equal(left, right)
               for left, right in zip(eigenvalues, wrong_eigenvalues))
    assert all(not torch.equal(left, right)
               for left, right in zip(correct, wrong))


def test_projection_residual_increases_under_large_physical_perturbation():
    aligned = diffusion_green_tensor(shape=(12, 18, 18), contrast=0.,
                                     basis_cutoff=8, truth_modes=10)
    perturbed = diffusion_green_tensor(shape=(12, 18, 18), contrast=.8,
                                       basis_cutoff=8, truth_modes=10)
    assert (perturbed.metadata["oracle_product_projection_residual"] >
            aligned.metadata["oracle_product_projection_residual"] + .01)
