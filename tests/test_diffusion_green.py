import torch

from geoaware.masks import make_observation_split
from geoaware.operator_tucker_baselines import NeuralFunctionalTucker
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


def test_structured_masks_select_complete_named_fibers():
    data = diffusion_green_tensor(shape=(12, 18, 18), contrast=.4,
                                  basis_cutoff=6, truth_modes=10)
    source = make_observation_split(data, .1, "source_fibers", seed=101)
    source_cube = source.observed.reshape(data.shape)
    # Each fixed (time, receiver) pair contains either every source or none.
    assert torch.all((source_cube.sum(2) == 0) | (source_cube.sum(2) == data.shape[2]))
    assert abs(source.ratio_actual - .1) < .01

    receiver = make_observation_split(data, .05, "receiver_fibers", seed=102)
    receiver_cube = receiver.observed.reshape(data.shape)
    # Each fixed (time, source) pair contains either every receiver or none.
    assert torch.all((receiver_cube.sum(1) == 0) |
                     (receiver_cube.sum(1) == data.shape[1]))
    assert abs(receiver.ratio_actual - .05) < .01


def test_structured_mask_is_reproducible_and_seed_sensitive():
    data = diffusion_green_tensor(shape=(12, 18, 18), contrast=.4,
                                  basis_cutoff=6, truth_modes=10)
    first = make_observation_split(data, .05, "source_fibers", seed=101)
    repeat = make_observation_split(data, .05, "source_fibers", seed=101)
    other = make_observation_split(data, .05, "source_fibers", seed=103)
    assert torch.equal(first.observed, repeat.observed)
    assert not torch.equal(first.observed, other.observed)


def test_matched_neural_tucker_parameter_budget_is_auditable():
    model = NeuralFunctionalTucker((False, False, False), ranks=(4, 5, 5),
                                   hidden=3)
    assert sum(parameter.numel() for parameter in model.parameters()) == 210
