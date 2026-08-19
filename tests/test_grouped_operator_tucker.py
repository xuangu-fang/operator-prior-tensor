import torch

from geoaware.grouped_operator_tucker import (GroupFactorSpec,
                                              GroupedOperatorTucker,
                                              group_coordinates,
                                              grouped_indices,
                                              sobolev_penalty_operator)
from geoaware.joint_diffusion_2d import joint_diffusion_2d_tensor
from geoaware.operator_diagnostics import generalized_eigenpairs
from geoaware.tensor_bayes import OperatorBayesianTucker


def _small_dataset(coupling=.4):
    return joint_diffusion_2d_tensor(shape=(8, 9, 9, 5), coupling=coupling,
                                     joint_cutoff=10, axis_cutoff=5,
                                     time_cutoff=5)


def _joint_specs(data, ranks=(3, 5, 3)):
    matrices = data.operator_matrices
    return [
        GroupFactorSpec("operator", ranks[0], data.shape[0],
                        basis=matrices["time_basis"],
                        eigenvalues=matrices["time_eigenvalues"], name="time"),
        GroupFactorSpec("operator", ranks[1], data.shape[1] * data.shape[2],
                        basis=matrices["joint_basis"],
                        eigenvalues=matrices["joint_eigenvalues"], name="space"),
        GroupFactorSpec("neural", ranks[2], data.shape[3],
                        coordinates=group_coordinates(data.shape, (3,)),
                        hidden=8, name="scenario"),
    ]


def test_grouped_indices_follow_flat_order_and_partition_axes():
    shape = (2, 3, 4)
    index = grouped_indices(shape, ((0,), (1, 2)))
    assert index.shape == (24, 2)
    # Row 7 is flat index 7 = (0, 1, 3): time 0, grouped (y, z) index 1*4+3.
    assert tuple(index[7].tolist()) == (0, 7)
    assert int(index[:, 1].max()) == 11
    try:
        grouped_indices(shape, ((0,), (1,)))
    except ValueError:
        pass
    else:
        raise AssertionError("an incomplete partition must be rejected")


def test_singleton_partition_reproduces_the_frozen_order_three_design():
    torch.manual_seed(0)
    factors = [torch.randn(6, 3), torch.randn(7, 4), torch.randn(5, 2)]
    indices = torch.stack([torch.randint(0, n, (11,)) for n in (6, 7, 5)], 1)
    frozen = OperatorBayesianTucker.tucker_design(indices, factors)
    grouped = GroupedOperatorTucker.design(indices, factors)
    assert torch.allclose(frozen, grouped, atol=1e-6)


def test_grouped_model_fits_and_predicts_finite_values():
    data = _small_dataset()
    model = GroupedOperatorTucker(_joint_specs(data), device="cpu")
    index = grouped_indices(data.shape, data.groups)
    truth = data.values.flatten()
    generator = torch.Generator().manual_seed(3)
    observed = torch.randperm(len(truth), generator=generator)[:600]
    model.fit(index[observed], truth[observed], steps=40, seed=3)
    prediction = model.predict(index)
    assert prediction.mean.shape == truth.shape
    assert torch.isfinite(prediction.mean).all()
    assert torch.isfinite(prediction.std).all() and (prediction.std > 0).all()
    assert prediction.metadata["group_kinds"] == ["operator", "operator", "neural"]
    assert model.spatial_latent_dimension() == 15


def _fit_and_score(data, index, truth, observed, held, basis_key, eigen_key):
    specs = _joint_specs(data)
    specs[1] = GroupFactorSpec(
        "operator", specs[1].rank, specs[1].size,
        basis=data.operator_matrices[basis_key],
        eigenvalues=data.operator_matrices[eigen_key], name="space")
    torch.manual_seed(5)
    model = GroupedOperatorTucker(specs, device="cpu")
    model.fit(index[observed], truth[observed], steps=250, seed=5)
    # Equal trainable budget keeps every control a test of the operator rather
    # than of capacity.
    assert sum(p.numel() for p in model.parameters()) == 15 + 50 + 115 + 45
    error = model.predict(index).mean[held] - truth[held]
    return float(error.square().mean().sqrt() / truth[held].std())


def test_node_permutation_destroys_the_operator_prior():
    """The destructive control must fail; a *misspecified* one need not.

    Only index-operator alignment is destroyed here: the basis keeps its
    columns, eigenvalues, parameter count, optimizer and mask.  Whether a
    merely misspecified coupling also degrades is a research question measured
    by the experiment, not a property a unit test may presuppose.
    """
    data = _small_dataset(coupling=.6)
    index = grouped_indices(data.shape, data.groups)
    truth = data.values.flatten()
    generator = torch.Generator().manual_seed(11)
    observed = torch.randperm(len(truth), generator=generator)[:1200]
    held = torch.ones(len(truth), dtype=torch.bool)
    held[observed] = False

    correct = _fit_and_score(data, index, truth, observed, held,
                             "joint_basis", "joint_eigenvalues")
    permuted = _fit_and_score(data, index, truth, observed, held,
                              "permuted_joint_basis", "permuted_joint_eigenvalues")
    assert correct < .5 * permuted


def test_misspecified_and_permuted_controls_are_distinct_operators():
    data = _small_dataset(coupling=.6)
    matrices = data.operator_matrices
    for key in ("wrong_joint_basis", "permuted_joint_basis"):
        assert matrices[key].shape == matrices["joint_basis"].shape
        assert not torch.allclose(matrices[key], matrices["joint_basis"])
    # A permutation is a rearrangement of the same rows; a misspecified
    # operator is not.  Sorting each column makes that difference explicit.
    joint_sorted = matrices["joint_basis"].sort(dim=0).values
    assert torch.allclose(matrices["permuted_joint_basis"].sort(dim=0).values,
                          joint_sorted, atol=1e-6)
    assert not torch.allclose(matrices["wrong_joint_basis"].sort(dim=0).values,
                              joint_sorted, atol=1e-4)
    # The misspecified control keeps a genuinely close low-frequency space,
    # which is why it must never be reported as the destructive control.
    assert 0 < data.metadata["wrong_operator_subspace_residual"] < .5


def test_table_group_penalty_matches_spectral_energy_on_eigenvectors():
    data = _small_dataset(coupling=.3)
    matrices = data.operator_matrices
    stiffness, mass = matrices["joint_stiffness"], matrices["joint_mass"]
    values, vectors = generalized_eigenpairs(stiffness, mass, 6)
    penalty = sobolev_penalty_operator(stiffness, mass, 1.5, values)
    span = values[1] - values[0]
    for k in range(6):
        column = vectors[:, k].float()
        expected = float((1 + (values[k] - values[0]) / span) ** 1.5)
        assert abs(float(column @ (penalty @ column)) - expected) < 1e-3


def test_table_group_is_accepted_and_reproducible():
    data = _small_dataset()
    matrices = data.operator_matrices
    values, _ = generalized_eigenpairs(matrices["joint_stiffness"],
                                       matrices["joint_mass"], 2)
    specs = _joint_specs(data)
    specs[1] = GroupFactorSpec(
        "table", specs[1].rank, specs[1].size,
        penalty_operator=sobolev_penalty_operator(
            matrices["joint_stiffness"], matrices["joint_mass"], 1.5, values),
        name="space")
    index = grouped_indices(data.shape, data.groups)
    truth = data.values.flatten()
    generator = torch.Generator().manual_seed(7)
    observed = torch.randperm(len(truth), generator=generator)[:600]
    means = []
    for _ in range(2):
        # The runner seeds before construction; initialization must be part of
        # what a seed reproduces, not only the optimizer trajectory.
        torch.manual_seed(7)
        model = GroupedOperatorTucker(specs, device="cpu")
        model.fit(index[observed], truth[observed], steps=30, seed=7)
        means.append(model.predict(index).mean)
    assert torch.equal(means[0], means[1])
    assert model.spatial_latent_dimension() == 5
