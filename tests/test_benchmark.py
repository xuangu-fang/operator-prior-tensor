"""The main benchmark: the properties every reported number depends on.

These are not smoke tests.  Each one pins a claim that a reader would otherwise
have to take on trust: that the geometry-aware and geometry-blind models differ
in exactly one thing, that the negative control really is a control, that the
CP form is the same machinery with a different core, and that the mode screen
uses no information it is not entitled to.
"""

import math

import torch

from geoaware.als_baselines import cp_als, tucker_als
from geoaware.benchmark import FAMILIES, build_family
from geoaware.grouped_operator_tucker import (GroupFactorSpec,
                                              GroupedOperatorTucker,
                                              grouped_indices)
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import (mode_observability, observable_modes,
                                           product_projection_residual)

SMALL = dict(n_scenarios=6, n_time=6, basis_cutoff=8, truth_modes=20)


def _residual(data, name):
    basis = data.operator_matrices[f"{name}_basis"]
    return product_projection_residual(data.values, [None, None, basis])


def test_a_barrier_free_domain_leaves_nothing_to_know():
    """The control the whole design rests on: no geometry, no advantage."""
    data = build_family("plane_barrier", "open", resolution=20, **SMALL)
    assert torch.allclose(data.operator_matrices["geometry_operator_basis"],
                          data.operator_matrices["blind_operator_basis"],
                          atol=1e-6)
    assert abs(_residual(data, "geometry_operator")
               - _residual(data, "blind_operator")) < 1e-6


def test_the_blind_operator_differs_only_in_the_geometry_it_sees():
    """Same nodes, same assembly, same everything but the obstacle layout."""
    plain = build_family("plane_barrier", "open", resolution=20, **SMALL)
    walled = build_family("plane_barrier", "sealed_4", resolution=20, **SMALL)
    assert plain.shape == walled.shape
    assert plain.metadata["mesh_hash"] == walled.metadata["mesh_hash"]
    assert torch.equal(plain.operator_matrices["coordinates"],
                       walled.operator_matrices["coordinates"])
    # Removing the walls from the operator returns the barrier-free basis.
    assert torch.allclose(walled.operator_matrices["blind_operator_basis"],
                          plain.operator_matrices["geometry_operator_basis"],
                          atol=1e-6)
    assert not torch.allclose(walled.operator_matrices["geometry_operator_basis"],
                              walled.operator_matrices["blind_operator_basis"])
    assert _residual(walled, "blind_operator") > 3 * _residual(walled, "geometry_operator")


def test_the_sphere_control_is_the_chart_and_it_fails_structurally():
    """On a bare sphere the geometry is the manifold, so the control is a chart.

    The reported cutoff is used rather than the smaller one the other tests
    share.  A shallow-water field is oscillatory and keeps energy well above the
    slowest modes for all time, so at eight columns *both* bases are starved and
    the comparison measures the truncation instead of the geometry.
    """
    options = dict(n_scenarios=6, n_time=6, basis_cutoff=16, truth_modes=40)
    data = build_family("sphere", "open_ocean", resolution=3, **options)
    finer = build_family("sphere", "open_ocean", resolution=4, **options)
    assert data.metadata["dynamics"] == "wave"
    assert "shallow water" in data.metadata["pde"]
    for tensor in (data, finer):
        assert _residual(tensor, "blind_operator") > 3 * _residual(
            tensor, "geometry_operator")
    # Refining the mesh does not repair a singular chart.
    assert abs(_residual(data, "blind_operator")
               - _residual(finer, "blind_operator")) < .05


def test_a_cp_core_is_the_same_model_with_a_smaller_core():
    data = build_family("plane_barrier", "sealed_4", resolution=20, **SMALL)
    matrices = data.operator_matrices
    specs = lambda rank: [
        GroupFactorSpec("table", rank, data.shape[0], name="scenario"),
        GroupFactorSpec("operator", rank, data.shape[1],
                        basis=matrices["time_basis"],
                        eigenvalues=matrices["time_eigenvalues"], name="time"),
        GroupFactorSpec("operator", rank, data.shape[2],
                        basis=matrices["geometry_operator_basis"],
                        eigenvalues=matrices["geometry_operator_eigenvalues"],
                        name="node")]
    torch.manual_seed(0)
    tucker = GroupedOperatorTucker(specs(4), device="cpu")
    torch.manual_seed(0)
    cp = GroupedOperatorTucker(specs(4), device="cpu", core="diagonal")
    assert tucker.core.numel() == 4 ** 3
    assert cp.core.numel() == 4
    index = grouped_indices(data.shape, ((0,), (1,), (2,)))[:64]
    assert GroupedOperatorTucker.design(
        index, tucker.factor_tables()).shape[1] == 4 ** 3
    assert GroupedOperatorTucker.design(
        index, cp.factor_tables(), diagonal=True).shape[1] == 4
    assert cp(index).shape == (64,)
    # A diagonal core is only defined when every group carries the same rank.
    try:
        GroupedOperatorTucker(specs(4)[:2] + [
            GroupFactorSpec("table", 5, data.shape[2], name="node")],
            device="cpu", core="diagonal")
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched ranks must be rejected for a CP core")


def test_the_mode_screen_reads_the_mask_and_never_the_data():
    """It may use where the sensors are; it may not use what they measured."""
    data = build_family("plane_barrier", "sealed_4", resolution=20, **SMALL)
    basis = data.operator_matrices["geometry_operator_basis"]
    split = make_observation_split(data, .1, "spatial_sensors", 41,
                                   sensor_axes=(2,))
    seen = split.observed.reshape(data.shape).reshape(-1, data.shape[2]).any(0)

    visibility = mode_observability(basis, seen)
    assert visibility.shape[0] == basis.shape[1]
    # Scaling the measurements cannot move a mask-only quantity.
    scaled = build_family("plane_barrier", "sealed_4", resolution=20,
                          contrast=.9, **SMALL)
    assert not torch.allclose(scaled.values, data.values)
    assert torch.allclose(
        mode_observability(scaled.operator_matrices["geometry_operator_basis"],
                           seen), visibility)
    # Observing everything makes every mode exactly as visible as average.
    everywhere = torch.ones(data.shape[2], dtype=torch.bool)
    assert torch.allclose(mode_observability(basis, everywhere),
                          torch.ones_like(visibility))
    assert len(observable_modes(basis, everywhere)) == basis.shape[1]


def test_classical_completion_is_undefined_under_sensor_sampling():
    """Not a weak baseline -- an undefined one, which is the point.

    Under spatial sensors an unobserved node appears in no observed entry, so
    its factor row is constrained by nothing at all.  That is the gap a geometry
    prior fills, and it is worth asserting rather than asserting a margin.
    """
    # Twelve scenarios and twelve times rather than six: at six, a rank-four CP
    # has more parameters than there are observed entries, so it would fail for
    # a reason that has nothing to do with geometry and the comparison below
    # would prove nothing.
    data = build_family("plane_barrier", "sealed_4", resolution=20,
                        n_scenarios=12, n_time=12, basis_cutoff=8,
                        truth_modes=20)
    split = make_observation_split(data, .1, "spatial_sensors", 41,
                                   sensor_axes=(2,))
    seen = split.observed.reshape(data.shape).reshape(-1, data.shape[2]).any(0)
    assert int(seen.sum()) < data.shape[2]

    truth = data.values
    held = split.held_out.reshape(truth.shape)
    predicted = cp_als(truth, split.observed, 4, seed=0, n_iter_max=50)
    unseen = ~seen
    # Nothing is recovered where no sensor ever looked.
    assert float(predicted[..., unseen].abs().max()) < 1e-6

    under_random = make_observation_split(data, .1, "random", 41)
    recovered = cp_als(truth, under_random.observed, 4, seed=0, n_iter_max=50)
    held_random = under_random.held_out.reshape(truth.shape)
    error = float((recovered[held_random] - truth[held_random]).square().mean().sqrt()
                  / truth[held_random].std())
    # Under random entries the same routine is a real competitor, not a stub.
    assert error < .5


def test_every_family_builds_and_records_what_it_did():
    for family, spec in FAMILIES.items():
        resolution = {"plane_barrier": 20, "plane_domain": 20,
                      "volume_barrier": 7, "sphere": 3}[family]
        for layout in spec.layouts:
            data = build_family(family, layout, resolution=resolution, **SMALL)
            assert torch.isfinite(data.values).all()
            assert data.metadata["family"] == family
            assert data.metadata["layout"] == layout
            assert data.metadata["n_nodes"] == data.shape[2]
            assert data.metadata["operator_information_tier"].startswith("geometry")
            for name in ("geometry_operator", "blind_operator", "flat_chart",
                         "permuted"):
                assert f"{name}_basis" in data.operator_matrices
            # The destructive control shares the columns and loses the alignment.
            assert torch.allclose(
                data.operator_matrices["permuted_basis"].sort(dim=0).values,
                data.operator_matrices["geometry_operator_basis"].sort(dim=0).values,
                atol=1e-6)
