import math

import torch

from geoaware.irregular_fem import (Hole, assemble_p1, build_mesh, free_nodes,
                                    restrict, triangle_centroids)
from geoaware.irregular_green_data import irregular_green_tensor
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import generalized_eigenpairs

HOLES = (Hole((.32, .62), .15), Hole((.68, .33), .13))


def _small(holes=HOLES, **kwargs):
    options = dict(resolution=12, n_time=8, n_sources=10, basis_cutoff=12,
                   truth_modes=24, contrast=.3, time_span=(.15, 3.))
    options.update(kwargs)
    return irregular_green_tensor(holes, **options)


def test_mesh_excludes_hole_interiors_and_tags_the_rims():
    mesh = build_mesh(16, HOLES, seed=0)
    points = mesh.nodes.numpy()
    for hole in HOLES:
        assert hole.signed_distance(points).min() > -1e-9
    assert int(mesh.hole_boundary.sum()) > 0
    assert int(mesh.outer_boundary.sum()) > 0
    assert not bool((mesh.hole_boundary & mesh.outer_boundary).any())
    # Rim membership is recoverable from geometry alone, so the tag can be
    # audited rather than trusted.
    for index in torch.where(mesh.hole_boundary)[0]:
        hole = HOLES[int(mesh.hole_index[index])]
        assert abs(float(torch.from_numpy(
            hole.signed_distance(mesh.nodes[index][None].numpy()))[0])) < 1e-9
    assert abs(mesh.area() - (1 - sum(math.pi * h.radius ** 2 for h in HOLES))) < .02


def test_finite_element_matrices_are_symmetric_positive_and_consistent():
    mesh = build_mesh(14, HOLES, seed=0)
    stiffness, mass = assemble_p1(mesh, 1.)
    assert torch.allclose(stiffness, stiffness.T, atol=1e-12)
    assert torch.allclose(mass, mass.T, atol=1e-12)
    # Total mass is the domain area, and a constant field has zero energy under
    # a pure Neumann stiffness matrix.
    assert abs(float(mass.sum()) - mesh.area()) < 1e-9
    constant = torch.ones(mesh.n_nodes, dtype=torch.float64)
    assert abs(float(constant @ stiffness @ constant)) < 1e-9
    assert float(torch.linalg.eigvalsh(stiffness).min()) > -1e-9


def test_dirichlet_rims_are_eliminated_and_raise_the_spectrum():
    mesh = build_mesh(16, HOLES, seed=0)
    stiffness, mass = assemble_p1(mesh, 1.)
    keep = free_nodes(mesh, "dirichlet", "neumann")
    assert len(keep) == mesh.n_nodes - int(mesh.hole_boundary.sum())
    values, vectors = generalized_eigenpairs(restrict(stiffness, keep),
                                             restrict(mass, keep), 8)
    identity = vectors.T @ restrict(mass, keep) @ vectors
    assert torch.allclose(identity, torch.eye(8, dtype=identity.dtype), atol=1e-8)
    residual = (restrict(stiffness, keep) @ vectors
                - restrict(mass, keep) @ vectors @ torch.diag(values))
    assert float(residual.norm()) < 1e-7
    # Constraining the rims removes the constant null mode.
    assert float(values[0]) > 1.
    insulating = free_nodes(mesh, "neumann", "neumann")
    assert len(insulating) == mesh.n_nodes


def test_tensor_is_reproducible_and_records_its_provenance():
    first, repeat = _small(), _small()
    assert torch.equal(first.values, repeat.values)
    assert first.metadata["mesh_hash"] == repeat.metadata["mesh_hash"]
    assert first.metadata["stiffness_checksum"] == repeat.metadata["stiffness_checksum"]
    assert torch.equal(first.operator_matrices["source_nodes"],
                       repeat.operator_matrices["source_nodes"])
    assert first.metadata["operator_information_tier"].startswith("geometry")
    assert first.metadata["n_free_nodes"] == first.shape[1]
    assert first.shape[2] == first.metadata["n_sources"]
    assert torch.isfinite(first.values).all()
    assert abs(float(first.values.mean())) < 1e-5

    other = _small(holes=(Hole((.35, .55), .16),))
    assert other.metadata["mesh_hash"] != first.metadata["mesh_hash"]


def test_every_spatial_basis_shares_the_node_set_and_stays_distinct():
    data = _small()
    matrices = data.operator_matrices
    names = ("fem_correct", "topology_erased", "bounding_box_product", "permuted")
    shapes = {name: tuple(matrices[f"{name}_basis"].shape) for name in names}
    assert len(set(shapes.values())) == 1, shapes
    assert shapes["fem_correct"][0] == data.shape[1]
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            assert not torch.allclose(matrices[f"{names[left]}_basis"],
                                      matrices[f"{names[right]}_basis"])
    # The permuted control is a row shuffle of the geometry-aware basis: same
    # columns and eigenvalues, no index-operator alignment.
    assert torch.allclose(matrices["permuted_basis"].sort(dim=0).values,
                          matrices["fem_correct_basis"].sort(dim=0).values, atol=1e-6)
    assert torch.equal(matrices["permuted_eigenvalues"],
                       matrices["fem_correct_eigenvalues"])


def test_learner_bases_never_read_the_truth_material():
    """Changing only the truth material must leave every learner basis fixed."""
    low, high = _small(contrast=.1), _small(contrast=.9)
    assert not torch.allclose(low.values, high.values)
    for name in ("fem_correct", "topology_erased", "bounding_box_product"):
        assert torch.equal(low.operator_matrices[f"{name}_basis"],
                           high.operator_matrices[f"{name}_basis"])
    assert torch.equal(low.operator_matrices["time_basis"],
                       high.operator_matrices["time_basis"])


def test_masks_stay_valid_on_the_mesh_tensor():
    data = _small()
    split = make_observation_split(data, .1, "receiver_fibers", seed=41)
    cube = split.observed.reshape(data.shape)
    # A receiver fiber observes every receiver node at a fixed (time, source).
    assert torch.all((cube.sum(1) == 0) | (cube.sum(1) == data.shape[1]))
    assert not bool((split.observed & split.held_out).any())
    assert bool((split.observed | split.held_out).all())
