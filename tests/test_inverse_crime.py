"""The truth may be solved on a mesh the learner's operator was not built from.

An inverse-crime objection asks whether the reported advantage survives when the
data-generating discretization is not the same object the method is handed.
These tests pin the machinery that makes the answer checkable: exact P1
interpolation, bit-for-bit reproduction at the frozen default, and a genuinely
different discretization once refinement is switched on.
"""

import torch

from geoaware.irregular_fem import (UNIT_SQUARE, Hole, build_mesh,
                                    interpolate_p1)
from geoaware.irregular_green_data import (WALL_LAYOUTS, irregular_field_tensor,
                                           wall_field_tensor)
from geoaware.operator_diagnostics import product_projection_residual


def test_p1_interpolation_is_exact_on_affine_fields():
    """P1 elements reproduce affine functions, so interpolation must too."""
    mesh = build_mesh(14, (), polygon=UNIT_SQUARE, seed=0)
    affine = 1.3 - .7 * mesh.nodes[:, 0] + 2.1 * mesh.nodes[:, 1]
    points = torch.tensor([[.25, .3], [.5, .5], [.72, .18], [.1, .9]],
                          dtype=torch.float64)
    expected = 1.3 - .7 * points[:, 0] + 2.1 * points[:, 1]
    assert torch.allclose(interpolate_p1(mesh, affine, points), expected,
                          atol=1e-10)
    # Leading axes are carried through untouched.
    stacked = torch.stack([affine, 2 * affine]).reshape(1, 2, -1)
    out = interpolate_p1(mesh, stacked, points)
    assert out.shape == (1, 2, len(points))
    assert torch.allclose(out[0, 1], 2 * expected, atol=1e-10)


def _wall(resolution=12, **kwargs):
    return wall_field_tensor(WALL_LAYOUTS["sealed_4"], resolution=resolution,
                             n_scenarios=6, n_time=6, basis_cutoff=10,
                             truth_modes=30, **kwargs)


def test_default_refinement_reproduces_the_frozen_tensor_exactly():
    assert torch.equal(_wall().values, _wall(truth_refinement=1).values)
    assert _wall().metadata["inverse_crime"].startswith("present")


def test_refined_truth_is_a_different_discretization_of_the_same_field():
    coarse, refined = _wall(), _wall(truth_refinement=2)
    assert coarse.shape == refined.shape
    assert refined.metadata["truth_mesh_hash"] != coarse.metadata["mesh_hash"]
    assert refined.metadata["truth_mesh_nodes"] > refined.metadata["n_free_nodes"]
    assert refined.metadata["inverse_crime"].startswith("avoided")
    # The learner-side operators are untouched: only the data moved.
    for name in ("fem_correct", "topology_erased", "bounding_box_product"):
        assert torch.equal(coarse.operator_matrices[f"{name}_basis"],
                           refined.operator_matrices[f"{name}_basis"])
    # Same physics, different numerics: strongly correlated, not identical.
    a = coarse.values.flatten().double()
    b = refined.values.flatten().double()
    assert not torch.allclose(a, b)
    correlation = float((a * b).mean() / (a.std() * b.std()))
    assert correlation > .9


def test_the_geometry_gap_survives_the_refined_truth():
    """The mechanism the claim rests on, re-measured without the inverse crime.

    Paper resolution is used deliberately.  The baffles are thinner than one
    coarse element, so below resolution 18 the learner's own operator represents
    them as a jagged single-element layer and the gap against a refined truth is
    not reliable — a caveat about under-resolved barriers rather than about the
    method, and one the sweep in the iteration log reports rather than hides.
    """
    refined = _wall(resolution=18, truth_refinement=2)
    def residual(name):
        basis = refined.operator_matrices[f"{name}_basis"]
        return product_projection_residual(refined.values, [None, None, basis])
    assert residual("topology_erased") > 2 * residual("fem_correct")


def test_polygonal_domains_accept_the_refined_truth():
    holes = (Hole((.5, .5), .20),)
    common = dict(resolution=12, n_scenarios=6, n_time=6, basis_cutoff=10,
                  truth_modes=30)
    plain = irregular_field_tensor(holes, **common)
    refined = irregular_field_tensor(holes, truth_refinement=2, **common)
    assert plain.shape == refined.shape
    assert torch.equal(plain.operator_matrices["fem_correct_basis"],
                       refined.operator_matrices["fem_correct_basis"])
    assert torch.isfinite(refined.values).all()
    assert not torch.allclose(plain.values, refined.values)
    a, b = plain.values.flatten().double(), refined.values.flatten().double()
    assert float((a * b).mean() / (a.std() * b.std())) > .9
