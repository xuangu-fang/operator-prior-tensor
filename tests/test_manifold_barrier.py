"""P1 on simplices of any dimension, and the two new scenarios built on it.

The finite-element code is checked against quantities with known values -- the
volume of a cube, the area and the analytic Laplace-Beltrami spectrum of a
sphere -- rather than against itself, because everything downstream rests on it.
"""

import math

import torch

from geoaware.manifold_barrier_data import (BOX_LAYOUTS, SPHERE_LAYOUTS,
                                            barrier_coefficient,
                                            barrier_field_tensor)
from geoaware.operator_diagnostics import (generalized_eigenpairs,
                                           product_projection_residual)
from geoaware.simplex_fem import (assemble, build_box_mesh, build_sphere_mesh,
                                  cell_volumes)


def test_tetrahedral_elements_reproduce_the_cube():
    mesh = build_box_mesh(7, seed=0)
    stiffness, mass = assemble(mesh, 1.)
    assert mesh.degree == 3
    assert abs(mesh.volume() - 1.) < 1e-9
    assert abs(float(mass.sum()) - 1.) < 1e-9
    assert torch.allclose(stiffness, stiffness.T, atol=1e-10)
    constant = torch.ones(mesh.n_nodes, dtype=torch.float64)
    assert abs(float(constant @ stiffness @ constant)) < 1e-9
    assert float(torch.linalg.eigvalsh(stiffness).min()) > -1e-9
    assert float(cell_volumes(mesh).min()) > 0


def test_surface_elements_reproduce_the_sphere_spectrum():
    """The Laplace-Beltrami eigenvalues of the unit sphere are ``l(l+1)``."""
    mesh = build_sphere_mesh(3)
    stiffness, mass = assemble(mesh, 1.)
    assert mesh.degree == 2 and mesh.nodes.shape[1] == 3
    assert abs(mesh.volume() - 4 * math.pi) < .1
    values, vectors = generalized_eigenpairs(stiffness, mass, 9)
    assert abs(float(values[0])) < 1e-8                       # the constant mode
    for index in (1, 2, 3):
        assert abs(float(values[index]) - 2.) < .05           # l = 1, three-fold
    for index in (4, 5, 6, 7, 8):
        assert abs(float(values[index]) - 6.) < .15           # l = 2, five-fold
    identity = vectors.T @ mass @ vectors
    assert torch.allclose(identity, torch.eye(9, dtype=identity.dtype), atol=1e-8)


def test_the_lat_lon_chart_fails_structurally_not_by_resolution():
    """More basis columns and a finer mesh do not repair a singular chart.

    Every longitude meets at a pole, so a separable cosine basis of the
    ``(theta, phi)`` rectangle cannot represent a field that is smooth across
    one.  This is the control the sphere setting exists to make.
    """
    def residual(subdivisions, cutoff, name):
        data = barrier_field_tensor((), geometry="sphere",
                                    subdivisions=subdivisions, n_scenarios=8,
                                    n_time=8, basis_cutoff=cutoff,
                                    truth_modes=40, dynamics="wave")
        basis = data.operator_matrices[f"{name}_basis"]
        return product_projection_residual(data.values, [None, None, basis])

    coarse = residual(3, 10, "lat_lon_product")
    finer = residual(3, 16, "lat_lon_product")
    finest = residual(4, 16, "lat_lon_product")
    operator = residual(4, 16, "fem_correct")
    # The chart basis barely moves while the operator basis improves sharply.
    assert abs(coarse - finest) < .15
    assert finest > 3 * operator


def test_a_bare_sphere_still_has_geometry_to_know():
    """With no land the two operator bases coincide, and both beat Euclidean."""
    data = barrier_field_tensor((), geometry="sphere", subdivisions=3,
                                n_scenarios=8, n_time=8, basis_cutoff=12,
                                truth_modes=40, dynamics="wave")
    matrices = data.operator_matrices
    assert torch.allclose(matrices["fem_correct_basis"],
                          matrices["topology_erased_basis"], atol=1e-9)

    def residual(name):
        return product_projection_residual(
            data.values, [None, None, matrices[f"{name}_basis"]])
    assert residual("bounding_box_product") > residual("fem_correct")
    assert residual("lat_lon_product") > residual("fem_correct")


def test_wave_dynamics_keep_energy_out_of_the_slowest_modes():
    """Diffusion collapses onto the slow end; the wave propagator must not.

    R5a recorded that a benchmark whose fields decay onto the operator's own
    leading eigenvectors is degenerately easy.  The shallow-water setting exists
    partly to avoid that, so the property is asserted rather than assumed.
    """
    common = dict(geometry="sphere", subdivisions=3, n_scenarios=8, n_time=8,
                  basis_cutoff=12, truth_modes=40)
    parabolic = barrier_field_tensor((), dynamics="diffusion", **common)
    hyperbolic = barrier_field_tensor((), dynamics="wave", **common)
    assert parabolic.metadata["dynamics"] == "diffusion"
    assert "shallow water" in hyperbolic.metadata["pde"]

    def late_share(data, count=4):
        """Energy in the slowest few modes, on the last time slice."""
        basis = data.operator_matrices["fem_correct_basis"].double()
        mass = data.operator_matrices["mass"].double()
        final = data.values.double()[:, -1, :]
        projected = final @ mass @ basis[:, :count]
        return float(projected.square().sum() / final.square().sum().clamp_min(1e-12))

    assert late_share(parabolic) > .9
    assert late_share(hyperbolic) < late_share(parabolic)


def test_barrier_material_knows_the_layout_and_not_the_background():
    mesh = build_box_mesh(7, seed=0)
    centroids = mesh.centroids()
    barriers = BOX_LAYOUTS["chamber"]
    truth = barrier_coefficient(centroids, barriers, .5, background=True)
    learner = barrier_coefficient(centroids, barriers, .5, background=False)
    inside = torch.zeros(len(centroids), dtype=torch.bool)
    for barrier in barriers:
        inside |= barrier.contains(centroids)
    assert bool(inside.any())
    assert torch.allclose(truth[inside], learner[inside])
    assert torch.allclose(learner[~inside], torch.ones_like(learner[~inside]))
    assert not torch.allclose(truth[~inside], learner[~inside])


def test_every_layout_shares_one_mesh_and_node_set():
    for geometry, catalogue, options in (
            ("box", BOX_LAYOUTS, dict(resolution=7)),
            ("sphere", SPHERE_LAYOUTS, dict(subdivisions=2))):
        reference = None
        for layout, barriers in catalogue.items():
            data = barrier_field_tensor(barriers, geometry=geometry,
                                        n_scenarios=6, n_time=6,
                                        basis_cutoff=8, truth_modes=20,
                                        **options)
            if reference is None:
                reference = data
                continue
            assert data.shape == reference.shape
            assert data.metadata["mesh_hash"] == reference.metadata["mesh_hash"]
            assert torch.equal(data.operator_matrices["mass"],
                               reference.operator_matrices["mass"])
            assert torch.equal(data.operator_matrices["blind_stiffness"],
                               reference.operator_matrices["blind_stiffness"])
