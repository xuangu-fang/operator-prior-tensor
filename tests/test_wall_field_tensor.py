import torch

from geoaware.irregular_green_data import (WALL_LAYOUTS, ArcWall, Wall,
                                           wall_coefficient, wall_field_tensor)
from geoaware.irregular_fem import UNIT_SQUARE, build_mesh, triangle_centroids
from geoaware.masks import make_observation_split
from geoaware.operator_diagnostics import product_projection_residual


def _small(layout="chamber", **kwargs):
    options = dict(resolution=12, n_scenarios=8, n_time=8, basis_cutoff=16,
                   truth_modes=30)
    options.update(kwargs)
    return wall_field_tensor(WALL_LAYOUTS[layout], **options)


def test_every_layout_shares_one_mesh_and_node_set():
    """The whole point of walls-inside-a-mesh: controls need no interpolation."""
    reference = _small("open")
    for layout in WALL_LAYOUTS:
        data = _small(layout)
        assert data.shape == reference.shape
        assert data.metadata["mesh_hash"] == reference.metadata["mesh_hash"]
        assert torch.equal(data.operator_matrices["coordinates"],
                           reference.operator_matrices["coordinates"])
        assert torch.equal(data.operator_matrices["mass"],
                           reference.operator_matrices["mass"])


def test_walls_change_the_operator_but_not_the_blind_one():
    open_data, walled = _small("open"), _small("sealed_4")
    aware = "nominal_stiffness"
    assert torch.equal(open_data.operator_matrices["blind_stiffness"],
                       walled.operator_matrices["blind_stiffness"])
    assert not torch.allclose(open_data.operator_matrices[aware],
                              walled.operator_matrices[aware])
    # With no wall the two operators coincide, so the 2x2 ablation degenerates
    # exactly where it should: there is no geometry to know.
    assert torch.allclose(open_data.operator_matrices[aware],
                          open_data.operator_matrices["blind_stiffness"], atol=1e-9)


def test_learner_material_knows_walls_and_not_the_background():
    mesh = build_mesh(12, (), polygon=UNIT_SQUARE, seed=0)
    centroids = triangle_centroids(mesh)
    walls = WALL_LAYOUTS["chamber"]
    truth = wall_coefficient(centroids, walls, .5, background=True)
    learner = wall_coefficient(centroids, walls, .5, background=False)
    inside = torch.zeros(len(centroids), dtype=torch.bool)
    for wall in walls:
        inside |= wall.contains(centroids)
    assert bool(inside.any())
    # Identical inside the barriers: the wall layout is known metadata.
    assert torch.allclose(truth[inside], learner[inside])
    # Different outside: the smooth background material is not.
    assert not torch.allclose(truth[~inside], learner[~inside])
    assert torch.allclose(learner[~inside], torch.ones_like(learner[~inside]))


def test_learner_bases_do_not_move_when_only_the_truth_material_changes():
    low, high = _small(contrast=.1), _small(contrast=.9)
    assert not torch.allclose(low.values, high.values)
    for name in ("fem_correct", "topology_erased", "bounding_box_product"):
        assert torch.equal(low.operator_matrices[f"{name}_basis"],
                           high.operator_matrices[f"{name}_basis"])


def test_barriers_raise_the_geometry_blind_bias_floor():
    """The mechanism the design depends on, asserted rather than assumed."""
    def residual(data, name):
        basis = data.operator_matrices[f"{name}_basis"]
        return product_projection_residual(data.values, [None, None, basis])

    open_data = _small("open")
    assert abs(residual(open_data, "fem_correct")
               - residual(open_data, "topology_erased")) < 1e-6

    sealed = _small("sealed_4")
    aware = residual(sealed, "fem_correct")
    blind = residual(sealed, "topology_erased")
    # The margin is much larger at paper resolution; five-fold is the level that
    # is safely reproducible on the coarse mesh a unit test can afford.
    assert blind > 5 * aware
    assert blind > .1


def test_arc_walls_are_curved_and_leave_their_aperture_open():
    arc = ArcWall((.5, .5), .3, .02, gap=(1.25, 1.90))
    on_ring = torch.tensor([[.8, .5], [.5, .8], [.2, .5], [.5, .2]],
                           dtype=torch.float64)
    assert bool(arc.contains(on_ring)[0])
    assert not bool(arc.contains(on_ring)[1])          # inside the aperture
    assert not bool(arc.contains(torch.tensor([[.5, .5]], dtype=torch.float64))[0])
    slab = Wall((.48, .52), (0., 1.))
    assert bool(slab.contains(torch.tensor([[.5, .3]], dtype=torch.float64))[0])
    assert not bool(slab.contains(torch.tensor([[.1, .3]], dtype=torch.float64))[0])


def test_sensor_mask_observes_whole_trajectories_at_a_few_nodes():
    data = _small()
    split = make_observation_split(data, .1, "spatial_sensors", seed=41,
                                   sensor_axes=(2,))
    cube = split.observed.reshape(data.shape)
    per_node = cube.reshape(-1, data.shape[2]).sum(0)
    full = data.shape[0] * data.shape[1]
    assert torch.all((per_node == 0) | (per_node == full))
    assert 0 < int((per_node > 0).sum()) < data.shape[2]
    assert not bool((split.observed & split.held_out).any())
