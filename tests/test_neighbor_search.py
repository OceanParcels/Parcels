import pytest
import numpy as np

from parcels.interaction.neighborsearch import BruteFlatNeighborSearch
from parcels.interaction.neighborsearch import BruteSphericalNeighborSearch
from parcels.interaction.neighborsearch import HashFlatNeighborSearch
from parcels.interaction.neighborsearch import HashSphericalNeighborSearch
from parcels.interaction.neighborsearch import KDTreeFlatNeighborSearch
from parcels.interaction.neighborsearch.basehash import BaseHashNeighborSearch


def compare_results_by_idx(instance, particle_idx, ref_result, active_idx=None):
    cur_neigh, _ = instance.find_neighbors_by_idx(particle_idx)
    assert isinstance(cur_neigh, np.ndarray)
    assert len(cur_neigh) == len(set(cur_neigh))
    if active_idx is None:
        active_idx = np.arange(instance._values.shape[1])
    if isinstance(instance, BaseHashNeighborSearch):
        instance.consistency_check()
    for neigh in cur_neigh:
        assert neigh in active_idx
    assert set(cur_neigh) <= set(active_idx)
    neigh_by_coor, _ = instance.find_neighbors_by_coor(
        instance._values[:, particle_idx])
    assert np.allclose(cur_neigh, neigh_by_coor)

    assert isinstance(cur_neigh, np.ndarray)
    assert set(ref_result) == set(cur_neigh)


@pytest.mark.parametrize(
    "test_class", [KDTreeFlatNeighborSearch, HashFlatNeighborSearch,
                   BruteFlatNeighborSearch])
def test_flat_neighbors(test_class):
    np.random.seed(129873)
    ref_class = BruteFlatNeighborSearch
    n_particle = 1000
    positions = np.random.rand(n_particle*3).reshape(3, n_particle)
    ref_instance = ref_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)
    test_instance = test_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)
    ref_instance.rebuild(positions)
    test_instance.rebuild(positions)

    for particle_idx in np.random.choice(positions.shape[1], 100, replace=False):
        ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
        compare_results_by_idx(test_instance, particle_idx, ref_result)


def create_spherical_positions(n_particles, max_depth=100000):
    yrange = 2*np.random.rand(n_particles)
    lat = 180*(np.arccos(1-yrange)-0.5*np.pi)/np.pi
    lon = 360*np.random.rand(n_particles)
    depth = max_depth*np.random.rand(n_particles)
    return np.array((depth, lat, lon))


def create_flat_positions(n_particle):
    return np.random.rand(n_particle*3).reshape(3, n_particle)


@pytest.mark.parametrize(
    "test_class", [BruteSphericalNeighborSearch, HashSphericalNeighborSearch])
def test_spherical_neighbors(test_class):
    np.random.seed(9837452)
    ref_class = BruteSphericalNeighborSearch

    positions = create_spherical_positions(10000, max_depth=100000)
    ref_instance = ref_class(inter_dist_vert=100000,
                             inter_dist_horiz=1000000)
    test_instance = test_class(inter_dist_vert=100000,
                               inter_dist_horiz=1000000)
    ref_instance.rebuild(positions)
    test_instance.rebuild(positions)

    for particle_idx in np.random.choice(positions.shape[1], 100, replace=False):
        ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
        compare_results_by_idx(test_instance, particle_idx, ref_result)


@pytest.mark.parametrize(
    "test_class", [KDTreeFlatNeighborSearch, HashFlatNeighborSearch,
                   BruteFlatNeighborSearch])
def test_flat_update(test_class):
    np.random.seed(9182741)
    n_particle = 1000
    n_test_particle = 10
    n_active_mask = 10
    ref_class = BruteFlatNeighborSearch
    ref_instance = ref_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)
    test_instance = test_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)

    for i in range(n_active_mask):
        positions = create_flat_positions(n_particle) + 10*np.random.rand()
        if i == 0:
            active_mask = None
        else:
            active_mask = np.random.rand(n_particle) > 0.5
        ref_instance.update_values(positions, active_mask)
        test_instance.update_values(positions, active_mask)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            continue
        test_particles = np.random.choice(
            active_idx, size=min(n_test_particle, len(active_idx)), replace=False)
        for particle_idx in test_particles:
            ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
            compare_results_by_idx(test_instance, particle_idx, ref_result,
                                   active_idx=active_idx)


@pytest.mark.parametrize(
    "test_class", [BruteSphericalNeighborSearch, HashSphericalNeighborSearch])
def test_spherical_update(test_class):
    np.random.seed(9182741)
    n_particle = 1000
    n_test_particle = 10
    n_active_mask = 10
    ref_class = BruteSphericalNeighborSearch

    ref_instance = ref_class(inter_dist_vert=100000, inter_dist_horiz=1000000)
    test_instance = test_class(inter_dist_vert=100000, inter_dist_horiz=1000000)

    for _ in range(n_active_mask):
        positions = create_spherical_positions(n_particle)
        active_mask = np.random.rand(n_particle) > 0.5
        ref_instance.update_values(positions, active_mask)
        test_instance.update_values(positions, active_mask)

        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            continue
        test_particles = np.random.choice(
            active_idx, size=min(n_test_particle, len(active_idx)), replace=False)
        for particle_idx in test_particles:
            ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
            compare_results_by_idx(test_instance, particle_idx, ref_result, active_idx=active_idx)
