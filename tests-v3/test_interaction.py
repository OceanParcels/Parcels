import numpy as np
import pytest

from parcels import Field, FieldSet, ParticleSet
from parcels._core.particle import InteractionParticle, Variable
from parcels.interaction.neighborsearch import (
    BruteFlatNeighborSearch,
    BruteSphericalNeighborSearch,
    HashFlatNeighborSearch,
    HashSphericalNeighborSearch,
    KDTreeFlatNeighborSearch,
)
from parcels.interaction.neighborsearch.basehash import BaseHashNeighborSearch
from parcels.kernels import AdvectionRK4
from parcels.kernels.interaction import (
    AsymmetricAttraction,
    MergeWithNearestNeighbor,
    NearestNeighborWithinRange,
)
from tests.common_kernels import DoNothing
from tests.utils import create_fieldset_unit_mesh, create_flat_positions, create_spherical_positions


def DummyMoveNeighbor(particle, fieldset, time, neighbors, mutator):
    """A particle boosts the movement of its nearest neighbor, by adding 0.1 to its lat position."""
    if len(neighbors) == 0:
        pass

    distances = [np.sqrt(n.vert_dist**2 + n.horiz_dist**2) for n in neighbors]
    i_min_dist = np.argmin(distances)

    def f(p):
        p.lat_nextloop += 0.1

    neighbor_id = neighbors[i_min_dist].trajectory
    mutator[neighbor_id].append((f, ()))

    pass


@pytest.fixture
def fieldset_unit_mesh():
    return create_fieldset_unit_mesh(mesh="spherical")


def test_simple_interaction_kernel(fieldset_unit_mesh):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000 * 0.2 * np.pi / 180
    pset = ParticleSet(
        fieldset_unit_mesh,
        pclass=InteractionParticle,
        lon=lons,
        lat=lats,
        interaction_distance=interaction_distance,
    )
    pset.execute(DoNothing, pyfunc_inter=DummyMoveNeighbor, endtime=2.0, dt=1.0)
    assert np.allclose(pset.lat, [0.1, 0.2, 0.1, 0.0], rtol=1e-5)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
@pytest.mark.parametrize("periodic_domain_zonal", [False, True])
def test_zonal_periodic_distance(mesh, periodic_domain_zonal):
    fset = create_fieldset_unit_mesh(mesh=mesh)
    interaction_distance = 0.2 if mesh == "flat" else 6371000 * 0.2 * np.pi / 180
    lons = [0.05, 0.4, 0.95]
    pset = ParticleSet(
        fset,
        pclass=InteractionParticle,
        lon=lons,
        lat=[0.5] * len(lons),
        interaction_distance=interaction_distance,
        periodic_domain_zonal=periodic_domain_zonal,
    )
    pset.execute(DoNothing, pyfunc_inter=DummyMoveNeighbor, endtime=2.0, dt=1.0)
    if periodic_domain_zonal:
        assert np.allclose([pset[0].lat, pset[2].lat], 0.6)
        assert np.allclose(pset[1].lat, 0.5)
    else:
        assert np.allclose([p.lat for p in pset], 0.5)


def test_concatenate_interaction_kernels(fieldset_unit_mesh):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000 * 0.2 * np.pi / 180

    pset = ParticleSet(
        fieldset_unit_mesh,
        pclass=InteractionParticle,
        lon=lons,
        lat=lats,
        interaction_distance=interaction_distance,
    )
    pset.execute(
        DoNothing,
        pyfunc_inter=pset.InteractionKernel(DummyMoveNeighbor) + pset.InteractionKernel(DummyMoveNeighbor),
        endtime=2.0,
        dt=1.0,
    )
    # The kernel results are only applied after all interactionkernels
    # have been executed, so we expect the result to be double the
    # movement from executing the kernel once.
    assert np.allclose(pset.lat, [0.2, 0.4, 0.2, 0.0], rtol=1e-5)


def test_concatenate_interaction_kernels_as_pyfunc(fieldset_unit_mesh):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000 * 0.2 * np.pi / 180

    pset = ParticleSet(
        fieldset_unit_mesh,
        pclass=InteractionParticle,
        lon=lons,
        lat=lats,
        interaction_distance=interaction_distance,
    )
    pset.execute(
        DoNothing, pyfunc_inter=pset.InteractionKernel(DummyMoveNeighbor) + DummyMoveNeighbor, endtime=2.0, dt=1.0
    )
    # The kernel results are only applied after all interactionkernels
    # have been executed, so we expect the result to be double the
    # movement from executing the kernel once.
    assert np.allclose(pset.lat, [0.2, 0.4, 0.2, 0.0], rtol=1e-5)


def test_neighbor_merge(fieldset_unit_mesh):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000 * 5.5 * np.pi / 180
    MergeParticle = InteractionParticle.add_variables(
        [Variable("nearest_neighbor", dtype=np.int64, to_write=False), Variable("mass", initial=1, dtype=np.float32)]
    )
    pset = ParticleSet(
        fieldset_unit_mesh, pclass=MergeParticle, lon=lons, lat=lats, interaction_distance=interaction_distance
    )
    pyfunc_inter = pset.InteractionKernel(NearestNeighborWithinRange) + MergeWithNearestNeighbor
    pset.execute(DoNothing, pyfunc_inter=pyfunc_inter, runtime=3.0, dt=1.0)

    # After two steps, the particles should be removed.
    assert len(pset) == 1


def test_asymmetric_attraction(fieldset_unit_mesh):
    lons = [0.0, 0.1, 0.2]
    lats = [0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000 * 5.5 * np.pi / 180
    AttractingParticle = InteractionParticle.add_variable("attractor", dtype=np.bool_, to_write="once")
    pset = ParticleSet(
        fieldset_unit_mesh,
        pclass=AttractingParticle,
        lon=lons,
        lat=lats,
        interaction_distance=interaction_distance,
        attractor=[True, False, False],
    )
    pyfunc_inter = pset.InteractionKernel(AsymmetricAttraction)
    pset.execute(DoNothing, pyfunc_inter=pyfunc_inter, runtime=3.0, dt=1.0)

    assert lons[1] > pset.lon[1]
    assert lons[2] > pset.lon[2]
    assert len(pset) == 3


def ConstantMoveInteraction(particle, fieldset, time, neighbors, mutator):
    def f(p):
        p.lat_nextloop += p.dt

    mutator[particle.trajectory].append((f, ()))


@pytest.mark.parametrize("runtime, dt", [(1, 1e-2), (1, -2.123e-3), (1, -3.12452 - 3)])
def test_pseudo_interaction(runtime, dt):
    # A linear field where advected particles are moving at
    # 1 m/s in the longitudinal direction.
    xdim, ydim = (10, 20)
    Uflow = Field(
        "U",
        np.ones((ydim, xdim), dtype=np.float64),
        lon=np.linspace(0.0, 1e3, xdim, dtype=np.float64),
        lat=np.linspace(0.0, 1e3, ydim, dtype=np.float64),
    )
    Vflow = Field("V", np.zeros((ydim, xdim), dtype=np.float64), grid=Uflow.grid)
    fieldset = FieldSet(Uflow, Vflow)

    # Execute the advection kernel only
    pset = ParticleSet(fieldset, pclass=InteractionParticle, lon=[2], lat=[2])
    pset.execute(AdvectionRK4, runtime=runtime, dt=dt)

    # Execute both the advection and interaction kernel.
    pset2 = ParticleSet(fieldset, pclass=InteractionParticle, lon=[2], lat=[2], interaction_distance=1)
    pyfunc_inter = pset2.InteractionKernel(ConstantMoveInteraction)
    pset2.execute(AdvectionRK4, pyfunc_inter=pyfunc_inter, runtime=runtime, dt=dt)

    # Check to see whether they have moved as predicted.
    assert np.all(pset.lon == pset2.lon)
    assert np.all(pset2.lat == pset2.lon)
    assert np.all(pset2.particledata.data["time"][0] == pset.particledata.data["time"][0])


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
    neigh_by_coor, _ = instance.find_neighbors_by_coor(instance._values[:, particle_idx])
    assert np.allclose(cur_neigh, neigh_by_coor)

    assert isinstance(cur_neigh, np.ndarray)
    assert set(ref_result) == set(cur_neigh)


@pytest.mark.parametrize("test_class", [KDTreeFlatNeighborSearch, HashFlatNeighborSearch, BruteFlatNeighborSearch])
def test_flat_neighbors(test_class):
    np.random.seed(129873)
    ref_class = BruteFlatNeighborSearch
    n_particle = 1000
    positions = np.random.rand(n_particle * 3).reshape(3, n_particle)
    ref_instance = ref_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)
    test_instance = test_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)
    ref_instance.rebuild(positions)
    test_instance.rebuild(positions)

    for particle_idx in np.random.choice(positions.shape[1], 100, replace=False):
        ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
        compare_results_by_idx(test_instance, particle_idx, ref_result)


@pytest.mark.parametrize("test_class", [BruteSphericalNeighborSearch, HashSphericalNeighborSearch])
def test_spherical_neighbors(test_class):
    np.random.seed(9837452)
    ref_class = BruteSphericalNeighborSearch

    positions = create_spherical_positions(10000, max_depth=100000)
    ref_instance = ref_class(inter_dist_vert=100000, inter_dist_horiz=1000000)
    test_instance = test_class(inter_dist_vert=100000, inter_dist_horiz=1000000)
    ref_instance.rebuild(positions)
    test_instance.rebuild(positions)

    for particle_idx in np.random.choice(positions.shape[1], 100, replace=False):
        ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
        compare_results_by_idx(test_instance, particle_idx, ref_result)


@pytest.mark.parametrize("test_class", [KDTreeFlatNeighborSearch, HashFlatNeighborSearch, BruteFlatNeighborSearch])
def test_flat_update(test_class):
    np.random.seed(9182741)
    n_particle = 1000
    n_test_particle = 10
    n_active_mask = 10
    ref_class = BruteFlatNeighborSearch
    ref_instance = ref_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)
    test_instance = test_class(inter_dist_vert=0.3, inter_dist_horiz=0.3)

    for _ in range(1, n_active_mask):
        positions = create_flat_positions(n_particle) + 10 * np.random.rand()
        active_mask = np.random.rand(n_particle) > 0.5
        ref_instance.update_values(positions, active_mask)
        test_instance.update_values(positions, active_mask)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            continue
        test_particles = np.random.choice(active_idx, size=min(n_test_particle, len(active_idx)), replace=False)
        for particle_idx in test_particles:
            ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
            compare_results_by_idx(test_instance, particle_idx, ref_result, active_idx=active_idx)


@pytest.mark.parametrize("test_class", [BruteSphericalNeighborSearch, HashSphericalNeighborSearch])
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
        test_particles = np.random.choice(active_idx, size=min(n_test_particle, len(active_idx)), replace=False)
        for particle_idx in test_particles:
            ref_result, _ = ref_instance.find_neighbors_by_idx(particle_idx)
            compare_results_by_idx(test_instance, particle_idx, ref_result, active_idx=active_idx)
