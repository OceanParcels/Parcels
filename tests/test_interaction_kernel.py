import numpy as np
import pytest

from parcels import (
    FieldSet, ParticleSet, JITParticle, StateCode, Field
)
from parcels.particle import ScipyInteractionParticle, Variable, ScipyParticle
from parcels.application_kernels.interaction import NearestNeighborWithinRange
from parcels.application_kernels.interaction import AsymmetricAttraction
from parcels.application_kernels.interaction import MergeWithNearestNeighbor
from parcels.application_kernels.advection import AdvectionRK4

ptype = {'scipy': ScipyInteractionParticle, 'jit': JITParticle}


def DummyMoveNeighbor(particle, fieldset, time, neighbors, mutator):
    """A particle boosts the movement of its nearest neighbor, by adding
    0.1 to its lat position.
    """
    if len(neighbors) == 0:
        return StateCode.Success

    distances = [np.sqrt(n.vert_dist**2 + n.horiz_dist**2) for n in neighbors]
    i_min_dist = np.argmin(distances)

    def f(p):
        p.lat += 0.1

    neighbor_id = neighbors[i_min_dist].id
    mutator[neighbor_id].append((f, ()))

    return StateCode.Success


def DoNothing(particle, fieldset, time):
    return StateCode.Success


def fieldset(xdim=20, ydim=20, mesh='spherical'):
    """ Standard unit mesh fieldset """
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


class MergeParticle(ScipyInteractionParticle):
    nearest_neighbor = Variable('nearest_neighbor', dtype=np.int64, to_write=False)
    mass = Variable('mass', initial=1, dtype=np.float32)


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=20, ydim=20):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('mode', ['scipy'])
def test_simple_interaction_kernel(fieldset, mode):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000*0.2*np.pi/180
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lons, lat=lats,
                       interaction_distance=interaction_distance)
    pset.execute(DoNothing, pyfunc_inter=DummyMoveNeighbor, endtime=1., dt=1.)
    assert np.allclose(pset.lat, [0.1, 0.2, 0.1, 0.0], rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy'])
@pytest.mark.parametrize('mesh', ['spherical', 'flat'])
@pytest.mark.parametrize('periodic_domain_zonal', [False, True])
def test_zonal_periodic_distance(mode, mesh, periodic_domain_zonal):
    fset = fieldset(mesh=mesh)
    interaction_distance = 0.2 if mesh == 'flat' else 6371000*0.2*np.pi/180
    lons = [0.05, 0.4, 0.95]
    pset = ParticleSet(fset, pclass=ptype[mode], lon=lons, lat=[0.5]*len(lons),
                       interaction_distance=interaction_distance, periodic_domain_zonal=periodic_domain_zonal)
    pset.execute(DoNothing, pyfunc_inter=DummyMoveNeighbor, endtime=1., dt=1.)
    if periodic_domain_zonal:
        assert np.allclose([pset[0].lat, pset[2].lat], 0.6)
        assert np.allclose(pset[1].lat, 0.5)
    else:
        assert np.allclose([p.lat for p in pset], 0.5)


@pytest.mark.parametrize('mode', ['scipy'])
def test_concatenate_interaction_kernels(fieldset, mode):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000*0.2*np.pi/180

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lons, lat=lats,
                       interaction_distance=interaction_distance)
    pset.execute(DoNothing,
                 pyfunc_inter=pset.InteractionKernel(DummyMoveNeighbor)
                 + pset.InteractionKernel(DummyMoveNeighbor), endtime=1.,
                 dt=1.)
    # The kernel results are only applied after all interactionkernels
    # have been executed, so we expect the result to be double the
    # movement from executing the kernel once.
    assert np.allclose(pset.lat, [0.2, 0.4, 0.1, 0.0], rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy'])
def test_concatenate_interaction_kernels_as_pyfunc(fieldset, mode):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000*0.2*np.pi/180

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lons, lat=lats,
                       interaction_distance=interaction_distance)
    pset.execute(DoNothing,
                 pyfunc_inter=pset.InteractionKernel(DummyMoveNeighbor)
                 + DummyMoveNeighbor, endtime=1., dt=1.)
    # The kernel results are only applied after all interactionkernels
    # have been executed, so we expect the result to be double the
    # movement from executing the kernel once.
    assert np.allclose(pset.lat, [0.2, 0.4, 0.1, 0.0], rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy'])
def test_neighbor_merge(fieldset, mode):
    lons = [0.0, 0.1, 0.25, 0.44]
    lats = [0.0, 0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000*5.5*np.pi/180
    pset = ParticleSet(fieldset, pclass=MergeParticle, lon=lons, lat=lats,
                       interaction_distance=interaction_distance)
    pyfunc_inter = (pset.InteractionKernel(NearestNeighborWithinRange)
                    + MergeWithNearestNeighbor)
    pset.execute(DoNothing,
                 pyfunc_inter=pyfunc_inter, runtime=3., dt=1.)

    # After two steps, the particles should be removed.
    assert len(pset) == 1


class AttractingParticle(ScipyInteractionParticle):
    attractor = Variable('attractor', dtype=np.bool_, to_write='once')


@pytest.mark.parametrize('mode', ['scipy'])
def test_asymmetric_attraction(fieldset, mode):
    lons = [0.0, 0.1, 0.2]
    lats = [0.0, 0.0, 0.0]
    # Distance in meters R_earth*0.2 degrees
    interaction_distance = 6371000*5.5*np.pi/180
    pset = ParticleSet(fieldset, pclass=AttractingParticle, lon=lons, lat=lats,
                       interaction_distance=interaction_distance,
                       attractor=[True, False, False])
    pyfunc_inter = pset.InteractionKernel(AsymmetricAttraction)
    pset.execute(DoNothing,
                 pyfunc_inter=pyfunc_inter, runtime=3., dt=1.)

    assert lons[1] > pset.lon[1]
    assert lons[2] > pset.lon[2]
    assert len(pset) == 3


def ConstantMoveInteraction(particle, fieldset, time, neighbors, mutator):
    def f(p):
        p.lat += p.dt
    mutator[particle.id].append((f, ()))


@pytest.mark.parametrize('runtime,dt',
                         [(1, 1e-4),
                          (1, -2.1234e-3),
                          (1, -3.12452-3)])
def test_pseudo_interaction(runtime, dt):
    # A linear field where advected particles are moving at
    # 1 m/s in the longitudinal direction.
    xdim, ydim = (10, 20)
    Uflow = Field('U', np.ones((ydim, xdim), dtype=np.float64),
                  lon=np.linspace(0., 1e3, xdim, dtype=np.float64),
                  lat=np.linspace(0., 1e3, ydim, dtype=np.float64))
    Vflow = Field('V', np.zeros((ydim, xdim), dtype=np.float64), grid=Uflow.grid)
    fieldset = FieldSet(Uflow, Vflow)

    # Execute the advection kernel only
    pset = ParticleSet(fieldset, pclass=ScipyParticle, lon=[2], lat=[2])
    pset.execute(AdvectionRK4, runtime=runtime, dt=dt)

    # Execute both the advection and interaction kernel.
    pset2 = ParticleSet(fieldset, pclass=ScipyInteractionParticle, lon=[2], lat=[2], interaction_distance=1)
    pyfunc_inter = pset2.InteractionKernel(ConstantMoveInteraction)
    pset2.execute(AdvectionRK4, pyfunc_inter=pyfunc_inter, runtime=runtime, dt=dt)

    # Check to see whether they have moved as predicted.
    assert np.all(pset.lon == pset2.lon)
    assert np.all(pset2.lat == pset2.lon)
    assert np.all(pset2._collection.data["time"][0] == pset._collection.data["time"][0])
