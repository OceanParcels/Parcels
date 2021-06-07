import numpy as np
import pytest

from parcels import (
    FieldSet, ParticleSet, JITParticle, StateCode
)
from parcels.particle import ScipyInteractionParticle, Variable
from parcels.application_kernels.interaction import NearestNeighborWithinRange,\
    AsymmetricAttraction
from parcels.application_kernels.interaction import MergeWithNearestNeighbor

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


def fieldset(xdim=20, ydim=20):
    """ Standard unit mesh fieldset """
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon}
    return FieldSet.from_data(data, dimensions, mesh='spherical')


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
    print(pset.lat)
    assert np.allclose(pset.lat, [0.1, 0.2, 0.1, 0.0], rtol=1e-5)


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
