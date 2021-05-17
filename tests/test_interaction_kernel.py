from os import path
from parcels import (
    FieldSet, ParticleSet, ScipyParticle, JITParticle, StateCode, OperationCode, ErrorCode, KernelError,
    OutOfBoundsError, AdvectionRK4
)
from parcels.particle import ScipyInteractionParticle
import numpy as np
import pytest
import sys

ptype = {'scipy': ScipyInteractionParticle, 'jit': JITParticle}


def DummyMoveNeighbor(particle, fieldset, time, neighbors, mutator):
    """A particle boosts the movement of its nearest neighbor, by adding
    0.1 to its lat position.
    """
    if len(neighbors) == 0:
        return StateCode.Success

    distances = [np.sqrt(n.surf_dist**2 + n.depth_dist**2) for n in neighbors]
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
