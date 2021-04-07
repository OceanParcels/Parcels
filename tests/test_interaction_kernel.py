from os import path
from parcels import (
    FieldSet, ParticleSet, ScipyParticle, JITParticle, StateCode, OperationCode, ErrorCode, KernelError,
    OutOfBoundsError, AdvectionRK4, DummyMoveNeighbour
)
import numpy as np
import pytest
import sys


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


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
    pset.execute(DoNothing, pyfunc_inter=DummyMoveNeighbour, endtime=1., dt=1.)
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
                 pyfunc_inter=pset.InteractionKernel(DummyMoveNeighbour)
                 + pset.InteractionKernel(DummyMoveNeighbour), endtime=1.,
                 dt=1.)
    # The kernel results are only applied after all interactionkernels
    # have been executed, so we expect the result to be double the
    # movement from executing the kernel once.
    assert np.allclose(pset.lat, [0.2, 0.4, 0.1, 0.0], rtol=1e-5)
