from parcels import (
    Grid, ScipyParticle, JITParticle, ErrorCode, KernelError,
    OutOfBoundsError
)
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def DoNothing(particle, grid, time, dt):
    return ErrorCode.Success


@pytest.fixture
def grid(xdim=20, ydim=20):
    """ Standard unit mesh grid """
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    return Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat,
                          mesh='flat')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_endtime(grid, mode, start, end, substeps, dt, npart=10):
    pset = grid.ParticleSet(npart, pclass=ptype[mode],
                            lon=np.linspace(0, 1, npart, dtype=np.float32),
                            lat=np.linspace(1, 0, npart, dtype=np.float32))
    pset.execute(DoNothing, starttime=start, endtime=end, dt=dt)
    assert np.allclose(np.array([p.time for p in pset]), end)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_runtime(grid, mode, start, end, substeps, dt, npart=10):
    pset = grid.ParticleSet(npart, pclass=ptype[mode],
                            lon=np.linspace(0, 1, npart, dtype=np.float32),
                            lat=np.linspace(1, 0, npart, dtype=np.float32))
    t_step = (end - start) / substeps
    for _ in range(substeps):
        pset.execute(DoNothing, starttime=start, runtime=t_step, dt=dt)
        start += t_step
    assert np.allclose(np.array([p.time for p in pset]), end)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_fail_timed(grid, mode, npart=10):
    def TimedFail(particle, grid, time, dt):
        if particle.time >= 10.:
            return ErrorCode.Error
        else:
            return ErrorCode.Success

    pset = grid.ParticleSet(npart, pclass=ptype[mode],
                            lon=np.linspace(0, 1, npart, dtype=np.float32),
                            lat=np.linspace(1, 0, npart, dtype=np.float32))
    error_thrown = False
    try:
        pset.execute(TimedFail, starttime=0., endtime=20., dt=2.)
    except KernelError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert np.allclose(np.array([p.time for p in pset]), 10.)


@pytest.mark.parametrize('mode', ['scipy'])
def test_execution_fail_python_exception(grid, mode, npart=10):
    def PythonFail(particle, grid, time, dt):
        if particle.time >= 10.:
            raise RuntimeError("Enough is enough!")
        else:
            return ErrorCode.Success

    pset = grid.ParticleSet(npart, pclass=ptype[mode],
                            lon=np.linspace(0, 1, npart, dtype=np.float32),
                            lat=np.linspace(1, 0, npart, dtype=np.float32))
    error_thrown = False
    try:
        pset.execute(PythonFail, starttime=0., endtime=20., dt=2.)
    except KernelError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert np.allclose(np.array([p.time for p in pset]), 10.)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_fail_out_of_bounds(grid, mode, npart=10):
    def MoveRight(particle, grid, time, dt):
        grid.U[time, particle.lon + 0.1, particle.lat]
        particle.lon += 0.1

    pset = grid.ParticleSet(npart, pclass=ptype[mode],
                            lon=np.linspace(0, 1, npart, dtype=np.float32),
                            lat=np.linspace(1, 0, npart, dtype=np.float32))
    error_thrown = False
    try:
        pset.execute(MoveRight, starttime=0., endtime=10., dt=1.)
    except OutOfBoundsError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert (np.array([p.lon - 1. for p in pset]) > -1.e12).all()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_recover_out_of_bounds(grid, mode, npart=2):
    def MoveRight(particle, grid, time, dt):
        grid.U[time, particle.lon + 0.1, particle.lat]
        particle.lon += 0.1

    def MoveLeft(particle):
        particle.lon -= 1.

    lon = np.linspace(0.05, 0.95, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = grid.ParticleSet(npart, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute(MoveRight, starttime=0., endtime=10., dt=1.,
                 recovery={ErrorCode.ErrorOutOfBounds: MoveLeft})
    assert len(pset) == npart
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-5)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-5)
