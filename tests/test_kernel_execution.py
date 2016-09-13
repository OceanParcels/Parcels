from parcels import Grid, ScipyParticle, JITParticle, KernelOp
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def DoNothing(particle, grid, time, dt):
    return KernelOp.SUCCESS


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
