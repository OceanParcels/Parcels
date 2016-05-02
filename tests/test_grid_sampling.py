from parcels import Grid, Particle, JITParticle
import numpy as np
import pytest


ptype = {'scipy': Particle, 'jit': JITParticle}


@pytest.fixture
def grid(xdim=200, ydim=100):
    """ Standard grid spanning the earth's coordinates with U and V
        equivalent to longitude and latitude.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)
    U, V = np.meshgrid(lat, lon)
    return Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat,
                          depth, time)


def test_grid_sample(grid, xdim=120, ydim=80):
    """ Sample the grid using indexing notation. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V[0, x, 70., 0.] for x in lon])
    u_s = np.array([grid.U[0, -45., y, 0.] for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-12)
    assert np.allclose(u_s, lat, rtol=1e-12)


def test_grid_sample_eval(grid, xdim=60, ydim=60):
    """ Sample the grid using the explicit eval function. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V.eval(0, x, 70., 0.) for x in lon])
    u_s = np.array([grid.U.eval(0, -45., y, 0.) for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-12)
    assert np.allclose(u_s, lat, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_sample_particle(grid, mode, npart=120):
    """ Sample the grid using an array of particles.

    Note that the low tolerances (1.e-6) are due to the first-order
    interpolation in JIT mode and give an indication of the
    corresponding sampling error.
    """
    class SampleParticle(ptype[mode]):
        user_vars = {'u': np.float32, 'v': np.float32}

    def Sample(particle, grid, time, dt):
        particle.u = grid.U[0., particle.lon, particle.lat, particle.dep]
        particle.v = grid.V[0., particle.lon, particle.lat, particle.dep]

    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)
    dep = np.linspace(0, 0, npart, dtype=np.float32)

    pset = grid.ParticleSet(npart, pclass=SampleParticle, lon=lon, dep=dep,
                            lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(Sample), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = grid.ParticleSet(npart, pclass=SampleParticle, lat=lat, dep=dep,
                            lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(Sample), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)
