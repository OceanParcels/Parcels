from parcels import Grid, Particle, JITParticle, Geographic, GeographicPolar
import numpy as np
import pytest
from math import cos, pi


ptype = {'scipy': Particle, 'jit': JITParticle}


@pytest.fixture
def grid(xdim=200, ydim=100):
    """ Standard grid spanning the earth's coordinates with U and V
        equivalent to longitude and latitude in deg.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    return Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat)


@pytest.fixture
def grid_geometric(xdim=200, ydim=100):
    """ Standard earth grid with U and V equivalent to lon/lat in m. """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    U *= 1000. * 1.852 * 60.
    V *= 1000. * 1.852 * 60.
    grid = Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat)
    grid.U.units = Geographic()
    grid.V.units = Geographic()
    return grid


@pytest.fixture
def grid_geometric_polar(xdim=200, ydim=100):
    """ Standard earth grid with U and V equivalent to lon/lat in m
        and the inversion of the pole correction applied to U.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    # Apply inverse of pole correction to U
    for i, y in enumerate(lat):
        U[:, i] *= cos(y * pi / 180)
    U *= 1000. * 1.852 * 60.
    V *= 1000. * 1.852 * 60.
    grid = Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat)
    grid.U.units = GeographicPolar()
    grid.V.units = Geographic()
    return grid


def pclass(mode):
    class SampleParticle(ptype[mode]):
        user_vars = {'u': np.float32, 'v': np.float32}
    return SampleParticle


@pytest.fixture
def samplefunc():
    def Sample(particle, grid, time, dt):
        particle.u = grid.U[0., particle.lon, particle.lat]
        particle.v = grid.V[0., particle.lon, particle.lat]
    return Sample


def test_grid_sample(grid, xdim=120, ydim=80):
    """ Sample the grid using indexing notation. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V[0, x, 70.] for x in lon])
    u_s = np.array([grid.U[0, -45., y] for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-12)
    assert np.allclose(u_s, lat, rtol=1e-12)


def test_grid_sample_eval(grid, xdim=60, ydim=60):
    """ Sample the grid using the explicit eval function. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V.eval(0, x, 70.) for x in lon])
    u_s = np.array([grid.U.eval(0, -45., y) for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-12)
    assert np.allclose(u_s, lat, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_sample_particle(grid, mode, samplefunc, npart=120):
    """ Sample the grid using an array of particles.

    Note that the low tolerances (1.e-6) are due to the first-order
    interpolation in JIT mode and give an indication of the
    corresponding sampling error.
    """
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lon=lon,
                            lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(samplefunc), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lat=lat,
                            lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(samplefunc), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_sample_geographic(grid_geometric, mode, samplefunc, npart=120):
    """ Sample a grid with conversion to geographic units (degrees). """
    grid = grid_geometric
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lon=lon,
                            lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(samplefunc), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lat=lat,
                            lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(samplefunc), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_sample_geographic_polar(grid_geometric_polar, mode, samplefunc, npart=120):
    """ Sample a grid with conversion to geographic units and a pole correction. """
    grid = grid_geometric_polar
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lon=lon,
                            lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(samplefunc), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lat=lat,
                            lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(samplefunc), endtime=1., dt=1.)
    # Note: 1.e-2 is a very low rtol, so there seems to be a rather
    # large sampling error for the JIT correction.
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-2)
