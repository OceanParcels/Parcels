from parcels import Grid, Particle, JITParticle, Geographic, AdvectionRK4_2D
import numpy as np
import pytest
from math import cos, pi
from datetime import timedelta as delta


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
                          np.array(V, dtype=np.float32), lon, lat,
                          mesh='flat')


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
                          np.array(V, dtype=np.float32), lon, lat,
                          mesh='spherical')
    return grid


def pclass(mode):
    class SampleParticle(ptype[mode]):
        user_vars = {'u': np.float32, 'v': np.float32}
    return SampleParticle


@pytest.fixture
def samplefunc():
    def Sample(particle, grid, time, dt):
        particle.u = grid.U[0., particle.lon, particle.lat, particle.dep]
        particle.v = grid.V[0., particle.lon, particle.lat, particle.dep]
    return Sample


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


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_meridionalflow_sperical(mode, xdim=100, ydim=200):
    """ Create uniform NORTHWARD flow on sperical earth and advect particles

    As flow is so simple, it can be directly compared to analytical solution
    """

    maxvel = 1.
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U = np.zeros([xdim, ydim])
    V = maxvel * np.ones([xdim, ydim])

    grid = Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat)

    lonstart = [0, 45]
    latstart = [0, 45]
    endtime = delta(hours=24)
    pset = grid.ParticleSet(2, pclass=pclass(mode), lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4_2D), endtime=endtime, dt=delta(hours=1))

    assert(pset[0].lat - (latstart[0] + endtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[0].lon - lonstart[0] < 1e-4)
    assert(pset[1].lat - (latstart[1] + endtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[1].lon - lonstart[1] < 1e-4)


def UpdateP(particle, grid, time, dt):
    particle.p = grid.P[time, particle.lon, particle.lat, particle.dep]


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_zonalflow_sperical(mode, xdim=100, ydim=200):
    """ Create uniform EASTWARD flow on sperical earth and advect particles

    As flow is so simple, it can be directly compared to analytical solution
    Note that in this case the cosine conversion is needed
    """

    ParticleClass = JITParticle if mode == 'jit' else Particle

    class MyParticle(ParticleClass):
        user_vars = {'p': np.float32}

        def __init__(self, *args, **kwargs):
            super(MyParticle, self).__init__(*args, **kwargs)
            self.p = 1.

        def __repr__(self):
            return "P(%.4f, %.4f)[p=%.5f]" % (self.lon, self.lat, self.p)

    maxvel = 1.
    p_fld = 10
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    V = np.zeros([xdim, ydim])
    U = maxvel * np.ones([xdim, ydim])
    P = p_fld * np.ones([xdim, ydim])

    grid = Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat,
                          field_data={'P': np.array(P, dtype=np.float32)})

    lonstart = [0, 45]
    latstart = [0, 45]
    endtime = delta(hours=24)
    pset = grid.ParticleSet(2, pclass=MyParticle, lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4_2D) + pset.Kernel(UpdateP),
                 endtime=endtime, dt=delta(hours=1))

    assert(pset[0].lat - latstart[0] < 1e-4)
    assert(pset[0].lon - (lonstart[0] + endtime.total_seconds() * maxvel / 1852 / 60
                          / cos(latstart[0] * pi / 180)) < 1e-4)
    assert(abs(pset[0].p - p_fld) < 1e-4)
    assert(pset[1].lat - latstart[1] < 1e-4)
    assert(pset[1].lon - (lonstart[1] + endtime.total_seconds() * maxvel / 1852 / 60
                          / cos(latstart[1] * pi / 180)) < 1e-4)
    assert(abs(pset[1].p - p_fld) < 1e-4)
