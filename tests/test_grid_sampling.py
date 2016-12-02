from parcels import Grid, ScipyParticle, JITParticle, Geographic, AdvectionRK4, Variable
import numpy as np
import pytest
from math import cos, pi
from datetime import timedelta as delta


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def pclass(mode):
    class SampleParticle(ptype[mode]):
        u = Variable('u', dtype=np.float32)
        v = Variable('v', dtype=np.float32)
        p = Variable('p', dtype=np.float32)
    return SampleParticle


@pytest.fixture
def k_sample_uv():
    def SampleUV(particle, grid, time, dt):
        particle.u = grid.U[time, particle.lon, particle.lat]
        particle.v = grid.V[time, particle.lon, particle.lat]
    return SampleUV


@pytest.fixture
def k_sample_p():
    def SampleP(particle, grid, time, dt):
        particle.p = grid.P[time, particle.lon, particle.lat]
    return SampleP


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


def test_grid_sample(grid, xdim=120, ydim=80):
    """ Sample the grid using indexing notation. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V[0, x, 70.] for x in lon])
    u_s = np.array([grid.U[0, -45., y] for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-7)
    assert np.allclose(u_s, lat, rtol=1e-7)


def test_grid_sample_eval(grid, xdim=60, ydim=60):
    """ Sample the grid using the explicit eval function. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V.eval(0, x, 70.) for x in lon])
    u_s = np.array([grid.U.eval(0, -45., y) for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-7)
    assert np.allclose(u_s, lat, rtol=1e-7)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_sample_particle(grid, mode, k_sample_uv, npart=120):
    """ Sample the grid using an array of particles.

    Note that the low tolerances (1.e-6) are due to the first-order
    interpolation in JIT mode and give an indication of the
    corresponding sampling error.
    """
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lon=lon,
                            lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lat=lat,
                            lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_sample_geographic(grid_geometric, mode, k_sample_uv, npart=120):
    """ Sample a grid with conversion to geographic units (degrees). """
    grid = grid_geometric
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lon=lon,
                            lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lat=lat,
                            lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_sample_geographic_polar(grid_geometric_polar, mode, k_sample_uv, npart=120):
    """ Sample a grid with conversion to geographic units and a pole correction. """
    grid = grid_geometric_polar
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lon=lon,
                            lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = grid.ParticleSet(npart, pclass=pclass(mode), lat=lat,
                            lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
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
    pset.execute(pset.Kernel(AdvectionRK4), endtime=endtime, dt=delta(hours=1))

    assert(pset[0].lat - (latstart[0] + endtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[0].lon - lonstart[0] < 1e-4)
    assert(pset[1].lat - (latstart[1] + endtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[1].lon - lonstart[1] < 1e-4)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_zonalflow_sperical(mode, k_sample_p, xdim=100, ydim=200):
    """ Create uniform EASTWARD flow on sperical earth and advect particles

    As flow is so simple, it can be directly compared to analytical solution
    Note that in this case the cosine conversion is needed
    """
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
    pset = grid.ParticleSet(2, pclass=pclass(mode), lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4) + k_sample_p,
                 endtime=endtime, dt=delta(hours=1))

    assert(pset[0].lat - latstart[0] < 1e-4)
    assert(pset[0].lon - (lonstart[0] + endtime.total_seconds() * maxvel / 1852 / 60
                          / cos(latstart[0] * pi / 180)) < 1e-4)
    assert(abs(pset[0].p - p_fld) < 1e-4)
    assert(pset[1].lat - latstart[1] < 1e-4)
    assert(pset[1].lon - (lonstart[1] + endtime.total_seconds() * maxvel / 1852 / 60
                          / cos(latstart[1] * pi / 180)) < 1e-4)
    assert(abs(pset[1].p - p_fld) < 1e-4)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_random_field(mode, k_sample_p, xdim=20, ydim=20, npart=100):
    """Sampling test that test for overshoots by sampling a field of
    random numbers between 0 and 1.
    """
    np.random.seed(123456)
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U = np.zeros((xdim, ydim), dtype=np.float32)
    V = np.zeros((xdim, ydim), dtype=np.float32)
    P = np.random.uniform(0, 1., size=(xdim, ydim))
    S = np.ones((xdim, ydim), dtype=np.float32)
    grid = Grid.from_data(U, lon, lat, V, lon, lat, mesh='flat',
                          field_data={'P': np.asarray(P, dtype=np.float32),
                                      'start': S})
    pset = grid.ParticleSet(size=npart, pclass=pclass(mode),
                            start_field=grid.start)
    pset.execute(k_sample_p, endtime=1., dt=1.0)
    sampled = np.array([p.p for p in pset])
    assert((sampled >= 0.).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_sampling_out_of_bounds_time(mode, k_sample_p, xdim=10, ydim=10, tdim=10):
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    time = np.linspace(0., 1., tdim, dtype=np.float64)
    U = np.zeros((xdim, ydim, tdim), dtype=np.float32)
    V = np.zeros((xdim, ydim, tdim), dtype=np.float32)
    P = np.ones((xdim, ydim, 1), dtype=np.float32) * time
    grid = Grid.from_data(U, lon, lat, V, lon, lat, time=time, mesh='flat',
                          field_data={'P': np.asarray(P, dtype=np.float32)})
    pset = grid.ParticleSet(size=1, pclass=pclass(mode),
                            start=(0.5, 0.5), finish=(0.5, 0.5))
    pset.execute(k_sample_p, starttime=-1.0, endtime=-0.9, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 0.0, rtol=1e-5)
    pset.execute(k_sample_p, starttime=0.0, endtime=0.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 0.0, rtol=1e-5)
    pset.execute(k_sample_p, starttime=0.5, endtime=0.6, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 0.5, rtol=1e-5)
    pset.execute(k_sample_p, starttime=1.0, endtime=1.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 1.0, rtol=1e-5)
    pset.execute(k_sample_p, starttime=2.0, endtime=2.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 1.0, rtol=1e-5)
