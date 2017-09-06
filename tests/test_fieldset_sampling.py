from parcels import (FieldSet, Field, ParticleSet, ScipyParticle, JITParticle,
                     Geographic, AdvectionRK4, Variable)
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
    def SampleUV(particle, fieldset, time, dt):
        particle.u = fieldset.U[time, particle.lon, particle.lat, particle.depth]
        particle.v = fieldset.V[time, particle.lon, particle.lat, particle.depth]
    return SampleUV


@pytest.fixture
def k_sample_p():
    def SampleP(particle, fieldset, time, dt):
        particle.p = fieldset.P[time, particle.lon, particle.lat, particle.depth]
    return SampleP


@pytest.fixture
def fieldset(xdim=200, ydim=100):
    """ Standard fieldset spanning the earth's coordinates with U and V
        equivalent to longitude and latitude in deg.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}

    return FieldSet.from_data(data, dimensions, mesh='flat')


@pytest.fixture
def fieldset_geometric(xdim=200, ydim=100):
    """ Standard earth fieldset with U and V equivalent to lon/lat in m. """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    U *= 1000. * 1.852 * 60.
    V *= 1000. * 1.852 * 60.
    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}
    fieldset = FieldSet.from_data(data, dimensions)
    fieldset.U.units = Geographic()
    fieldset.V.units = Geographic()
    return fieldset


@pytest.fixture
def fieldset_geometric_polar(xdim=200, ydim=100):
    """ Standard earth fieldset with U and V equivalent to lon/lat in m
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
    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions, mesh='spherical')


def test_fieldset_sample(fieldset, xdim=120, ydim=80):
    """ Sample the fieldset using indexing notation. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([fieldset.V[0, x, 70., 0.] for x in lon])
    u_s = np.array([fieldset.U[0, -45., y, 0.] for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-7)
    assert np.allclose(u_s, lat, rtol=1e-7)


def test_fieldset_sample_eval(fieldset, xdim=60, ydim=60):
    """ Sample the fieldset using the explicit eval function. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([fieldset.V.eval(0, x, 70., 0.) for x in lon])
    u_s = np.array([fieldset.U.eval(0, -45., y, 0.) for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-7)
    assert np.allclose(u_s, lat, rtol=1e-7)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_init_from_field(mode, npart=9):
    dims = (2, 2)
    dimensions = {'lon': np.linspace(0., 1., dims[0], dtype=np.float32),
                  'lat': np.linspace(0., 1., dims[1], dtype=np.float32)}
    data = {'U': np.zeros(dims, dtype=np.float32),
            'V': np.zeros(dims, dtype=np.float32),
            'P': np.zeros(dims, dtype=np.float32)}
    data['P'][0, 0] = 1.
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    xv, yv = np.meshgrid(np.linspace(0, 1, np.sqrt(npart)), np.linspace(0, 1, np.sqrt(npart)))

    class VarParticle(pclass(mode)):
        a = Variable('a', dtype=np.float32, initial=fieldset.P)

    pset = ParticleSet(fieldset, pclass=VarParticle,
                       lon=xv.flatten(), lat=yv.flatten())
    assert np.all([abs(p.a - fieldset.P[p.time, p.lat, p.lon, p.depth]) < 1e-6 for p in pset])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_nearest_neighbour_interpolation2D(mode, k_sample_p, npart=81):
    dims = (2, 2)
    dimensions = {'lon': np.linspace(0., 1., dims[0], dtype=np.float32),
                  'lat': np.linspace(0., 1., dims[1], dtype=np.float32)}
    data = {'U': np.zeros(dims, dtype=np.float32),
            'V': np.zeros(dims, dtype=np.float32),
            'P': np.zeros(dims, dtype=np.float32)}
    data['P'][0, 1] = 1.
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    fieldset.P.interp_method = 'nearest'
    xv, yv = np.meshgrid(np.linspace(0., 1.0, np.sqrt(npart)), np.linspace(0., 1.0, np.sqrt(npart)))
    pset = ParticleSet(fieldset, pclass=pclass(mode),
                       lon=xv.flatten(), lat=yv.flatten())
    pset.execute(k_sample_p, endtime=1, dt=1)
    assert np.allclose(np.array([p.p for p in pset if p.lon < 0.5 and p.lat > 0.5]), 1.0, rtol=1e-5)
    assert np.allclose(np.array([p.p for p in pset if p.lon > 0.5 or p.lat < 0.5]), 0.0, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_nearest_neighbour_interpolation3D(mode, k_sample_p, npart=81):
    dims = (2, 2, 2)
    dimensions = {'lon': np.linspace(0., 1., dims[0], dtype=np.float32),
                  'lat': np.linspace(0., 1., dims[1], dtype=np.float32),
                  'depth': np.linspace(0., 1., dims[2], dtype=np.float32)}
    data = {'U': np.zeros(dims, dtype=np.float32),
            'V': np.zeros(dims, dtype=np.float32),
            'P': np.zeros(dims, dtype=np.float32)}
    data['P'][0, 1, 1] = 1.
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    fieldset.P.interp_method = 'nearest'
    xv, yv = np.meshgrid(np.linspace(0, 1.0, np.sqrt(npart)), np.linspace(0, 1.0, np.sqrt(npart)))
    # combine a pset at 0m with pset at 1m, as meshgrid does not do 3D
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.zeros(npart))
    pset2 = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.ones(npart))
    pset.add(pset2)

    pset.execute(k_sample_p, endtime=1, dt=1)
    assert np.allclose(np.array([p.p for p in pset if p.lon < 0.5 and p.lat > 0.5 and p.depth > 0.5]), 1.0, rtol=1e-5)
    assert np.allclose(np.array([p.p for p in pset if p.lon > 0.5 or p.lat < 0.5 and p.depth < 0.5]), 0.0, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_sample_particle(fieldset, mode, k_sample_uv, npart=120):
    """ Sample the fieldset using an array of particles.

    Note that the low tolerances (1.e-6) are due to the first-order
    interpolation in JIT mode and give an indication of the
    corresponding sampling error.
    """
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_sample_geographic(fieldset_geometric, mode, k_sample_uv, npart=120):
    """ Sample a fieldset with conversion to geographic units (degrees). """
    fieldset = fieldset_geometric
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_sample_geographic_polar(fieldset_geometric_polar, mode, k_sample_uv, npart=120):
    """ Sample a fieldset with conversion to geographic units and a pole correction. """
    fieldset = fieldset_geometric_polar
    lon = np.linspace(-170, 170, npart, dtype=np.float32)
    lat = np.linspace(-80, 80, npart, dtype=np.float32)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart, dtype=np.float32) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart, dtype=np.float32) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    # Note: 1.e-2 is a very low rtol, so there seems to be a rather
    # large sampling error for the JIT correction.
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-2)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_meridionalflow_spherical(mode, xdim=100, ydim=200):
    """ Create uniform NORTHWARD flow on spherical earth and advect particles

    As flow is so simple, it can be directly compared to analytical solution
    """

    maxvel = 1.
    dimensions = {'lon': np.linspace(-180, 180, xdim, dtype=np.float32),
                  'lat': np.linspace(-90, 90, ydim, dtype=np.float32)}
    data = {'U': np.zeros([xdim, ydim]),
            'V': maxvel * np.ones([xdim, ydim])}

    fieldset = FieldSet.from_data(data, dimensions, mesh='spherical')

    lonstart = [0, 45]
    latstart = [0, 45]
    endtime = delta(hours=24)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4), endtime=endtime, dt=delta(hours=1))

    assert(pset[0].lat - (latstart[0] + endtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[0].lon - lonstart[0] < 1e-4)
    assert(pset[1].lat - (latstart[1] + endtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[1].lon - lonstart[1] < 1e-4)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_zonalflow_spherical(mode, k_sample_p, xdim=100, ydim=200):
    """ Create uniform EASTWARD flow on spherical earth and advect particles

    As flow is so simple, it can be directly compared to analytical solution
    Note that in this case the cosine conversion is needed
    """
    maxvel = 1.
    p_fld = 10
    dimensions = {'lon': np.linspace(-180, 180, xdim, dtype=np.float32),
                  'lat': np.linspace(-90, 90, ydim, dtype=np.float32)}
    data = {'U': maxvel * np.ones([xdim, ydim]),
            'V': np.zeros([xdim, ydim]),
            'P': p_fld * np.ones([xdim, ydim])}

    fieldset = FieldSet.from_data(data, dimensions, mesh='spherical')

    lonstart = [0, 45]
    latstart = [0, 45]
    endtime = delta(hours=24)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lonstart, lat=latstart)
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
    """Sampling test that tests for overshoots by sampling a field of
    random numbers between 0 and 1.
    """
    np.random.seed(123456)
    dimensions = {'lon': np.linspace(0., 1., xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1., ydim, dtype=np.float32)}
    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32),
            'P': np.random.uniform(0, 1., size=(xdim, ydim)),
            'start': np.ones((xdim, ydim), dtype=np.float32)}

    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    pset = ParticleSet.from_field(fieldset, size=npart, pclass=pclass(mode),
                                  start_field=fieldset.start)
    pset.execute(k_sample_p, endtime=1., dt=1.0)
    sampled = np.array([p.p for p in pset])
    assert((sampled >= 0.).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('allow_time_extrapolation', [True, False])
def test_sampling_out_of_bounds_time(mode, allow_time_extrapolation, k_sample_p,
                                     xdim=10, ydim=10, tdim=10):
    dimensions = {'lon': np.linspace(0., 1., xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1., ydim, dtype=np.float32),
                  'time': np.linspace(0., 1., tdim, dtype=np.float64)}
    data = {'U': np.zeros((xdim, ydim, tdim), dtype=np.float32),
            'V': np.zeros((xdim, ydim, tdim), dtype=np.float32),
            'P': np.ones((xdim, ydim, 1), dtype=np.float32) * dimensions['time']}

    fieldset = FieldSet.from_data(data, dimensions, mesh='flat',
                                  allow_time_extrapolation=allow_time_extrapolation)
    pset = ParticleSet.from_line(fieldset, size=1, pclass=pclass(mode),
                                 start=(0.5, 0.5), finish=(0.5, 0.5))
    if allow_time_extrapolation:
        pset.execute(k_sample_p, starttime=-1.0, endtime=-0.9, dt=0.1)
        assert np.allclose(np.array([p.p for p in pset]), 0.0, rtol=1e-5)
    else:
        with pytest.raises(RuntimeError):
            pset.execute(k_sample_p, starttime=-1.0, endtime=-0.9, dt=0.1)
    pset.execute(k_sample_p, starttime=0.0, endtime=0.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 0.0, rtol=1e-5)
    pset.execute(k_sample_p, starttime=0.5, endtime=0.6, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 0.5, rtol=1e-5)
    pset.execute(k_sample_p, starttime=1.0, endtime=1.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 1.0, rtol=1e-5)
    if allow_time_extrapolation:
        pset.execute(k_sample_p, starttime=2.0, endtime=2.1, dt=0.1)
        assert np.allclose(np.array([p.p for p in pset]), 1.0, rtol=1e-5)
    else:
        with pytest.raises(RuntimeError):
            pset.execute(k_sample_p, starttime=2.0, endtime=2.1, dt=0.1)


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
def test_sampling_multiple_grid_sizes(mode):
    """Sampling test that tests for FieldSet with different grid sizes

    While this currently works fine in Scipy mode, it fails in JIT mode with
    an out_of_bounds_error because there is only one (xi, yi, zi) for each particle
    A solution would be to define xi, yi, zi for each field separately
    """
    xdim = 10
    ydim = 20
    gf = 10  # factor by which the resolution of U is higher than of V
    U = Field('U', np.zeros((xdim*gf, ydim*gf), dtype=np.float32),
              np.linspace(0., 1., xdim*gf, dtype=np.float32),
              np.linspace(0., 1., ydim*gf, dtype=np.float32))
    V = Field('V', np.zeros((xdim, ydim), dtype=np.float32),
              np.linspace(0., 1., xdim, dtype=np.float32),
              np.linspace(0., 1., ydim, dtype=np.float32))
    fieldset = FieldSet(U, V)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.8], lat=[0.9])

    pset.execute(AdvectionRK4, runtime=10, dt=1)
    assert np.isclose(pset[0].lon, 0.8)
