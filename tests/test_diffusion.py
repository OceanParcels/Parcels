from parcels import (FieldSet, Field, RectilinearZGrid, JITParticle,
                     DiffusionUniformKh, AdvectionDiffusionM1, AdvectionDiffusionEM,
                     ScipyParticle, Variable)
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
from parcels import ParcelsRandom
from datetime import timedelta as delta
import numpy as np
import pytest
from scipy import stats

pset_modes = ['soa', 'aos']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


def zeros_fieldset(mesh='spherical', xdim=200, ydim=100, mesh_conversion=1):
    """Generates a zero velocity field"""
    lon = np.linspace(-1e5*mesh_conversion, 1e5*mesh_conversion, xdim, dtype=np.float32)
    lat = np.linspace(-1e5*mesh_conversion, 1e5*mesh_conversion, ydim, dtype=np.float32)

    dimensions = {'lon': lon, 'lat': lat}
    data = {'U': np.zeros((ydim, xdim), dtype=np.float32),
            'V': np.zeros((ydim, xdim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


@pytest.mark.parametrize('mesh', ['spherical', 'flat'])
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('pset_mode', pset_modes)
def test_fieldKh_Brownian(mesh, mode, pset_mode, xdim=200, ydim=100, kh_zonal=100, kh_meridional=50):
    mesh_conversion = 1/1852./60 if mesh == 'spherical' else 1
    fieldset = zeros_fieldset(mesh=mesh, xdim=xdim, ydim=ydim, mesh_conversion=mesh_conversion)

    fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)
    fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)

    npart = 1000
    runtime = delta(days=1)

    ParcelsRandom.seed(1234)
    pset = pset_type[pset_mode]['pset'](fieldset=fieldset, pclass=ptype[mode], lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(DiffusionUniformKh), runtime=runtime, dt=delta(hours=1))

    expected_std_lon = np.sqrt(2*kh_zonal*mesh_conversion**2*runtime.total_seconds())
    expected_std_lat = np.sqrt(2*kh_meridional*mesh_conversion**2*runtime.total_seconds())

    lats = pset.lat
    lons = pset.lon

    tol = 200*mesh_conversion  # effectively 200 m errors
    assert np.allclose(np.std(lats), expected_std_lat, atol=tol)
    assert np.allclose(np.std(lons), expected_std_lon, atol=tol)
    assert np.allclose(np.mean(lons), 0, atol=tol)
    assert np.allclose(np.mean(lats), 0, atol=tol)


@pytest.mark.parametrize('mesh', ['spherical', 'flat'])
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('kernel', [AdvectionDiffusionM1,
                                    AdvectionDiffusionEM])
def test_fieldKh_SpatiallyVaryingDiffusion(mesh, mode, pset_mode, kernel, xdim=200, ydim=100):
    """Test advection-diffusion kernels on a non-uniform diffusivity field
    with a linear gradient in one direction"""
    mesh_conversion = 1/1852./60 if mesh == 'spherical' else 1
    fieldset = zeros_fieldset(mesh=mesh, xdim=xdim, ydim=ydim, mesh_conversion=mesh_conversion)

    Kh = np.zeros((ydim, xdim), dtype=np.float32)
    for x in range(xdim):
        Kh[:, x] = np.tanh(fieldset.U.lon[x]/fieldset.U.lon[-1]*10.)*xdim/2.+xdim/2. + 100.

    grid = RectilinearZGrid(lon=fieldset.U.lon, lat=fieldset.U.lat, mesh=mesh)
    fieldset.add_field(Field('Kh_zonal', Kh, grid=grid))
    fieldset.add_field(Field('Kh_meridional', Kh, grid=grid))
    fieldset.add_constant('dres', fieldset.U.lon[1]-fieldset.U.lon[0])

    npart = 100
    runtime = delta(days=1)

    ParcelsRandom.seed(1636)
    pset = pset_type[pset_mode]['pset'](fieldset=fieldset, pclass=ptype[mode], lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(kernel), runtime=runtime, dt=delta(hours=1))

    lats = pset.lat
    lons = pset.lon
    tol = 2000*mesh_conversion  # effectively 2000 m errors (because of low numbers of particles)
    assert np.allclose(np.mean(lons), 0, atol=tol)
    assert np.allclose(np.mean(lats), 0, atol=tol)
    assert(stats.skew(lons) > stats.skew(lats))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('lambd', [1, 5])
def test_randomexponential(mode, pset_mode, lambd, npart=1000):
    fieldset = zeros_fieldset()

    # Rate parameter for random.expovariate
    fieldset.lambd = lambd

    # Set random seed
    ParcelsRandom.seed(1234)

    pset = pset_type[pset_mode]['pset'](fieldset=fieldset, pclass=ptype[mode],
                                        lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart))

    def vertical_randomexponential(particle, fieldset, time):
        # Kernel for random exponential variable in depth direction
        particle.depth = ParcelsRandom.expovariate(fieldset.lambd)

    pset.execute(vertical_randomexponential, runtime=1, dt=1)

    depth = pset.depth
    expected_mean = 1./fieldset.lambd
    assert np.allclose(np.mean(depth), expected_mean, rtol=.1)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mu', [0.8*np.pi, np.pi])
@pytest.mark.parametrize('kappa', [2, 4])
def test_randomvonmises(mode, pset_mode, mu, kappa, npart=10000):
    fieldset = zeros_fieldset()

    # Parameters for random.vonmisesvariate
    fieldset.mu = mu
    fieldset.kappa = kappa

    # Set random seed
    ParcelsRandom.seed(1234)

    class AngleParticle(ptype[mode]):
        angle = Variable('angle')
    pset = pset_type[pset_mode]['pset'](fieldset=fieldset, pclass=AngleParticle, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart))

    def vonmises(particle, fieldset, time):
        particle.angle = ParcelsRandom.vonmisesvariate(fieldset.mu, fieldset.kappa)

    pset.execute(vonmises, runtime=1, dt=1)

    angles = np.array([p.angle for p in pset])

    assert np.allclose(np.mean(angles), mu, atol=.1)
    scipy_mises = stats.vonmises.rvs(kappa, loc=mu, size=10000)
    assert np.allclose(np.mean(angles), np.mean(scipy_mises), atol=.1)
    assert np.allclose(np.std(angles), np.std(scipy_mises), atol=.1)
