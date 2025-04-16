import random
from datetime import timedelta

import numpy as np
import pytest
from scipy import stats

from parcels import (
    AdvectionDiffusionEM,
    AdvectionDiffusionM1,
    DiffusionUniformKh,
    Field,
    Particle,
    ParticleSet,
    RectilinearZGrid,
)
from tests.utils import create_fieldset_zeros_conversion


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_fieldKh_Brownian(mesh):
    xdim = 200
    ydim = 100
    kh_zonal = 100
    kh_meridional = 50

    mesh_conversion = 1 / 1852.0 / 60 if mesh == "spherical" else 1
    fieldset = create_fieldset_zeros_conversion(mesh=mesh, xdim=xdim, ydim=ydim, mesh_conversion=mesh_conversion)

    fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)
    fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)

    npart = 1000
    runtime = timedelta(days=1)

    random.seed(1234)
    pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(DiffusionUniformKh), runtime=runtime, dt=timedelta(hours=1))

    expected_std_lon = np.sqrt(2 * kh_zonal * mesh_conversion**2 * runtime.total_seconds())
    expected_std_lat = np.sqrt(2 * kh_meridional * mesh_conversion**2 * runtime.total_seconds())

    lats = pset.lat
    lons = pset.lon

    tol = 500 * mesh_conversion  # effectively 500 m errors
    assert np.allclose(np.std(lats), expected_std_lat, atol=tol)
    assert np.allclose(np.std(lons), expected_std_lon, atol=tol)
    assert np.allclose(np.mean(lons), 0, atol=tol)
    assert np.allclose(np.mean(lats), 0, atol=tol)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("mesh", ["spherical", "flat"])
@pytest.mark.parametrize("kernel", [AdvectionDiffusionM1, AdvectionDiffusionEM])
def test_fieldKh_SpatiallyVaryingDiffusion(mesh, kernel):
    """Test advection-diffusion kernels on a non-uniform diffusivity field with a linear gradient in one direction."""
    xdim = 200
    ydim = 100
    mesh_conversion = 1 / 1852.0 / 60 if mesh == "spherical" else 1
    fieldset = create_fieldset_zeros_conversion(mesh=mesh, xdim=xdim, ydim=ydim, mesh_conversion=mesh_conversion)

    Kh = np.zeros((ydim, xdim), dtype=np.float32)
    for x in range(xdim):
        Kh[:, x] = np.tanh(fieldset.U.lon[x] / fieldset.U.lon[-1] * 10.0) * xdim / 2.0 + xdim / 2.0 + 100.0

    grid = RectilinearZGrid(lon=fieldset.U.lon, lat=fieldset.U.lat, mesh=mesh)
    fieldset.add_field(Field("Kh_zonal", Kh, grid=grid))
    fieldset.add_field(Field("Kh_meridional", Kh, grid=grid))
    fieldset.add_constant("dres", fieldset.U.lon[1] - fieldset.U.lon[0])

    npart = 100
    runtime = timedelta(days=1)

    random.seed(1636)
    pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(kernel), runtime=runtime, dt=timedelta(hours=1))

    lats = pset.lat
    lons = pset.lon
    tol = 2000 * mesh_conversion  # effectively 2000 m errors (because of low numbers of particles)
    assert np.allclose(np.mean(lons), 0, atol=tol)
    assert np.allclose(np.mean(lats), 0, atol=tol)
    assert stats.skew(lons) > stats.skew(lats)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("lambd", [1, 5])
def test_randomexponential(lambd):
    fieldset = create_fieldset_zeros_conversion()
    npart = 1000

    # Rate parameter for random.expovariate
    fieldset.lambd = lambd

    # Set random seed
    random.seed(1234)

    pset = ParticleSet(
        fieldset=fieldset, pclass=Particle, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart)
    )

    def vertical_randomexponential(particle, fieldset, time):  # pragma: no cover
        # Kernel for random exponential variable in depth direction
        particle.depth = random.expovariate(fieldset.lambd)

    pset.execute(vertical_randomexponential, runtime=1, dt=1)

    depth = pset.depth
    expected_mean = 1.0 / fieldset.lambd
    assert np.allclose(np.mean(depth), expected_mean, rtol=0.1)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("mu", [0.8 * np.pi, np.pi])
@pytest.mark.parametrize("kappa", [2, 4])
def test_randomvonmises(mu, kappa):
    npart = 10000
    fieldset = create_fieldset_zeros_conversion()

    # Parameters for random.vonmisesvariate
    fieldset.mu = mu
    fieldset.kappa = kappa

    # Set random seed
    random.seed(1234)

    AngleParticle = Particle.add_variable("angle")
    pset = ParticleSet(
        fieldset=fieldset, pclass=AngleParticle, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart)
    )

    def vonmises(particle, fieldset, time):  # pragma: no cover
        particle.angle = random.vonmisesvariate(fieldset.mu, fieldset.kappa)

    pset.execute(vonmises, runtime=1, dt=1)

    angles = np.array([p.angle for p in pset])

    assert np.allclose(np.mean(angles), mu, atol=0.1)
    vonmises_mean = stats.vonmises.mean(kappa=kappa, loc=mu)
    assert np.allclose(np.mean(angles), vonmises_mean, atol=0.1)
    vonmises_var = stats.vonmises.var(kappa=kappa, loc=mu)
    assert np.allclose(np.var(angles), vonmises_var, atol=0.1)
