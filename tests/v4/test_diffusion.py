import random

import numpy as np
import pytest
from scipy import stats

from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.application_kernels import AdvectionDiffusionEM, AdvectionDiffusionM1, DiffusionUniformKh
from parcels.application_kernels.interpolation import XBiLinear
from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.particle import Particle, Variable
from parcels.particleset import ParticleSet
from parcels.xgrid import XGrid
from tests.utils import create_fieldset_zeros_conversion


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_fieldKh_Brownian(mesh_type):
    kh_zonal = 100
    kh_meridional = 50
    mesh_conversion = 1 / 1852.0 / 60 if mesh_type == "spherical" else 1

    ds = simple_UV_dataset(dims=(2, 1, 2, 2), mesh_type=mesh_type)
    ds["lon"].data = np.array([-1e6, 1e6])
    ds["lat"].data = np.array([-1e6, 1e6])
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=XBiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=XBiLinear)
    ds["Kh_zonal"] = (["time", "depth", "YG", "XG"], np.full((2, 1, 2, 2), kh_zonal))
    ds["Kh_meridional"] = (["time", "depth", "YG", "XG"], np.full((2, 1, 2, 2), kh_meridional))
    Kh_zonal = Field("Kh_zonal", ds["Kh_zonal"], grid=grid, mesh_type=mesh_type, interp_method=XBiLinear)
    Kh_meridional = Field("Kh_meridional", ds["Kh_meridional"], grid=grid, mesh_type=mesh_type, interp_method=XBiLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV, Kh_zonal, Kh_meridional])

    npart = 100
    runtime = np.timedelta64(2, "h")

    random.seed(1234)
    pset = ParticleSet(fieldset=fieldset, lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(DiffusionUniformKh), runtime=runtime, dt=np.timedelta64(1, "h"))

    expected_std_lon = np.sqrt(2 * kh_zonal * mesh_conversion**2 * (runtime / np.timedelta64(1, "s")))
    expected_std_lat = np.sqrt(2 * kh_meridional * mesh_conversion**2 * (runtime / np.timedelta64(1, "s")))

    tol = 500 * mesh_conversion  # effectively 500 m errors
    assert np.allclose(np.std(pset.lat), expected_std_lat, atol=tol)
    assert np.allclose(np.std(pset.lon), expected_std_lon, atol=tol)
    assert np.allclose(np.mean(pset.lon), 0, atol=tol)
    assert np.allclose(np.mean(pset.lat), 0, atol=tol)


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
@pytest.mark.parametrize("kernel", [AdvectionDiffusionM1, AdvectionDiffusionEM])
def test_fieldKh_SpatiallyVaryingDiffusion(mesh_type, kernel):
    """Test advection-diffusion kernels on a non-uniform diffusivity field with a linear gradient in one direction."""
    ydim, xdim = 100, 200

    mesh_conversion = 1 / 1852.0 / 60 if mesh_type == "spherical" else 1
    ds = simple_UV_dataset(dims=(2, 1, ydim, xdim), mesh_type=mesh_type)
    ds["lon"].data = np.linspace(-1e6, 1e6, xdim)
    ds["lat"].data = np.linspace(-1e6, 1e6, ydim)
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=XBiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=XBiLinear)

    Kh = np.zeros((ydim, xdim), dtype=np.float32)
    for x in range(xdim):
        Kh[:, x] = np.tanh(ds["lon"][x] / ds["lon"][-1] * 10.0) * xdim / 2.0 + xdim / 2.0 + 100.0

    ds["Kh_zonal"] = (["time", "depth", "YG", "XG"], np.full((2, 1, ydim, xdim), Kh))
    ds["Kh_meridional"] = (["time", "depth", "YG", "XG"], np.full((2, 1, ydim, xdim), Kh))
    Kh_zonal = Field("Kh_zonal", ds["Kh_zonal"], grid=grid, mesh_type=mesh_type, interp_method=XBiLinear)
    Kh_meridional = Field("Kh_meridional", ds["Kh_meridional"], grid=grid, mesh_type=mesh_type, interp_method=XBiLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV, Kh_zonal, Kh_meridional])
    fieldset.add_constant("dres", ds["lon"][1] - ds["lon"][0])

    npart = 100

    random.seed(1636)
    pset = ParticleSet(fieldset=fieldset, lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(kernel), runtime=np.timedelta64(4, "h"), dt=np.timedelta64(1, "h"))

    tol = 2000 * mesh_conversion  # effectively 2000 m errors (because of low numbers of particles)
    assert np.allclose(np.mean(pset.lon), 0, atol=tol)
    assert np.allclose(np.mean(pset.lat), 0, atol=tol)
    assert stats.skew(pset.lon) > stats.skew(pset.lat)


@pytest.mark.parametrize("lambd", [1, 5])
def test_randomexponential(lambd):
    fieldset = create_fieldset_zeros_conversion()
    npart = 1000

    # Rate parameter for random.expovariate
    fieldset.lambd = lambd

    # Set random seed
    random.seed(1234)

    pset = ParticleSet(fieldset=fieldset, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart))

    def vertical_randomexponential(particle, fieldset, time):  # pragma: no cover
        # Kernel for random exponential variable in depth direction
        particle.depth = random.expovariate(fieldset.lambd)

    pset.execute(vertical_randomexponential, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))

    expected_mean = 1.0 / fieldset.lambd
    assert np.allclose(np.mean(pset.depth), expected_mean, rtol=0.1)


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

    AngleParticle = Particle.add_variable(Variable("angle"))
    pset = ParticleSet(
        fieldset=fieldset, pclass=AngleParticle, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart)
    )

    def vonmises(particle, fieldset, time):  # pragma: no cover
        particle.angle = random.vonmisesvariate(fieldset.mu, fieldset.kappa)

    pset.execute(vonmises, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))

    angles = np.array([p.angle for p in pset])

    assert np.allclose(np.mean(angles), mu, atol=0.1)
    vonmises_mean = stats.vonmises.mean(kappa=kappa, loc=mu)
    assert np.allclose(np.mean(angles), vonmises_mean, atol=0.1)
    vonmises_var = stats.vonmises.var(kappa=kappa, loc=mu)
    assert np.allclose(np.var(angles), vonmises_var, atol=0.1)
