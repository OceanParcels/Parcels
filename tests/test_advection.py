import math
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from parcels import (
    AdvectionAnalytical,
    AdvectionDiffusionEM,
    AdvectionDiffusionM1,
    AdvectionEE,
    AdvectionRK4,
    AdvectionRK4_3D,
    AdvectionRK45,
    FieldSet,
    Particle,
    ParticleSet,
    Variable,
)
from tests.utils import TEST_DATA

kernel = {
    "EE": AdvectionEE,
    "RK4": AdvectionRK4,
    "RK45": AdvectionRK45,
    "AA": AdvectionAnalytical,
    "AdvDiffEM": AdvectionDiffusionEM,
    "AdvDiffM1": AdvectionDiffusionM1,
}

# Some constants
f = 1.0e-4
u_0 = 0.3
u_g = 0.04
gamma = 1 / (86400.0 * 2.89)
gamma_g = 1 / (86400.0 * 28.9)


@pytest.fixture
def lon():
    xdim = 200
    return np.linspace(-170, 170, xdim, dtype=np.float32)


@pytest.fixture
def lat():
    ydim = 100
    return np.linspace(-80, 80, ydim, dtype=np.float32)


@pytest.fixture
def depth():
    zdim = 2
    return np.linspace(0, 30, zdim, dtype=np.float32)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
def test_advection_meridional(lon, lat):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    npart = 10
    data = {"U": np.zeros((lat.size, lon.size), dtype=np.float32), "V": np.ones((lat.size, lon.size), dtype=np.float32)}
    dimensions = {"lon": lon, "lat": lat}
    fieldset = FieldSet.from_data(data, dimensions, mesh="spherical")

    pset = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(-60, 60, npart), lat=np.linspace(0, 30, npart))
    delta_lat = np.diff(pset.lat)
    pset.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))
    assert np.allclose(np.diff(pset.lat), delta_lat, rtol=1.0e-4)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("rk45_tol", [10, 100])
def test_advection_RK45(lon, lat, rk45_tol):
    npart = 10
    data2D = {
        "U": np.ones((lat.size, lon.size), dtype=np.float32),
        "V": np.zeros((lat.size, lon.size), dtype=np.float32),
    }
    dimensions = {"lon": lon, "lat": lat}
    fieldset = FieldSet.from_data(data2D, dimensions, mesh="spherical")
    fieldset.add_constant("RK45_tol", rk45_tol)

    dt = timedelta(seconds=30).total_seconds()
    RK45Particles = Particle.add_variable("next_dt", dtype=np.float32, initial=dt)
    pset = ParticleSet(fieldset, pclass=RK45Particles, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset.execute(AdvectionRK45, runtime=timedelta(hours=2), dt=dt)
    assert (np.diff(pset.lon) > 1.0e-4).all()
    assert np.isclose(fieldset.RK45_tol, rk45_tol / (1852 * 60))
    print(fieldset.RK45_tol)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="When refactoring fieldfilebuffer croco support was dropped. This will be fixed in v4.")
def test_conversion_3DCROCO():
    """Test of the (SciPy) version of the conversion from depth to sigma in CROCO

    Values below are retrieved using xroms and hardcoded in the method (to avoid dependency on xroms):
    ```py
    x, y = 10, 20
    s_xroms = ds.s_w.values
    z_xroms = ds.z_w.isel(time=0).isel(eta_rho=y).isel(xi_rho=x).values
    lat, lon = ds.y_rho.values[y, x], ds.x_rho.values[y, x]
    ```
    """
    fieldset = FieldSet.from_modulefile(TEST_DATA / "fieldset_CROCO3D.py")

    lat, lon = 78000.0, 38000.0
    s_xroms = np.array([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0], dtype=np.float32)
    z_xroms = np.array(
        [
            -1.26000000e02,
            -1.10585846e02,
            -9.60985413e01,
            -8.24131317e01,
            -6.94126511e01,
            -5.69870148e01,
            -4.50318756e01,
            -3.34476166e01,
            -2.21383114e01,
            -1.10107975e01,
            2.62768921e-02,
        ],
        dtype=np.float32,
    )

    sigma = np.zeros_like(z_xroms)
    from parcels.field import _croco_from_z_to_sigma_scipy

    for zi, z in enumerate(z_xroms):
        sigma[zi] = _croco_from_z_to_sigma_scipy(fieldset, 0, z, lat, lon, None)

    assert np.allclose(sigma, s_xroms, atol=1e-3)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="CROCO 3D interpolation is not yet implemented correctly in v4. ")
def test_advection_3DCROCO():
    fieldset = FieldSet.from_modulefile(TEST_DATA / "fieldset_CROCO3D.py")

    runtime = 1e4
    X, Z = np.meshgrid([40e3, 80e3, 120e3], [-10, -130])
    Y = np.ones(X.size) * 100e3

    pclass = Particle.add_variable(Variable("w"))
    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=X, lat=Y, depth=Z)

    def SampleW(particle, fieldset, time):  # pragma: no cover
        particle.w = fieldset.W[time, particle.depth, particle.lat, particle.lon]

    pset.execute([AdvectionRK4_3D, SampleW], runtime=runtime, dt=100)
    assert np.allclose(pset.depth, Z.flatten(), atol=5)  # TODO lower this atol
    assert np.allclose(pset.lon_nextloop, [x + runtime for x in X.flatten()], atol=1e-3)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="When refactoring fieldfilebuffer croco support was dropped. This will be fixed in v4.")
def test_advection_2DCROCO():
    fieldset = FieldSet.from_modulefile(TEST_DATA / "fieldset_CROCO2D.py")

    runtime = 1e4
    X = np.array([40e3, 80e3, 120e3])
    Y = np.ones(X.size) * 100e3
    Z = np.zeros(X.size)
    pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=X, lat=Y, depth=Z)

    pset.execute([AdvectionRK4], runtime=runtime, dt=100)
    assert np.allclose(pset.depth, Z.flatten(), atol=1e-3)
    assert np.allclose(pset.lon_nextloop, [x + runtime for x in X], atol=1e-3)


def create_periodic_fieldset(xdim, ydim, uvel, vvel):
    dimensions = {
        "lon": np.linspace(0.0, 1.0, xdim + 1, dtype=np.float32)[1:],  # don't include both 0 and 1, for periodic b.c.
        "lat": np.linspace(0.0, 1.0, ydim + 1, dtype=np.float32)[1:],
    }

    data = {"U": uvel * np.ones((ydim, xdim), dtype=np.float32), "V": vvel * np.ones((ydim, xdim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh="spherical")


def periodicBC(particle, fieldset, time):  # pragma: no cover
    particle.lon = math.fmod(particle.lon, 1)
    particle.lat = math.fmod(particle.lat, 1)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="Calls fieldset.add_periodic_halo(). In v4, interpolation should work without adding halo.")
def test_advection_periodic_meridional():
    xdim, ydim = 100, 100
    fieldset = create_periodic_fieldset(xdim, ydim, uvel=0.0, vvel=1.0)
    fieldset.add_periodic_halo(meridional=True)
    assert len(fieldset.U.lat) == ydim + 10  # default halo size is 5 grid points

    pset = ParticleSet(fieldset, pclass=Particle, lon=[0.5], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=timedelta(hours=20), dt=timedelta(seconds=30))
    assert abs(pset.lat[0] - 0.15) < 0.1


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="Calls fieldset.add_periodic_halo(). In v4, interpolation should work without adding halo.")
def test_advection_periodic_zonal_meridional():
    xdim, ydim = 100, 100
    fieldset = create_periodic_fieldset(xdim, ydim, uvel=1.0, vvel=1.0)
    fieldset.add_periodic_halo(zonal=True, meridional=True)
    assert len(fieldset.U.lat) == ydim + 10  # default halo size is 5 grid points
    assert len(fieldset.U.lon) == xdim + 10  # default halo size is 5 grid points
    assert np.allclose(np.diff(fieldset.U.lat), fieldset.U.lat[1] - fieldset.U.lat[0], rtol=0.001)
    assert np.allclose(np.diff(fieldset.U.lon), fieldset.U.lon[1] - fieldset.U.lon[0], rtol=0.001)

    pset = ParticleSet(fieldset, pclass=Particle, lon=[0.4], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=timedelta(hours=20), dt=timedelta(seconds=30))
    assert abs(pset.lon[0] - 0.05) < 0.1
    assert abs(pset.lat[0] - 0.15) < 0.1


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("u", [-0.3, np.array(0.2)])
@pytest.mark.parametrize("v", [0.2, np.array(1)])
@pytest.mark.parametrize("w", [None, -0.2, np.array(0.7)])
def test_length1dimensions(u, v, w):
    (lon, xdim) = (np.linspace(-10, 10, 21), 21) if isinstance(u, np.ndarray) else (0, 1)
    (lat, ydim) = (np.linspace(-15, 15, 31), 31) if isinstance(v, np.ndarray) else (-4, 1)
    (depth, zdim) = (np.linspace(-5, 5, 11), 11) if (isinstance(w, np.ndarray) and w is not None) else (3, 1)
    dimensions = {"lon": lon, "lat": lat, "depth": depth}

    dims = []
    if zdim > 1:
        dims.append(zdim)
    if ydim > 1:
        dims.append(ydim)
    if xdim > 1:
        dims.append(xdim)
    if len(dims) > 0:
        U = u * np.ones(dims, dtype=np.float32)
        V = v * np.ones(dims, dtype=np.float32)
        if w is not None:
            W = w * np.ones(dims, dtype=np.float32)
    else:
        U, V, W = u, v, w

    data = {"U": U, "V": V}
    if w is not None:
        data["W"] = W
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")

    x0, y0, z0 = 2, 8, -4
    pset = ParticleSet(fieldset, pclass=Particle, lon=x0, lat=y0, depth=z0)
    pfunc = AdvectionRK4 if w is None else AdvectionRK4_3D
    kernel = pset.Kernel(pfunc)
    pset.execute(kernel, runtime=5, dt=1)

    assert len(pset.lon) == len([p.lon for p in pset])
    assert ((np.array([p.lon - x0 for p in pset]) - 4 * u) < 1e-6).all()
    assert ((np.array([p.lat - y0 for p in pset]) - 4 * v) < 1e-6).all()
    if w:
        assert ((np.array([p.depth - y0 for p in pset]) - 4 * w) < 1e-6).all()


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
def test_analyticalAgrid():
    lon = np.arange(0, 15, dtype=np.float32)
    lat = np.arange(0, 15, dtype=np.float32)
    U = np.ones((lat.size, lon.size), dtype=np.float32)
    V = np.ones((lat.size, lon.size), dtype=np.float32)
    fieldset = FieldSet.from_data({"U": U, "V": V}, {"lon": lon, "lat": lat}, mesh="flat")
    pset = ParticleSet(fieldset, pclass=Particle, lon=1, lat=1)

    with pytest.raises(NotImplementedError):
        pset.execute(AdvectionAnalytical, runtime=1)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1927")
@pytest.mark.parametrize("u", [1, -0.2, -0.3, 0])
@pytest.mark.parametrize("v", [1, -0.3, 0, -1])
@pytest.mark.parametrize("w", [None, 1, -0.3, 0, -1])
@pytest.mark.parametrize("direction", [1, -1])
def test_uniform_analytical(u, v, w, direction, tmp_zarrfile):
    lon = np.arange(0, 15, dtype=np.float32)
    lat = np.arange(0, 15, dtype=np.float32)
    if w is not None:
        depth = np.arange(0, 40, 2, dtype=np.float32)
        U = u * np.ones((depth.size, lat.size, lon.size), dtype=np.float32)
        V = v * np.ones((depth.size, lat.size, lon.size), dtype=np.float32)
        W = w * np.ones((depth.size, lat.size, lon.size), dtype=np.float32)
        fieldset = FieldSet.from_data({"U": U, "V": V, "W": W}, {"lon": lon, "lat": lat, "depth": depth}, mesh="flat")
        fieldset.W.interp_method = "cgrid_velocity"
    else:
        U = u * np.ones((lat.size, lon.size), dtype=np.float32)
        V = v * np.ones((lat.size, lon.size), dtype=np.float32)
        fieldset = FieldSet.from_data({"U": U, "V": V}, {"lon": lon, "lat": lat}, mesh="flat")
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"

    x0, y0, z0 = 6.1, 6.2, 20
    pset = ParticleSet(fieldset, pclass=Particle, lon=x0, lat=y0, depth=z0)

    outfile = pset.ParticleFile(name=tmp_zarrfile, outputdt=1, chunks=(1, 1))
    pset.execute(AdvectionAnalytical, runtime=4, dt=direction, output_file=outfile)
    assert np.abs(pset.lon - x0 - pset.time * u) < 1e-6
    assert np.abs(pset.lat - y0 - pset.time * v) < 1e-6
    if w is not None:
        assert np.abs(pset.depth - z0 - pset.time * w) < 1e-4

    ds = xr.open_zarr(tmp_zarrfile)
    times = (direction * ds["time"][:]).values.astype("timedelta64[s]")[0]
    timeref = np.arange(1, 5).astype("timedelta64[s]")
    assert np.allclose(times, timeref, atol=np.timedelta64(1, "ms"))
    lons = ds["lon"][:].values
    assert np.allclose(lons, x0 + direction * u * np.arange(1, 5))
