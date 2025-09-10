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
