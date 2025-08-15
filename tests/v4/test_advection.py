import numpy as np
import pytest
import xarray as xr

import parcels
from parcels._datasets.structured.generated import (
    decaying_moving_eddy_dataset,
    moving_eddy_dataset,
    peninsula_dataset,
    radial_rotation_dataset,
    simple_UV_dataset,
    stommel_gyre_dataset,
)
from parcels.application_kernels.advection import AdvectionEE, AdvectionRK4, AdvectionRK4_3D, AdvectionRK45
from parcels.application_kernels.advectiondiffusion import AdvectionDiffusionEM, AdvectionDiffusionM1
from parcels.application_kernels.interpolation import CGrid_Tracer, CGrid_Velocity, XLinear
from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.particle import Particle, Variable
from parcels.particleset import ParticleSet
from parcels.tools.statuscodes import StatusCode
from parcels.xgrid import XGrid

kernel = {
    "EE": AdvectionEE,
    "RK4": AdvectionRK4,
    "RK4_3D": AdvectionRK4_3D,
    "RK45": AdvectionRK45,
    # "AA": AdvectionAnalytical,
    "AdvDiffEM": AdvectionDiffusionEM,
    "AdvDiffM1": AdvectionDiffusionM1,
}


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_advection_zonal(mesh_type, npart=10):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    ds = simple_UV_dataset(mesh_type=mesh_type)
    ds["U"].data[:] = 1.0
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=XLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    pset = ParticleSet(fieldset, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    if mesh_type == "spherical":
        assert (np.diff(pset.lon) > 1.0e-4).all()
    else:
        assert (np.diff(pset.lon) < 1.0e-4).all()


def periodicBC(particle, fieldset, time):
    particle.total_dlon += particle.dlon
    particle.lon = np.fmod(particle.lon, 2)


def test_advection_zonal_periodic():
    ds = simple_UV_dataset(dims=(2, 2, 2, 2), mesh_type="flat")
    ds["U"].data[:] = 0.1
    ds["lon"].data = np.array([0, 2])
    ds["lat"].data = np.array([0, 2])

    # add a halo
    halo = ds.isel(XG=0)
    halo.lon.values = ds.lon.values[1] + 1
    halo.XG.values = ds.XG.values[1] + 2
    ds = xr.concat([ds, halo], dim="XG")

    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    PeriodicParticle = Particle.add_variable(Variable("total_dlon", initial=0))
    startlon = np.array([0.5, 0.4])
    pset = ParticleSet(fieldset, pclass=PeriodicParticle, lon=startlon, lat=[0.5, 0.5])
    pset.execute([AdvectionEE, periodicBC], runtime=np.timedelta64(40, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.total_dlon, 4, atol=1e-5)
    np.testing.assert_allclose(pset.lon_nextloop, startlon, atol=1e-5)
    np.testing.assert_allclose(pset.lat_nextloop, 0.5, atol=1e-5)


def test_horizontal_advection_in_3D_flow(npart=10):
    """Flat 2D zonal flow that increases linearly with depth from 0 m/s to 1 m/s."""
    ds = simple_UV_dataset(mesh_type="flat")
    ds["U"].data[:] = 1.0
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    U.data[:, 0, :, :] = 0.0  # Set U to 0 at the surface
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.linspace(0.1, 0.9, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    expected_lon = pset.depth * (pset.time - fieldset.time_interval.left) / np.timedelta64(1, "s")
    np.testing.assert_allclose(pset.lon, expected_lon, atol=1.0e-1)


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("wErrorThroughSurface", [True, False])
def test_advection_3D_outofbounds(direction, wErrorThroughSurface):
    ds = simple_UV_dataset(mesh_type="flat")
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    U.data[:] = 0.01  # Set U to small value (to avoid horizontal out of bounds)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    W = Field("W", ds["V"], grid, interp_method=XLinear)  # Use V as W for testing
    W.data[:] = -1.0 if direction == "up" else 1.0
    UVW = VectorField("UVW", U, V, W)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, W, UVW, UV])

    def DeleteParticle(particle, fieldset, time):  # pragma: no cover
        particle.state = np.where(particle.state == StatusCode.ErrorOutOfBounds, StatusCode.Delete, particle.state)
        particle.state = np.where(particle.state == StatusCode.ErrorThroughSurface, StatusCode.Delete, particle.state)

    def SubmergeParticle(particle, fieldset, time):  # pragma: no cover
        if len(particle.state) == 0:
            return
        inds = np.argwhere(particle.state == StatusCode.ErrorThroughSurface).flatten()
        if len(inds) == 0:
            return
        dt = particle.dt / np.timedelta64(1, "s")
        (u, v) = fieldset.UV[particle[inds]]
        particle[inds].dlon = u * dt
        particle[inds].dlat = v * dt
        particle[inds].ddepth = 0.0
        particle[inds].depth = 0
        particle[inds].state = StatusCode.Evaluate

    kernels = [AdvectionRK4_3D]
    if wErrorThroughSurface:
        kernels.append(SubmergeParticle)
    kernels.append(DeleteParticle)

    pset = ParticleSet(fieldset=fieldset, lon=0.5, lat=0.5, depth=0.9)
    pset.execute(kernels, runtime=np.timedelta64(11, "s"), dt=np.timedelta64(1, "s"))

    if direction == "up" and wErrorThroughSurface:
        np.testing.assert_allclose(pset.lon[0], 0.6, atol=1e-5)
        np.testing.assert_allclose(pset.depth[0], 0, atol=1e-5)
    else:
        assert len(pset) == 0


@pytest.mark.parametrize("u", [-0.3, np.array(0.2)])
@pytest.mark.parametrize("v", [0.2, np.array(1)])
@pytest.mark.parametrize("w", [None, -0.2, np.array(0.7)])
def test_length1dimensions(u, v, w):  # TODO: Refactor this test to be more readable (and isolate test setup)
    (lon, xdim) = (np.linspace(-10, 10, 21), 21) if isinstance(u, np.ndarray) else (np.array([0]), 1)
    (lat, ydim) = (np.linspace(-15, 15, 31), 31) if isinstance(v, np.ndarray) else (np.array([-4]), 1)
    (depth, zdim) = (
        (np.linspace(-5, 5, 11), 11) if (isinstance(w, np.ndarray) and w is not None) else (np.array([3]), 1)
    )

    tdim = 2  # TODO make this also work for length-1 time dimensions
    dims = (tdim, zdim, ydim, xdim)
    U = u * np.ones(dims, dtype=np.float32)
    V = v * np.ones(dims, dtype=np.float32)
    if w is not None:
        W = w * np.ones(dims, dtype=np.float32)

    ds = xr.Dataset(
        {
            "U": (["time", "depth", "YG", "XG"], U),
            "V": (["time", "depth", "YG", "XG"], V),
        },
        coords={
            "time": (["time"], [np.timedelta64(0, "s"), np.timedelta64(10, "s")], {"axis": "T"}),
            "depth": (["depth"], depth, {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X", "c_grid_axis_shift": -0.5}),
            "lat": (["YG"], lat, {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "lon": (["XG"], lon, {"axis": "X", "c_grid_axis_shift": -0.5}),
        },
    )
    if w:
        ds["W"] = (["time", "depth", "YG", "XG"], W)

    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    fields = [U, V, VectorField("UV", U, V)]
    if w:
        W = Field("W", ds["W"], grid, interp_method=XLinear)
        fields.append(VectorField("UVW", U, V, W))
    fieldset = FieldSet(fields)

    x0, y0, z0 = 2, 8, -4
    pset = ParticleSet(fieldset, lon=x0, lat=y0, depth=z0)
    kernel = AdvectionRK4 if w is None else AdvectionRK4_3D
    pset.execute(kernel, runtime=np.timedelta64(5, "s"), dt=np.timedelta64(1, "s"))

    assert len(pset.lon) == len([p.lon for p in pset])
    np.testing.assert_allclose(np.array([p.lon - x0 for p in pset]), 4 * u, atol=1e-6)
    np.testing.assert_allclose(np.array([p.lat - y0 for p in pset]), 4 * v, atol=1e-6)
    if w:
        np.testing.assert_allclose(np.array([p.depth - z0 for p in pset]), 4 * w, atol=1e-6)


def test_radialrotation(npart=10):
    ds = radial_rotation_dataset()
    grid = XGrid.from_dataset(ds)
    U = parcels.Field("U", ds["U"], grid, mesh_type="flat", interp_method=XLinear)
    V = parcels.Field("V", ds["V"], grid, mesh_type="flat", interp_method=XLinear)
    UV = parcels.VectorField("UV", U, V)
    fieldset = parcels.FieldSet([U, V, UV])

    dt = np.timedelta64(30, "s")
    lon = np.linspace(32, 50, npart)
    lat = np.ones(npart) * 30
    starttime = np.arange(np.timedelta64(0, "s"), npart * dt, dt)

    pset = parcels.ParticleSet(fieldset, lon=lon, lat=lat, time=starttime)
    pset.execute(parcels.AdvectionRK4, endtime=np.timedelta64(10, "m"), dt=dt)

    theta = 2 * np.pi * (pset.time_nextloop - starttime) / np.timedelta64(24 * 3600, "s")
    true_lon = (lon - 30.0) * np.cos(theta) + 30.0
    true_lat = -(lon - 30.0) * np.sin(theta) + 30.0

    np.testing.assert_allclose(pset.lon, true_lon, atol=5e-2)
    np.testing.assert_allclose(pset.lat, true_lat, atol=5e-2)


@pytest.mark.parametrize(
    "method, rtol",
    [
        ("EE", 1e-2),
        ("AdvDiffEM", 1e-2),
        ("AdvDiffM1", 1e-2),
        ("RK4", 1e-5),
        ("RK4_3D", 1e-5),
        ("RK45", 1e-4),
    ],
)
def test_moving_eddy(method, rtol):
    ds = moving_eddy_dataset()
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    if method == "RK4_3D":
        # Using W to test 3D advection (assuming same velocity as V)
        W = Field("W", ds["V"], grid, interp_method=XLinear)
        UVW = VectorField("UVW", U, V, W)
        fieldset = FieldSet([U, V, W, UVW])
    else:
        UV = VectorField("UV", U, V)
        fieldset = FieldSet([U, V, UV])
    if method in ["AdvDiffEM", "AdvDiffM1"]:
        # Add zero diffusivity field for diffusion kernels
        ds["Kh"] = (["time", "depth", "YG", "XG"], np.full(ds["U"].shape, 0))
        fieldset.add_field(Field("Kh", ds["Kh"], grid, interp_method=XLinear), "Kh_zonal")
        fieldset.add_field(Field("Kh", ds["Kh"], grid, interp_method=XLinear), "Kh_meridional")
        fieldset.add_constant("dres", 0.1)

    start_lon, start_lat, start_depth = 12000, 12500, 12500
    dt = np.timedelta64(30, "m")

    if method == "RK45":
        fieldset.add_constant("RK45_tol", rtol)

    pset = ParticleSet(fieldset, lon=start_lon, lat=start_lat, depth=start_depth, time=np.timedelta64(0, "s"))
    pset.execute(kernel[method], dt=dt, endtime=np.timedelta64(1, "h"))

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lat = y_0 - (ds.u_0 - ds.u_g) / ds.f * (1 - np.cos(ds.f * t))
        lon = x_0 + ds.u_g * t + (ds.u_0 - ds.u_g) / ds.f * np.sin(ds.f * t)
        return lon, lat

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, pset.time_nextloop[0])
    np.testing.assert_allclose(pset.lon_nextloop, exp_lon, rtol=rtol)
    np.testing.assert_allclose(pset.lat_nextloop, exp_lat, rtol=rtol)
    if method == "RK4_3D":
        np.testing.assert_allclose(pset.depth_nextloop, exp_lat, rtol=rtol)


@pytest.mark.parametrize(
    "method, rtol",
    [
        ("EE", 1e-1),
        ("RK4", 1e-5),
        ("RK45", 1e-4),
    ],
)
def test_decaying_moving_eddy(method, rtol):
    ds = decaying_moving_eddy_dataset()
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    start_lon, start_lat = 10000, 10000
    dt = np.timedelta64(60, "m")

    if method == "RK45":
        fieldset.add_constant("RK45_tol", rtol)
        fieldset.add_constant("RK45_min_dt", 10 * 60)

    pset = ParticleSet(fieldset, lon=start_lon, lat=start_lat, time=np.timedelta64(0, "s"))
    pset.execute(kernel[method], dt=dt, endtime=np.timedelta64(1, "D"))

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lon = (
            x_0
            + (ds.u_g / ds.gamma_g) * (1 - np.exp(-ds.gamma_g * t))
            + ds.f
            * ((ds.u_0 - ds.u_g) / (ds.f**2 + ds.gamma**2))
            * ((ds.gamma / ds.f) + np.exp(-ds.gamma * t) * (np.sin(ds.f * t) - (ds.gamma / ds.f) * np.cos(ds.f * t)))
        )
        lat = y_0 - ((ds.u_0 - ds.u_g) / (ds.f**2 + ds.gamma**2)) * ds.f * (
            1 - np.exp(-ds.gamma * t) * (np.cos(ds.f * t) + (ds.gamma / ds.f) * np.sin(ds.f * t))
        )
        return lon, lat

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, pset.time_nextloop[0])
    np.testing.assert_allclose(pset.lon_nextloop, exp_lon, rtol=rtol)
    np.testing.assert_allclose(pset.lat_nextloop, exp_lat, rtol=rtol)


@pytest.mark.parametrize(
    "method, rtol",
    [
        ("RK4", 0.1),
        ("RK45", 0.1),
    ],
)
@pytest.mark.parametrize("grid_type", ["A", "C"])
def test_stommelgyre_fieldset(method, rtol, grid_type):
    npart = 2
    ds = stommel_gyre_dataset(grid_type=grid_type)
    grid = XGrid.from_dataset(ds)
    vector_interp_method = None if grid_type == "A" else CGrid_Velocity
    tracer_interp_method = XLinear if grid_type == "A" else CGrid_Tracer
    U = Field("U", ds["U"], grid)
    V = Field("V", ds["V"], grid)
    P = Field("P", ds["P"], grid, interp_method=tracer_interp_method)
    UV = VectorField("UV", U, V, vector_interp_method=vector_interp_method)
    fieldset = FieldSet([U, V, P, UV])

    dt = np.timedelta64(30, "m")
    runtime = np.timedelta64(1, "D")
    start_lon = np.linspace(10e3, 100e3, npart)
    start_lat = np.ones_like(start_lon) * 5000e3

    if method == "RK45":
        fieldset.add_constant("RK45_tol", rtol)

    SampleParticle = Particle.add_variable(
        [Variable("p", initial=0.0, dtype=np.float32), Variable("p_start", initial=0.0, dtype=np.float32)]
    )

    def UpdateP(particle, fieldset, time):  # pragma: no cover
        particle.p = fieldset.P[particle.time, particle.depth, particle.lat, particle.lon]
        particle.p_start = np.where(particle.time == 0, particle.p, particle.p_start)

    pset = ParticleSet(fieldset, pclass=SampleParticle, lon=start_lon, lat=start_lat, time=np.timedelta64(0, "s"))
    pset.execute([kernel[method], UpdateP], dt=dt, runtime=runtime)
    np.testing.assert_allclose(pset.p, pset.p_start, rtol=rtol)


@pytest.mark.parametrize(
    "method, rtol",
    [
        ("RK4", 5e-3),
        ("RK45", 1e-4),
    ],
)
@pytest.mark.parametrize("grid_type", ["A"])  # TODO also implement C-grid once available
def test_peninsula_fieldset(method, rtol, grid_type):
    npart = 2
    ds = peninsula_dataset(grid_type=grid_type)
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    P = Field("P", ds["P"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, P, UV])

    dt = np.timedelta64(30, "m")
    runtime = np.timedelta64(1, "D")
    start_lat = np.linspace(3e3, 47e3, npart)
    start_lon = 3e3 * np.ones_like(start_lat)

    if method == "RK45":
        fieldset.add_constant("RK45_tol", rtol)

    SampleParticle = Particle.add_variable(
        [Variable("p", initial=0.0, dtype=np.float32), Variable("p_start", initial=0.0, dtype=np.float32)]
    )

    def UpdateP(particle, fieldset, time):  # pragma: no cover
        particle.p = fieldset.P[particle.time, particle.depth, particle.lat, particle.lon]
        particle.p_start = np.where(particle.time == 0, particle.p, particle.p_start)

    pset = ParticleSet(fieldset, pclass=SampleParticle, lon=start_lon, lat=start_lat, time=np.timedelta64(0, "s"))
    pset.execute([kernel[method], UpdateP], dt=dt, runtime=runtime)
    np.testing.assert_allclose(pset.p, pset.p_start, rtol=rtol)


def test_nemo_curvilinear_fieldset():
    data_folder = parcels.download_example_dataset("NemoCurvilinear_data")
    files = data_folder.glob("*.nc4")
    ds = xr.open_mfdataset(files, combine="nested", data_vars="minimal", coords="minimal", compat="override")
    ds = ds.isel(time_counter=0, drop=True).drop_vars({"time"}).rename({"glamf": "lon", "gphif": "lat", "z": "depth"})

    xgcm_grid = parcels.xgcm.Grid(
        ds,
        coords={
            "X": {"left": "x"},
            "Y": {"left": "y"},
        },
        periodic=False,
    )
    grid = XGrid(xgcm_grid)

    U = parcels.Field("U", ds["U"], grid)
    V = parcels.Field("V", ds["V"], grid)
    U.units = parcels.GeographicPolar()
    V.units = parcels.Geographic()
    UV = parcels.VectorField("UV", U, V, vector_interp_method=CGrid_Velocity)
    fieldset = parcels.FieldSet([U, V, UV])

    npart = 20
    lonp = 30 * np.ones(npart)
    latp = np.linspace(-70, 88, npart)
    runtime = np.timedelta64(12, "h")  # TODO increase to 160 days

    def periodicBC(particle, fieldSet, time):  # pragma: no cover
        particle.dlon = np.where(particle.lon > 180, particle.dlon - 360, particle.dlon)

    pset = parcels.ParticleSet(fieldset, lon=lonp, lat=latp)
    pset.execute([AdvectionRK4, periodicBC], runtime=runtime, dt=np.timedelta64(6, "h"))
    np.testing.assert_allclose(pset.lat_nextloop, latp, atol=1e-1)
