import numpy as np
import pytest
import xarray as xr

import parcels
from parcels._datasets.structured.generated import moving_eddy_dataset, radial_rotation_dataset, simple_UV_dataset
from parcels.application_kernels.advection import AdvectionEE, AdvectionRK4, AdvectionRK4_3D, AdvectionRK45
from parcels.application_kernels.advectiondiffusion import AdvectionDiffusionEM, AdvectionDiffusionM1
from parcels.application_kernels.interpolation import XBiLinear, XBiLinearPeriodic, XTriLinear
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
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=XBiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=XBiLinear)
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
    particle.lon = np.fmod(particle.lon, fieldset.U.grid.lon[-1])
    particle.lat = np.fmod(particle.lat, fieldset.U.grid.lat[-1])


def test_advection_zonal_periodic():
    ds = simple_UV_dataset(dims=(2, 2, 2, 2), mesh_type="flat")
    ds["U"].data[:] = 0.1
    ds["lon"].data = np.array([0, 2])
    ds["lat"].data = np.array([0, 2])

    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XBiLinearPeriodic)
    V = Field("V", ds["V"], grid, interp_method=XBiLinearPeriodic)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    PeriodicParticle = Particle.add_variable(Variable("total_dlon", initial=0))
    pset = ParticleSet(fieldset, pclass=PeriodicParticle, lon=[0.5], lat=[0.5])
    pset.execute([AdvectionEE, periodicBC], runtime=np.timedelta64(40, "s"), dt=np.timedelta64(1, "s"))
    assert np.isclose(pset.total_dlon[0], 4, atol=1e-5)
    assert np.isclose(pset.lon_nextloop[0], 0.5, atol=1e-5)


def test_horizontal_advection_in_3D_flow(npart=10):
    """Flat 2D zonal flow that increases linearly with depth from 0 m/s to 1 m/s."""
    ds = simple_UV_dataset(mesh_type="flat")
    ds["U"].data[:] = 1.0
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XTriLinear)
    U.data[:, 0, :, :] = 0.0  # Set U to 0 at the surface
    V = Field("V", ds["V"], grid, interp_method=XTriLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.linspace(0.1, 0.9, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    expected_lon = pset.depth * (pset.time - fieldset.time_interval.left) / np.timedelta64(1, "s")
    assert np.allclose(expected_lon, pset.lon_nextloop, atol=1.0e-1)


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("wErrorThroughSurface", [True, False])
def test_advection_3D_outofbounds(direction, wErrorThroughSurface):
    ds = simple_UV_dataset(mesh_type="flat")
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XTriLinear)
    U.data[:] = 0.01  # Set U to 0 at the surface
    V = Field("V", ds["V"], grid, interp_method=XTriLinear)
    W = Field("W", ds["V"], grid, interp_method=XTriLinear)  # Use V as W for testing
    W.data[:] = -1.0 if direction == "up" else 1.0
    UVW = VectorField("UVW", U, V, W)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, W, UVW, UV])

    def DeleteParticle(particle, fieldset, time):  # pragma: no cover
        if particle.state == StatusCode.ErrorOutOfBounds or particle.state == StatusCode.ErrorThroughSurface:
            particle.state = StatusCode.Delete

    def SubmergeParticle(particle, fieldset, time):  # pragma: no cover
        if particle.state == StatusCode.ErrorThroughSurface:
            dt = particle.dt / np.timedelta64(1, "s")
            (u, v) = fieldset.UV[particle]
            particle.dlon = u * dt
            particle.dlat = v * dt
            particle.ddepth = 0.0
            particle.depth = 0
            particle.state = StatusCode.Evaluate

    kernels = [AdvectionRK4_3D]
    if wErrorThroughSurface:
        kernels.append(SubmergeParticle)
    kernels.append(DeleteParticle)

    pset = ParticleSet(fieldset=fieldset, lon=0.5, lat=0.5, depth=0.9)
    pset.execute(kernels, runtime=np.timedelta64(11, "s"), dt=np.timedelta64(1, "s"))

    if direction == "up" and wErrorThroughSurface:
        assert np.allclose(pset.lon[0], 0.6)
        assert np.allclose(pset.depth[0], 0)
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
    U = Field("U", ds["U"], grid, interp_method=XTriLinear)
    V = Field("V", ds["V"], grid, interp_method=XTriLinear)
    fields = [U, V, VectorField("UV", U, V)]
    if w:
        W = Field("W", ds["W"], grid, interp_method=XTriLinear)
        fields.append(VectorField("UVW", U, V, W))
    fieldset = FieldSet(fields)

    x0, y0, z0 = 2, 8, -4
    pset = ParticleSet(fieldset, lon=x0, lat=y0, depth=z0)
    kernel = AdvectionRK4 if w is None else AdvectionRK4_3D
    pset.execute(kernel, runtime=np.timedelta64(5, "s"), dt=np.timedelta64(1, "s"))

    assert len(pset.lon) == len([p.lon for p in pset])
    assert ((np.array([p.lon - x0 for p in pset]) - 4 * u) < 1e-6).all()
    assert ((np.array([p.lat - y0 for p in pset]) - 4 * v) < 1e-6).all()
    if w:
        assert ((np.array([p.depth - z0 for p in pset]) - 4 * w) < 1e-6).all()


def test_radialrotation(npart=10):
    ds = radial_rotation_dataset()
    grid = XGrid.from_dataset(ds)
    U = parcels.Field("U", ds["U"], grid, mesh_type="flat", interp_method=XBiLinear)
    V = parcels.Field("V", ds["V"], grid, mesh_type="flat", interp_method=XBiLinear)
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

    assert np.allclose(pset.lon, true_lon, atol=5e-2)
    assert np.allclose(pset.lat, true_lat, atol=5e-2)


@pytest.mark.parametrize(
    "method, rtol",
    [
        ("EE", 1e-2),
        ("AdvDiffEM", 1e-2),
        ("AdvDiffM1", 1e-2),
        ("RK4", 1e-5),
        ("RK4_3D", 1e-5),
        ("RK45", 1e-5),
    ],
)
def test_moving_eddy(method, rtol):
    ds = moving_eddy_dataset()
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=XBiLinear)
    V = Field("V", ds["V"], grid, interp_method=XBiLinear)
    if method == "RK4_3D":
        # Using W to test 3D advection (assuming same velocity as V)
        W = Field("W", ds["V"], grid, interp_method=XTriLinear)
        UVW = VectorField("UVW", U, V, W)
        fieldset = FieldSet([U, V, W, UVW])
    else:
        UV = VectorField("UV", U, V)
        fieldset = FieldSet([U, V, UV])
    if method in ["AdvDiffEM", "AdvDiffM1"]:
        # Add zero diffusivity field for diffusion kernels
        ds["Kh"] = (["time", "depth", "YG", "XG"], np.full(ds["U"].shape, 0))
        fieldset.add_field(Field("Kh", ds["Kh"], grid, interp_method=XBiLinear), "Kh_zonal")
        fieldset.add_field(Field("Kh", ds["Kh"], grid, interp_method=XBiLinear), "Kh_meridional")
        fieldset.add_constant("dres", 0.1)

    start_lon, start_lat, start_depth = 12000, 12500, 12500
    dt = np.timedelta64(3, "m")

    if method == "RK45":
        # Use RK45Particles to set next_dt
        RK45Particles = Particle.add_variable(Variable("next_dt", initial=dt, dtype=np.timedelta64))
        fieldset.add_constant("RK45_tol", 1e-6)

    pclass = RK45Particles if method == "RK45" else Particle
    pset = ParticleSet(
        fieldset, pclass=pclass, lon=start_lon, lat=start_lat, depth=start_depth, time=np.timedelta64(0, "s")
    )
    pset.execute(kernel[method], dt=dt, endtime=np.timedelta64(6, "h"))

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lat = y_0 - (ds.u_0 - ds.u_g) / ds.f * (1 - np.cos(ds.f * t))
        lon = x_0 + ds.u_g * t + (ds.u_0 - ds.u_g) / ds.f * np.sin(ds.f * t)
        return lon, lat

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, pset.time_nextloop[0])
    assert np.allclose(pset.lon_nextloop, exp_lon, rtol=rtol)
    assert np.allclose(pset.lat_nextloop, exp_lat, rtol=rtol)
    if method == "RK4_3D":
        assert np.allclose(pset.depth_nextloop, exp_lat, rtol=rtol)
