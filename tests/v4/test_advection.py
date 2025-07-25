import numpy as np
import pytest

from parcels._datasets.structured.generic import datasets, simple_UV_dataset
from parcels.application_kernels import AdvectionEE, AdvectionRK4
from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.particle import Particle, Variable
from parcels.particleset import ParticleSet
from parcels.xgrid import _XGRID_AXES, XGrid


def BiLinear(  # TODO move to interpolation file
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Bilinear interpolation on a regular grid."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    data = field.data.data[:, zi, yi : yi + 2, xi : xi + 2]
    data = (1 - tau) * data[ti, :, :] + tau * data[ti + 1, :, :]

    return (
        (1 - xsi) * (1 - eta) * data[0, 0]
        + xsi * (1 - eta) * data[0, 1]
        + xsi * eta * data[1, 1]
        + (1 - xsi) * eta * data[1, 0]
    )


def TriLinear(  # TODO move to interpolation file
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Trilinear interpolation on a regular grid."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    data = field.data.data[:, zi : zi + 2, yi : yi + 2, xi : xi + 2]
    data = (1 - tau) * data[ti, :, :, :] + tau * data[ti + 1, :, :, :]
    data = (1 - zeta) * data[zi, :, :] + zeta * data[zi + 1, :, :]

    return (
        (1 - xsi) * (1 - eta) * data[0, 0]
        + xsi * (1 - eta) * data[0, 1]
        + xsi * eta * data[1, 1]
        + (1 - xsi) * eta * data[1, 0]
    )


kernel = {
    "EE": AdvectionEE,
    "RK4": AdvectionRK4,
    # "RK45": AdvectionRK45,
    # "AA": AdvectionAnalytical,
    # "AdvDiffEM": AdvectionDiffusionEM,
    # "AdvDiffM1": AdvectionDiffusionM1,
}


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_advection_zonal(mesh_type, npart=10):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    ds = datasets[f"pure_zonal_flow_{mesh_type}"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=BiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=BiLinear)
    UV = VectorField("UV", U, V)
    fieldset2D = FieldSet([U, V, UV])

    pset2D = ParticleSet(fieldset2D, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset2D.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    if mesh_type == "spherical":
        assert (np.diff(pset2D.lon) > 1.0e-4).all()
    else:
        assert (np.diff(pset2D.lon) < 1.0e-4).all()


def test_advection_3D(npart=10):
    """Flat 2D zonal flow that increases linearly with depth from 0 m/s to 1 m/s."""
    ds = datasets["pure_zonal_flow_flat"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=TriLinear)
    U.data[:, 0, :, :] = 0.0  # Set U to 0 at the surface
    V = Field("V", ds["V"], grid, interp_method=TriLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.linspace(0.1, 0.9, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    expected_lon = pset.depth * (pset.time - fieldset.time_interval.left) / np.timedelta64(1, "s")
    assert np.allclose(expected_lon, pset.lon_nextloop, atol=1.0e-1)


@pytest.mark.parametrize(
    "method, rtol",
    [
        ("EE", 1e-2),
        # ("AdvDiffEM", 1e-2),
        # ("AdvDiffM1", 1e-2),
        ("RK4", 1e-5),
        # ("RK45", 1e-5),
    ],
)
def test_moving_eddy(method, rtol):
    f, u_0, u_g = 1.0e-4, 0.3, 0.04  # Some constants
    start_lon, start_lat = 12000, 12500

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lat = y_0 - (u_0 - u_g) / f * (1 - np.cos(f * t))
        lon = x_0 + u_g * t + (u_0 - u_g) / f * np.sin(f * t)
        return lon, lat

    dt = np.timedelta64(3, "m")
    time = np.arange(np.timedelta64(0, "s"), np.timedelta64(7, "h"), np.timedelta64(1, "m"))
    ds = simple_UV_dataset(dims=(len(time), 2, 2, 2), mesh_type="flat")
    grid = XGrid.from_dataset(ds)
    for t in range(len(time)):
        ds["U"].data[t, :, :, :] = u_g + (u_0 - u_g) * np.cos(f * (time[t] / np.timedelta64(1, "s")))
        ds["V"].data[t, :, :, :] = -(u_0 - u_g) * np.sin(f * (time[t] / np.timedelta64(1, "s")))
    ds["lon"].data = np.array([0, 25000])
    ds["lat"].data = np.array([0, 25000])
    ds = ds.assign_coords(time=time)
    U = Field("U", ds["U"], grid, interp_method=BiLinear)
    V = Field("V", ds["V"], grid, interp_method=BiLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    RK45Particles = Particle.add_variable(Variable("next_dt", initial=dt))

    pclass = RK45Particles if method == "RK45" else Particle
    pset = ParticleSet(fieldset, pclass=pclass, lon=start_lon, lat=start_lat, time=np.timedelta64(0, "s"))
    pset.execute(kernel[method], dt=dt, endtime=np.timedelta64(6, "h"))

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, pset.time[0])
    assert np.allclose(pset.lon_nextloop, exp_lon, rtol=rtol)
    assert np.allclose(pset.lat_nextloop, exp_lat, rtol=rtol)
