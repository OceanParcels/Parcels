import numpy as np
import pytest
import xarray as xr

from parcels._datasets.structured.generated import simple_UV_dataset
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels._core.index_search import _search_time_index
from parcels import Field, VectorField
from parcels import FieldSet
from parcels.interpolators import (
    UXPiecewiseLinearNode,
    XFreeslip,
    XLinear,
    XNearest,
    XPartialslip,
    ZeroInterpolator,
)
from parcels.kernels import AdvectionRK4_3D
from parcels import Particle, Variable
from parcels import ParticleFile
from parcels import ParticleSet
from parcels import StatusCode
from parcels import UxGrid
from parcels import XGrid
from tests.utils import TEST_DATA


@pytest.fixture
def field():
    """Reference data used for testing interpolation."""
    z0 = np.array(  # each x is +1 from the previous, each y is +2 from the previous
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0, 9.0],
        ]
    )
    spatial_data = np.array([z0, z0 + 3, z0 + 6, z0 + 9])  # each z is +3 from the previous
    temporal_data = np.array([spatial_data, spatial_data + 10, spatial_data + 20])  # each t is +10 from the previous

    ds = xr.Dataset(
        {"U": (["time", "depth", "lat", "lon"], temporal_data)},
        coords={
            "time": (["time"], [np.timedelta64(t, "s") for t in [0, 2, 4]], {"axis": "T"}),
            "depth": (["depth"], [0, 1, 2, 3], {"axis": "Z"}),
            "lat": (["lat"], [0, 1, 2, 3], {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "lon": (["lon"], [0, 1, 2, 3], {"axis": "X", "c_grid_axis_shift": -0.5}),
            "x": (["x"], [0.5, 1.5, 2.5, 3.5], {"axis": "X"}),
            "y": (["y"], [0.5, 1.5, 2.5, 3.5], {"axis": "Y"}),
        },
    )
    return Field("U", ds["U"], XGrid.from_dataset(ds))


@pytest.mark.parametrize(
    "func, t, z, y, x, expected",
    [
        pytest.param(ZeroInterpolator, np.timedelta64(1, "s"), 2.5, 0.49, 0.51, 0, id="Zero"),
        pytest.param(
            XLinear,
            [np.timedelta64(0, "s"), np.timedelta64(1, "s")],
            [0, 0],
            [0.49, 0.49],
            [0.51, 0.51],
            [1.49, 6.49],
            id="Linear",
        ),
        pytest.param(XLinear, np.timedelta64(1, "s"), 2.5, 0.49, 0.51, 13.99, id="Linear-2"),
        pytest.param(
            XNearest,
            [np.timedelta64(0, "s"), np.timedelta64(3, "s")],
            [0.2, 0.2],
            [0.2, 0.2],
            [0.51, 0.51],
            [1.0, 16.0],
            id="Nearest",
        ),
    ],
)
def test_raw_2d_interpolation(field, func, t, z, y, x, expected):
    """Test the interpolation functions on the Field."""
    tau, ti = _search_time_index(field, t)
    position = field.grid.search(z, y, x)

    value = func(field, ti, position, tau, 0, 0, y, x)
    np.testing.assert_equal(value, expected)


@pytest.mark.parametrize(
    "func, t, z, y, x, expected",
    [
        (XPartialslip, np.timedelta64(1, "s"), 0, 0, 0.0, [[1], [1]]),
        (XFreeslip, np.timedelta64(1, "s"), 0, 0.5, 1.5, [[1], [0.5]]),
        (XPartialslip, np.timedelta64(1, "s"), 0, 2.5, 1.5, [[0.75], [0.5]]),
        (XFreeslip, np.timedelta64(1, "s"), 0, 2.5, 1.5, [[1], [0.5]]),
        (XPartialslip, np.timedelta64(1, "s"), 0, 1.5, 0.5, [[0.5], [0.75]]),
        (XFreeslip, np.timedelta64(1, "s"), 0, 1.5, 0.5, [[0.5], [1]]),
        (
            XFreeslip,
            [np.timedelta64(1, "s"), np.timedelta64(0, "s")],
            [0, 2],
            [1.5, 1.5],
            [2.5, 0.5],
            [[0.5, 0.5], [1, 1]],
        ),
    ],
)
def test_spatial_slip_interpolation(field, func, t, z, y, x, expected):
    field.data[:] = 1.0
    field.data[:, :, 1:3, 1:3] = 0.0  # Set zero land value to test spatial slip
    U = field
    V = field
    UV = VectorField("UV", U, V, vector_interp_method=func)

    velocities = UV[t, z, y, x]
    np.testing.assert_array_almost_equal(velocities, expected)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_interpolation_mesh_type(mesh, npart=10):
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0
    grid = XGrid.from_dataset(ds, mesh=mesh)
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)

    lat = 30.0
    time = U.time_interval.left
    u_expected = 1.0 if mesh == "flat" else 1.0 / (1852 * 60 * np.cos(np.radians(lat)))

    assert np.isclose(U[time, 0, lat, 0], u_expected, atol=1e-7)
    assert V[time, 0, lat, 0] == 0.0

    u, v = UV[time, 0, lat, 0]
    assert np.isclose(u, u_expected, atol=1e-7)
    assert v == 0.0

    assert U.eval(time, 0, lat, 0, applyConversion=False) == 1


def test_default_interpolator_set_correctly():
    ds = simple_UV_dataset()
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid)
    assert U.interp_method == XLinear

    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"])
    U = Field("U", ds["U"], grid)
    assert U.interp_method == UXPiecewiseLinearNode


interp_methods = {
    "linear": XLinear,
}


@pytest.mark.xfail(reason="ParticleFile not implemented yet")
@pytest.mark.parametrize(
    "interp_name",
    [
        "linear",
        # "freeslip",
        # "nearest",
        # "cgrid_velocity",
    ],
)
def test_interp_regression_v3(interp_name):
    """Test that the v4 versions of the interpolation are the same as the v3 versions."""
    ds_input = xr.open_dataset(str(TEST_DATA / f"test_interpolation_data_random_{interp_name}.nc"))
    ydim = ds_input["U"].shape[2]
    xdim = ds_input["U"].shape[3]
    time = [np.timedelta64(int(t), "s") for t in ds_input["time"].values]

    ds = xr.Dataset(
        {
            "U": (["time", "depth", "YG", "XG"], ds_input["U"].values),
            "V": (["time", "depth", "YG", "XG"], ds_input["V"].values),
            "W": (["time", "depth", "YG", "XG"], ds_input["W"].values),
        },
        coords={
            "time": (["time"], time, {"axis": "T"}),
            "depth": (["depth"], ds_input["depth"].values, {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X", "c_grid_axis_shift": -0.5}),
            "lat": (["YG"], ds_input["lat"].values, {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "lon": (["XG"], ds_input["lon"].values, {"axis": "X", "c_grid_axis_shift": -0.5}),
        },
    )

    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U"], grid, interp_method=interp_methods[interp_name])
    V = Field("V", ds["V"], grid, interp_method=interp_methods[interp_name])
    W = Field("W", ds["W"], grid, interp_method=interp_methods[interp_name])
    fieldset = FieldSet([U, V, W, VectorField("UVW", U, V, W)])

    x, y, z = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 13), np.linspace(0, 1, 5))

    TestP = Particle.add_variable(Variable("pid", dtype=np.int32, initial=0))
    pset = ParticleSet(fieldset, pclass=TestP, lon=x, lat=y, depth=z, pid=np.arange(x.size))

    def DeleteParticle(particle, fieldset, time):
        if particle.state >= 50:
            particle.state = StatusCode.Delete

    outfile = ParticleFile(f"test_interpolation_v4_{interp_name}", outputdt=np.timedelta64(1, "s"))
    pset.execute(
        [AdvectionRK4_3D, DeleteParticle],
        runtime=np.timedelta64(4, "s"),
        dt=np.timedelta64(1, "s"),
        output_file=outfile,
    )

    print(str(TEST_DATA / f"test_interpolation_jit_{interp_name}.zarr"))
    ds_v3 = xr.open_zarr(str(TEST_DATA / f"test_interpolation_jit_{interp_name}.zarr"))
    ds_v4 = xr.open_zarr(f"test_interpolation_v4_{interp_name}.zarr")

    tol = 1e-6
    np.testing.assert_allclose(ds_v3.lon, ds_v4.lon, atol=tol)
    np.testing.assert_allclose(ds_v3.lat, ds_v4.lat, atol=tol)
    np.testing.assert_allclose(ds_v3.z, ds_v4.z, atol=tol)
