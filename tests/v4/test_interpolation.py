import numpy as np
import pytest
import xarray as xr

from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.application_kernels.advection import AdvectionRK4_3D
from parcels.application_kernels.interpolation import XBiLinear, XTriLinear
from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.particle import Particle, Variable
from parcels.particleset import ParticleSet
from parcels.xgrid import XGrid
from tests.utils import TEST_DATA


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_interpolation_mesh_type(mesh_type, npart=10):
    ds = simple_UV_dataset(mesh_type=mesh_type)
    ds["U"].data[:] = 1.0
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=XBiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=XBiLinear)
    UV = VectorField("UV", U, V)

    lat = 30.0
    time = U.time_interval.left
    u_expected = 1.0 if mesh_type == "flat" else 1.0 / (1852 * 60 * np.cos(np.radians(lat)))

    assert np.isclose(U[time, 0, lat, 0], u_expected, atol=1e-7)
    assert V[time, 0, lat, 0] == 0.0

    u, v = UV[time, 0, lat, 0]
    assert np.isclose(u, u_expected, atol=1e-7)
    assert v == 0.0

    assert U.eval(time, 0, lat, 0, applyConversion=False) == 1


interp_methods = {
    "linear": XTriLinear,
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

    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type="flat", interp_method=interp_methods[interp_name])
    V = Field("V", ds["V"], grid, mesh_type="flat", interp_method=interp_methods[interp_name])
    W = Field("W", ds["W"], grid, mesh_type="flat", interp_method=interp_methods[interp_name])
    fieldset = FieldSet([U, V, W, VectorField("UVW", U, V, W)])

    x, y, z = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 13), np.linspace(0, 1, 5))

    TestP = Particle.add_variable(Variable("pid", dtype=np.int32, initial=0))
    pset = ParticleSet(fieldset, pclass=TestP, lon=x, lat=y, depth=z, pid=np.arange(x.size))

    def DeleteParticle(particle, fieldset, time):
        if particle.state >= 50:
            particle.delete()

    outfile = pset.ParticleFile(f"test_interpolation_v4_{interp_name}", outputdt=np.timedelta64(1, "s"))
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
