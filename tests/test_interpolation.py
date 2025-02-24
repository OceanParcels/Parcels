import numpy as np
import pytest
import xarray as xr

import parcels._interpolation as interpolation
from parcels import AdvectionRK4_3D, FieldSet, Particle, ParticleSet
from tests.utils import TEST_DATA, create_fieldset_zeros_3d


@pytest.fixture
def tmp_interpolator_registry():
    """Resets the interpolator registry after the test. Vital when testing manipulating the registry."""
    old_2d = interpolation._interpolator_registry_2d.copy()
    old_3d = interpolation._interpolator_registry_3d.copy()
    yield
    interpolation._interpolator_registry_2d = old_2d
    interpolation._interpolator_registry_3d = old_3d


@pytest.mark.usefixtures("tmp_interpolator_registry")
def test_interpolation_registry():
    @interpolation.register_3d_interpolator("test")
    @interpolation.register_2d_interpolator("test")
    def some_function():
        return "test"

    assert "test" in interpolation.get_2d_interpolator_registry()
    assert "test" in interpolation.get_3d_interpolator_registry()

    f = interpolation.get_2d_interpolator_registry()["test"]
    g = interpolation.get_3d_interpolator_registry()["test"]
    assert f() == g() == "test"


def create_interpolation_data():
    """Reference data used for testing interpolation.

    Most interpolation will be focussed around index
    (depth, lat, lon) = (zi, yi, xi) = (1, 1, 1) with ti=0.
    """
    z0 = np.array(  # each x is +1 from the previous, each y is +2 from the previous
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0, 9.0],
        ]
    )
    spatial_data = [z0, z0 + 3, z0 + 6, z0 + 9]  # each z is +3 from the previous
    return xr.DataArray([spatial_data, spatial_data, spatial_data], dims=("time", "depth", "lat", "lon"))


@pytest.fixture
def data_2d():
    """2D slice of the reference data at depth=0."""
    return create_interpolation_data().isel(depth=0).values


@pytest.mark.parametrize(
    "func, eta, xsi, expected",
    [
        pytest.param(interpolation._nearest_2d, 0.49, 0.49, 3.0, id="nearest_2d-1"),
        pytest.param(interpolation._nearest_2d, 0.49, 0.51, 4.0, id="nearest_2d-2"),
        pytest.param(interpolation._nearest_2d, 0.51, 0.49, 5.0, id="nearest_2d-3"),
        pytest.param(interpolation._nearest_2d, 0.51, 0.51, 6.0, id="nearest_2d-4"),
        pytest.param(interpolation._tracer_2d, None, None, 6.0, id="tracer_2d"),
    ],
)
def test_raw_2d_interpolation(data_2d, func, eta, xsi, expected):
    """Test the 2D interpolation functions on the raw arrays."""
    ti = 0
    yi, xi = 1, 1
    ctx = interpolation.InterpolationContext2D(data_2d, eta, xsi, ti, yi, xi)
    assert func(ctx) == expected


@pytest.mark.usefixtures("tmp_interpolator_registry")
def test_interpolator_override():
    fieldset = create_fieldset_zeros_3d()

    @interpolation.register_3d_interpolator("linear")
    def test_interpolator(ctx: interpolation.InterpolationContext3D):
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        fieldset.U[0, 0.5, 0.5, 0.5]


@pytest.mark.usefixtures("tmp_interpolator_registry")
def test_full_depth_provided_to_interpolators():
    """The full depth needs to be provided to the interpolation schemes as some interpolators
    need to know whether they are at the surface or bottom of the water column.

    https://github.com/OceanParcels/Parcels/pull/1816#discussion_r1908840408
    """
    xdim, ydim, zdim = 10, 11, 12
    fieldset = create_fieldset_zeros_3d(xdim=xdim, ydim=ydim, zdim=zdim)

    @interpolation.register_3d_interpolator("linear")
    def test_interpolator2(ctx: interpolation.InterpolationContext3D):
        assert ctx.data.shape[1] == zdim
        # The array z dimension is the same as the fieldset z dimension
        return 0

    fieldset.U[0.5, 0.5, 0.5, 0.5]


@pytest.mark.parametrize(
    "interp_method",
    [
        "linear",
        "freeslip",
        "nearest",
        "cgrid_velocity",
    ],
)
def test_interp_regression_v3(interp_method):
    """Test that the v4 versions of the interpolation are the same as the v3 versions."""
    variables = {"U": "U", "V": "V", "W": "W"}
    dimensions = {"time": "time", "lon": "lon", "lat": "lat", "depth": "depth"}
    ds = xr.open_dataset(str(TEST_DATA / f"test_interpolation_data_random_{interp_method}.nc"))
    fieldset = FieldSet.from_xarray_dataset(
        ds,
        variables,
        dimensions,
        mesh="flat",
    )

    for field in [fieldset.U, fieldset.V, fieldset.W]:  # Set a land point (for testing freeslip)
        field.interp_method = interp_method

    x, y, z = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 13), np.linspace(0, 1, 5))

    TestP = Particle.add_variable("pid", dtype=np.int32, initial=0)
    pset = ParticleSet(fieldset, pclass=TestP, lon=x, lat=y, depth=z, pid=np.arange(x.size))

    def DeleteParticle(particle, fieldset, time):
        if particle.state >= 50:
            particle.delete()

    outfile = pset.ParticleFile(f"test_interpolation_v4_{interp_method}", outputdt=1)
    pset.execute([AdvectionRK4_3D, DeleteParticle], runtime=4, dt=1, output_file=outfile)

    ds_v3 = xr.open_zarr(str(TEST_DATA / f"test_interpolation_jit_{interp_method}.zarr"))
    ds_v4 = xr.open_zarr(f"test_interpolation_v4_{interp_method}.zarr")

    tol = 1e-6
    np.testing.assert_allclose(ds_v3.lon, ds_v4.lon, atol=tol)
    np.testing.assert_allclose(ds_v3.lat, ds_v4.lat, atol=tol)
    np.testing.assert_allclose(ds_v3.z, ds_v4.z, atol=tol)
