import numpy as np
import pytest
import xarray as xr

import parcels._interpolation as interpolation
from tests.utils import create_fieldset_zeros_3d


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


def create_interpolation_data_random(*, with_land_point: bool) -> xr.Dataset:
    tdim, zdim, ydim, xdim = 20, 5, 10, 10
    ds = xr.Dataset(
        {
            "U": (("time", "depth", "lat", "lon"), np.random.random((tdim, zdim, ydim, xdim)) / 1e3),
            "V": (("time", "depth", "lat", "lon"), np.random.random((tdim, zdim, ydim, xdim)) / 1e3),
            "W": (("time", "depth", "lat", "lon"), np.random.random((tdim, zdim, ydim, xdim)) / 1e3),
        },
        coords={
            "time": np.linspace(0, tdim - 1, tdim),
            "depth": np.linspace(0, 1, zdim),
            "lat": np.linspace(0, 1, ydim),
            "lon": np.linspace(0, 1, xdim),
        },
    )
    # Set a land point (for testing freeslip)
    if with_land_point:
        ds["U"][:, :, 2, 5] = 0.0
        ds["V"][:, :, 2, 5] = 0.0
        ds["W"][:, :, 2, 5] = 0.0

    return ds


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
