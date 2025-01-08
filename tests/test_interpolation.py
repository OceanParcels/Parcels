import numpy as np
import pytest
import xarray as xr

import parcels._interpolation as interpolation


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


@pytest.fixture
def data_3d():
    """Reference data used for testing interpolation."""
    return create_interpolation_data().values


@pytest.fixture
def tmp_interpolator_registry():
    """Resets the interpolator registry after the test. Vital when testing manipulating the registry."""
    old_2d = interpolation.interpolator_registry_2d.copy()
    old_3d = interpolation.interpolator_registry_3d.copy()
    yield
    interpolation.interpolator_registry_2d = old_2d
    interpolation.interpolator_registry_3d = old_3d


def test_interpolation_registry(tmp_interpolator_registry):
    @interpolation.register_3d_interpolator("test")
    @interpolation.register_2d_interpolator("test")
    def some_function():
        return "test"

    assert "test" in interpolation.interpolator_registry_2d
    assert "test" in interpolation.interpolator_registry_3d

    f = interpolation.interpolator_registry_2d["test"]
    g = interpolation.interpolator_registry_3d["test"]
    assert f() == g() == "test"
