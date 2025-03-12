import cftime
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from parcels import Field
from parcels.tools.converters import (
    _get_cftime_calendars,
    _get_cftime_datetimes,
)
from tests.utils import TEST_DATA


def test_field_from_netcdf_variables():
    filename = str(TEST_DATA / "perlinfieldsU.nc")
    dims = {"lon": "x", "lat": "y"}

    variable = "vozocrtx"
    f1 = Field.from_netcdf(filename, variable, dims)
    variable = ("U", "vozocrtx")
    f2 = Field.from_netcdf(filename, variable, dims)
    variable = {"U": "vozocrtx"}
    f3 = Field.from_netcdf(filename, variable, dims)

    assert_allclose(f1.data, f2.data, atol=1e-12)
    assert_allclose(f1.data, f3.data, atol=1e-12)

    with pytest.raises(AssertionError):
        variable = {"U": "vozocrtx", "nav_lat": "nav_lat"}  # multiple variables will fail
        f3 = Field.from_netcdf(filename, variable, dims)


def test_field_from_netcdf():
    filenames = {
        "lon": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "lat": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "data": str(TEST_DATA / "Uu_eastward_nemo_cross_180lon.nc"),
    }
    variable = "U"
    dimensions = {"lon": "glamf", "lat": "gphif"}
    Field.from_netcdf(filenames, variable, dimensions, interp_method="cgrid_velocity")


@pytest.mark.parametrize(
    "calendar, cftime_datetime", zip(_get_cftime_calendars(), _get_cftime_datetimes(), strict=True)
)
def test_field_nonstandardtime(calendar, cftime_datetime, tmpdir):
    xdim = 4
    ydim = 6
    filepath = tmpdir.join("test_nonstandardtime.nc")
    dates = [getattr(cftime, cftime_datetime)(1, m, 1) for m in range(1, 13)]
    da = xr.DataArray(
        np.random.rand(12, ydim, xdim), coords=[dates, range(ydim), range(xdim)], dims=["time", "lat", "lon"], name="U"
    )
    da.to_netcdf(str(filepath))

    dims = {"lon": "lon", "lat": "lat", "time": "time"}
    try:
        field = Field.from_netcdf(filepath, "U", dims)
    except NotImplementedError:
        field = None

    if field is not None:
        assert field.grid.time_origin.calendar == calendar
