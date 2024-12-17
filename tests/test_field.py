from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

from parcels import Field
from parcels.field import _expand_filename, _sanitize_field_filenames
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

    assert np.allclose(f1.data, f2.data, atol=1e-12)
    assert np.allclose(f1.data, f3.data, atol=1e-12)

    with pytest.raises(AssertionError):
        variable = {"U": "vozocrtx", "nav_lat": "nav_lat"}  # multiple variables will fail
        f3 = Field.from_netcdf(filename, variable, dims)


@pytest.mark.parametrize("with_timestamps", [True, False])
def test_field_from_netcdf(with_timestamps):
    filenames = {
        "lon": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "lat": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "data": str(TEST_DATA / "Uu_eastward_nemo_cross_180lon.nc"),
    }
    variable = "U"
    dimensions = {"lon": "glamf", "lat": "gphif"}
    if with_timestamps:
        timestamp_types = [[[2]], [[np.datetime64("2000-01-01")]]]
        for timestamps in timestamp_types:
            Field.from_netcdf(filenames, variable, dimensions, interp_method="cgrid_velocity", timestamps=timestamps)
    else:
        Field.from_netcdf(filenames, variable, dimensions, interp_method="cgrid_velocity")


@pytest.mark.parametrize(
    "f",
    [
        pytest.param(lambda x: x, id="Path"),
        pytest.param(lambda x: str(x), id="str"),
    ],
)
def test_from_netcdf_path_object(f):
    filenames = {
        "lon": f(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "lat": f(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "data": f(TEST_DATA / "Uu_eastward_nemo_cross_180lon.nc"),
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
        np.random.rand(12, xdim, ydim), coords=[dates, range(xdim), range(ydim)], dims=["time", "lon", "lat"], name="U"
    )
    da.to_netcdf(str(filepath))

    dims = {"lon": "lon", "lat": "lat", "time": "time"}
    try:
        field = Field.from_netcdf(filepath, "U", dims)
    except NotImplementedError:
        field = None

    if field is not None:
        assert field.grid.time_origin.calendar == calendar


@pytest.mark.parametrize(
    "input_,expected",
    [
        pytest.param("file1.nc", ["file1.nc"], id="str"),
        pytest.param(["file1.nc", "file2.nc"], ["file1.nc", "file2.nc"], id="list"),
        pytest.param(["file2.nc", "file1.nc"], ["file1.nc", "file2.nc"], id="list-unsorted"),
        pytest.param([Path("file1.nc"), Path("file2.nc")], ["file1.nc", "file2.nc"], id="list-Path"),
        pytest.param(
            {
                "lon": "lon_file.nc",
                "lat": ["lat_file1.nc", Path("lat_file2.nc")],
                "depth": Path("depth_file.nc"),
                "data": ["data_file1.nc", "data_file2.nc"],
            },
            {
                "lon": ["lon_file.nc"],
                "lat": ["lat_file1.nc", "lat_file2.nc"],
                "depth": ["depth_file.nc"],
                "data": ["data_file1.nc", "data_file2.nc"],
            },
            id="dict-mix",
        ),
    ],
)
def test_sanitize_field_filenames_cases(input_, expected):
    assert _sanitize_field_filenames(input_) == expected


@pytest.mark.parametrize(
    "input_,expected",
    [
        pytest.param("file*.nc", [], id="glob-no-match"),
    ],
)
def test_sanitize_field_filenames_glob(input_, expected):
    assert _sanitize_field_filenames(input_) == expected


@pytest.mark.parametrize(
    "input_,expected",
    [
        pytest.param("test", ["test"], id="str"),
        pytest.param(Path("test"), ["test"], id="Path"),
        pytest.param("file*.nc", [], id="glob-no-match"),
    ],
)
def test_expand_filename(input_, expected):
    assert _expand_filename(input_) == expected
