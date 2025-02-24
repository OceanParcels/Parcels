from __future__ import annotations

import inspect
from math import cos, pi

import cftime
import numpy as np
import numpy.typing as npt
import xarray as xr

__all__ = [
    "Geographic",
    "GeographicPolar",
    "GeographicPolarSquare",
    "GeographicSquare",
    "UnitConverter",
    "convert_to_flat_array",
    "convert_xarray_time_units",
    "unitconverters_map",
]


def convert_to_flat_array(var: npt.ArrayLike) -> npt.NDArray:
    """Convert lists and single integers/floats to one-dimensional numpy arrays

    Parameters
    ----------
    var : Array
        list or numeric to convert to a one-dimensional numpy array
    """
    return np.array(var).flatten()


def _get_cftime_datetimes() -> list[str]:
    # Is there a more elegant way to parse these from cftime?
    cftime_calendars = tuple(x[1].__name__ for x in inspect.getmembers(cftime._cftime, inspect.isclass))
    cftime_datetime_names = [ca for ca in cftime_calendars if "Datetime" in ca]
    return cftime_datetime_names


def _get_cftime_calendars() -> list[str]:  # TODO v4: check if function used?
    return [getattr(cftime, cf_datetime)(1990, 1, 1).calendar for cf_datetime in _get_cftime_datetimes()]


class UnitConverter:
    """Interface class for spatial unit conversion during field sampling that performs no conversion."""

    source_unit: str | None = None
    target_unit: str | None = None

    def to_target(self, value, z, y, x):
        return value

    def to_source(self, value, z, y, x):
        return value


class Geographic(UnitConverter):
    """Unit converter from geometric to geographic coordinates (m to degree)"""

    source_unit = "m"
    target_unit = "degree"

    def to_target(self, value, z, y, x):
        return value / 1000.0 / 1.852 / 60.0

    def to_source(self, value, z, y, x):
        return value * 1000.0 * 1.852 * 60.0


class GeographicPolar(UnitConverter):
    """Unit converter from geometric to geographic coordinates (m to degree)
    with a correction to account for narrower grid cells closer to the poles.
    """

    source_unit = "m"
    target_unit = "degree"

    def to_target(self, value, z, y, x):
        return value / 1000.0 / 1.852 / 60.0 / cos(y * pi / 180)

    def to_source(self, value, z, y, x):
        return value * 1000.0 * 1.852 * 60.0 * cos(y * pi / 180)


class GeographicSquare(UnitConverter):
    """Square distance converter from geometric to geographic coordinates (m2 to degree2)"""

    source_unit = "m2"
    target_unit = "degree2"

    def to_target(self, value, z, y, x):
        return value / pow(1000.0 * 1.852 * 60.0, 2)

    def to_source(self, value, z, y, x):
        return value * pow(1000.0 * 1.852 * 60.0, 2)


class GeographicPolarSquare(UnitConverter):
    """Square distance converter from geometric to geographic coordinates (m2 to degree2)
    with a correction to account for narrower grid cells closer to the poles.
    """

    source_unit = "m2"
    target_unit = "degree2"

    def to_target(self, value, z, y, x):
        return value / pow(1000.0 * 1.852 * 60.0 * cos(y * pi / 180), 2)

    def to_source(self, value, z, y, x):
        return value * pow(1000.0 * 1.852 * 60.0 * cos(y * pi / 180), 2)


unitconverters_map = {
    "U": GeographicPolar(),
    "V": Geographic(),
    "Kh_zonal": GeographicPolarSquare(),
    "Kh_meridional": GeographicSquare(),
}


def convert_xarray_time_units(ds, time):
    """Fixes DataArrays that have time.Unit instead of expected time.units"""
    da = ds[time] if isinstance(ds, xr.Dataset) else ds
    if "units" not in da.attrs and "Unit" in da.attrs:
        da.attrs["units"] = da.attrs["Unit"]
    da2 = xr.Dataset({time: da})
    try:
        da2 = xr.decode_cf(da2)
    except ValueError:
        raise RuntimeError(
            "Xarray could not convert the calendar. If you're using from_netcdf, "
            "try using the timestamps keyword in the construction of your Field. "
            "See also the tutorial at https://docs.oceanparcels.org/en/latest/examples/tutorial_timestamps.html"
        )
    ds[time] = da2[time]
