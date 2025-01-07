from __future__ import annotations

import inspect
from datetime import timedelta
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
    "TimeConverter",
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


def _get_cftime_calendars() -> list[str]:
    return [getattr(cftime, cf_datetime)(1990, 1, 1).calendar for cf_datetime in _get_cftime_datetimes()]


class TimeConverter:
    """Converter class for dates with different calendars in FieldSets

    Parameters
    ----------
    time_origin : float, integer, numpy.datetime64 or cftime.DatetimeNoLeap
        time origin of the class.
    """

    def __init__(self, time_origin: float | np.datetime64 | np.timedelta64 | cftime.datetime = 0):
        self.time_origin = time_origin
        self.calendar: str | None = None
        if isinstance(time_origin, np.datetime64):
            self.calendar = "np_datetime64"
        elif isinstance(time_origin, np.timedelta64):
            self.calendar = "np_timedelta64"
        elif isinstance(time_origin, cftime.datetime):
            self.calendar = time_origin.calendar

    def reltime(self, time: TimeConverter | np.datetime64 | np.timedelta64 | cftime.datetime) -> float | npt.NDArray:
        """Method to compute the difference, in seconds, between a time and the time_origin
        of the TimeConverter

        Parameters
        ----------
        time :


        Returns
        -------
        type
            time - self.time_origin

        """
        time = time.time_origin if isinstance(time, TimeConverter) else time
        if self.calendar in ["np_datetime64", "np_timedelta64"]:
            return (time - self.time_origin) / np.timedelta64(1, "s")  # type: ignore
        elif self.calendar in _get_cftime_calendars():
            if isinstance(time, (list, np.ndarray)):
                try:
                    return np.array([(t - self.time_origin).total_seconds() for t in time])  # type: ignore
                except ValueError:
                    raise ValueError(
                        f"Cannot subtract 'time' (a {type(time)} object) from a {self.calendar} calendar.\n"
                        f"Provide 'time' as a {type(self.time_origin)} object?"
                    )
            else:
                try:
                    return (time - self.time_origin).total_seconds()  # type: ignore
                except ValueError:
                    raise ValueError(
                        f"Cannot subtract 'time' (a {type(time)} object) from a {self.calendar} calendar.\n"
                        f"Provide 'time' as a {type(self.time_origin)} object?"
                    )
        elif self.calendar is None:
            return time - self.time_origin  # type: ignore
        else:
            raise RuntimeError(f"Calendar {self.calendar} not implemented in TimeConverter")

    def fulltime(self, time):
        """Method to convert a time difference in seconds to a date, based on the time_origin

        Parameters
        ----------
        time :


        Returns
        -------
        type
            self.time_origin + time

        """
        time = time.time_origin if isinstance(time, TimeConverter) else time
        if self.calendar in ["np_datetime64", "np_timedelta64"]:
            if isinstance(time, (list, np.ndarray)):
                return [self.time_origin + np.timedelta64(int(t), "s") for t in time]
            else:
                return self.time_origin + np.timedelta64(int(time), "s")
        elif self.calendar in _get_cftime_calendars():
            return self.time_origin + timedelta(seconds=time)
        elif self.calendar is None:
            return self.time_origin + time
        else:
            raise RuntimeError(f"Calendar {self.calendar} not implemented in TimeConverter")

    def __repr__(self):
        return f"{self.time_origin}"

    def __eq__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        return self.time_origin == other

    def __ne__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        if not isinstance(other, type(self.time_origin)):
            return True
        return self.time_origin != other

    def __gt__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        return self.time_origin > other

    def __lt__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        return self.time_origin < other

    def __ge__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        return self.time_origin >= other

    def __le__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        return self.time_origin <= other


class UnitConverter:
    """Interface class for spatial unit conversion during field sampling that performs no conversion."""

    source_unit: str | None = None
    target_unit: str | None = None

    def to_target(self, value, z, y, x):
        return value

    def ccode_to_target(self, z, y, x):
        return "1.0"

    def to_source(self, value, z, y, x):
        return value

    def ccode_to_source(self, z, y, x):
        return "1.0"


class Geographic(UnitConverter):
    """Unit converter from geometric to geographic coordinates (m to degree)"""

    source_unit = "m"
    target_unit = "degree"

    def to_target(self, value, z, y, x):
        return value / 1000.0 / 1.852 / 60.0

    def to_source(self, value, z, y, x):
        return value * 1000.0 * 1.852 * 60.0

    def ccode_to_target(self, z, y, x):
        return "(1.0 / (1000.0 * 1.852 * 60.0))"

    def ccode_to_source(self, z, y, x):
        return "(1000.0 * 1.852 * 60.0)"


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

    def ccode_to_target(self, z, y, x):
        return f"(1.0 / (1000. * 1.852 * 60. * cos({y} * M_PI / 180)))"

    def ccode_to_source(self, z, y, x):
        return f"(1000. * 1.852 * 60. * cos({y} * M_PI / 180))"


class GeographicSquare(UnitConverter):
    """Square distance converter from geometric to geographic coordinates (m2 to degree2)"""

    source_unit = "m2"
    target_unit = "degree2"

    def to_target(self, value, z, y, x):
        return value / pow(1000.0 * 1.852 * 60.0, 2)

    def to_source(self, value, z, y, x):
        return value * pow(1000.0 * 1.852 * 60.0, 2)

    def ccode_to_target(self, z, y, x):
        return "pow(1.0 / (1000.0 * 1.852 * 60.0), 2)"

    def ccode_to_source(self, z, y, x):
        return "pow((1000.0 * 1.852 * 60.0), 2)"


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

    def ccode_to_target(self, z, y, x):
        return f"pow(1.0 / (1000. * 1.852 * 60. * cos({y} * M_PI / 180)), 2)"

    def ccode_to_source(self, z, y, x):
        return f"pow((1000. * 1.852 * 60. * cos({y} * M_PI / 180)), 2)"


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
