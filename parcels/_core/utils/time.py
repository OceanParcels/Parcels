from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, TypeVar

import cftime
import numpy as np

if TYPE_CHECKING:
    from parcels._typing import TimeLike

T = TypeVar("T", bound="TimeLike")


class TimeInterval:
    """A class representing a time interval between two datetime or np.timedelta64 objects.

    Parameters
    ----------
    left : np.datetime64 or cftime.datetime or np.timedelta64
        The left endpoint of the interval.
    right : np.datetime64 or cftime.datetime or np.timedelta64
        The right endpoint of the interval.

    Notes
    -----
    For the purposes of this codebase, the interval can be thought of as closed on the left and right.
    """

    def __init__(self, left: T, right: T) -> None:
        if not isinstance(left, (np.timedelta64, datetime, cftime.datetime, np.datetime64)):
            raise ValueError(
                f"Expected right to be a np.timedelta64, datetime, cftime.datetime, or np.datetime64. Got {type(left)}."
            )
        if not isinstance(right, (np.timedelta64, datetime, cftime.datetime, np.datetime64)):
            raise ValueError(
                f"Expected right to be a np.timedelta64, datetime, cftime.datetime, or np.datetime64. Got {type(right)}."
            )
        if left >= right:
            raise ValueError(f"Expected left to be strictly less than right, got left={left} and right={right}.")

        if not is_compatible(left, right):
            raise ValueError(f"Expected left and right to be compatible, got left={left} and right={right}.")

        self.left = left
        self.right = right

    def __contains__(self, item: T) -> bool:
        return self.left <= item <= self.right

    def is_all_time_in_interval(self, time):
        item = np.atleast_1d(time)
        return (self.left <= item).all() and (item <= self.right).all()

    def __repr__(self) -> str:
        return f"TimeInterval(left={self.left!r}, right={self.right!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeInterval):
            return False
        return self.left == other.left and self.right == other.right

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def intersection(self, other: TimeInterval) -> TimeInterval | None:
        """Return the intersection of two time intervals. Returns None if there is no overlap."""
        if not is_compatible(self.left, other.left):
            raise ValueError("TimeIntervals are not compatible.")
        if not is_compatible(self.right, other.right):
            raise ValueError("TimeIntervals are not compatible.")

        start = max(self.left, other.left)
        end = min(self.right, other.right)

        return TimeInterval(start, end) if start <= end else None


def is_compatible(
    t1: datetime | cftime.datetime | np.timedelta64, t2: datetime | cftime.datetime | np.timedelta64
) -> bool:
    """
    Defines whether two datetime or np.timedelta64 objects are compatible in the context
    of being left and right sides of an interval.
    """
    # Ensure if either is a timedelta64, both must be
    if isinstance(t1, np.timedelta64) ^ isinstance(t2, np.timedelta64):
        return False

    try:
        t1 - t2
    except Exception:
        return False
    else:
        return True


def get_datetime_type_calendar(
    example_datetime: TimeLike,
) -> tuple[type, str | None]:
    """Get the type and calendar of a datetime object.

    Parameters
    ----------
    example_datetime : datetime, cftime.datetime, or np.datetime64
        The datetime object to check.

    Returns
    -------
    tuple[type, str | None]
        A tuple containing the type of the datetime object and its calendar.
        The calendar will be None if the datetime object is not a cftime datetime object.
    """
    calendar = None
    try:
        calendar = example_datetime.calendar
    except AttributeError:
        # datetime isn't a cftime datetime object
        pass
    return type(example_datetime), calendar


_TD_PRECISION_GETTER_FOR_UNIT = (
    (lambda dt: dt.days, "D"),
    (lambda dt: dt.seconds, "s"),
    (lambda dt: dt.microseconds, "us"),
)


def maybe_convert_python_timedelta_to_numpy(dt: timedelta | np.timedelta64) -> np.timedelta64:
    if isinstance(dt, np.timedelta64):
        return dt

    try:
        dts = []
        for get_value_for_unit, np_unit in _TD_PRECISION_GETTER_FOR_UNIT:
            value = get_value_for_unit(dt)
            if value != 0:
                dts.append(np.timedelta64(value, np_unit))

        if dts:
            return sum(dts)
        else:
            return np.timedelta64(0, "s")
    except Exception as e:
        raise ValueError(f"Could not convert {dt!r} to np.timedelta64.") from e
