from __future__ import annotations

from datetime import datetime
from typing import TypeVar

import cftime

T = TypeVar("T", datetime, cftime.datetime)


class TimeInterval:
    """A class representing a time interval between two datetime objects.

    Parameters
    ----------
    left : datetime or cftime.datetime
        The left endpoint of the interval.
    right : datetime or cftime.datetime
        The right endpoint of the interval.

    Notes
    -----
    For the purposes of this codebase, the interval can be thought of as closed on the left and right.
    """

    def __init__(self, left: T, right: T) -> None:
        if not isinstance(left, (datetime, cftime.datetime)):
            raise ValueError(f"Expected left to be a datetime or cftime.datetime, got {type(left)}.")
        if not isinstance(right, (datetime, cftime.datetime)):
            raise ValueError(f"Expected right to be a datetime or cftime.datetime, got {type(right)}.")
        if left >= right:
            raise ValueError(f"Expected left to be strictly less than right, got left={left} and right={right}.")
        if not is_compatible(left, right):
            raise ValueError(f"Expected left and right to be compatible, got left={left} and right={right}.")

        self.left = left
        self.right = right

    def __contains__(self, item: T) -> bool:
        return self.left <= item <= self.right

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

        start = max(self.left, other.left)
        end = min(self.right, other.right)

        return TimeInterval(start, end) if start <= end else None


def is_compatible(t1: datetime | cftime.datetime, t2: datetime | cftime.datetime) -> bool:
    """Checks whether two (cftime.)datetime objects are compatible."""
    try:
        t1 - t2
    except Exception:
        return False
    else:
        return True
