from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeVar

from cftime import datetime as cftime_datetime

T = TypeVar("T", datetime, cftime_datetime)


class TimeInterval:
    def __init__(self, left: T, right: T, closed: Literal["right", "left", "both", "neither"] = "left") -> None:
        if not isinstance(left, (datetime, cftime_datetime)):
            raise ValueError(f"Expected left to be a datetime or cftime_datetime, got {type(left)}.")
        if not isinstance(right, (datetime, cftime_datetime)):
            raise ValueError(f"Expected right to be a datetime or cftime_datetime, got {type(right)}.")
        if left >= right:
            raise ValueError(f"Expected left to be strictly less than right, got left={left} and right={right}.")

        if closed not in ["right", "left", "both", "neither"]:
            raise ValueError(f"Invalid closed value: {closed}")

        self.left = left
        self.right = right
        self.closed = closed

    def __contains__(self, item: T) -> bool:
        if self.closed == "left":
            return self.left <= item < self.right
        elif self.closed == "right":
            return self.left < item <= self.right
        elif self.closed == "both":
            return self.left <= item <= self.right
        elif self.closed == "neither":
            return self.left < item < self.right
        else:
            raise ValueError(f"Invalid closed value: {self.closed}")

    def __repr__(self) -> str:
        return f"TimeInterval(left={self.left!r}, right={self.right!r}, closed={self.closed!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeInterval):
            return False
        return self.left == other.left and self.right == other.right and self.closed == other.closed

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def intersection(self, other: TimeInterval) -> TimeInterval: ...
