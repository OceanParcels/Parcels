from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from cftime import datetime as cftime_datetime
from hypothesis import given
from hypothesis import strategies as st

from parcels._core.utils.time import TimeInterval, maybe_convert_python_timedelta_to_numpy

calendar_strategy = st.sampled_from(
    [
        "gregorian",
        "proleptic_gregorian",
        "365_day",
        "360_day",
        "julian",
        "366_day",
        np.datetime64,
        datetime,
        np.timedelta64,
    ]
)


@st.composite
def np_timedelta64_strategy(draw):
    """Strategy for generating np.timedelta64 objects."""
    return np.timedelta64(draw(st.integers(1, 60 * 60 * 24 * 100 * 365)), "s")


@st.composite
def datetime_strategy(draw, calendar=None):
    if calendar is None:
        calendar = draw(calendar_strategy)
    if calendar is np.timedelta64:
        return draw(np_timedelta64_strategy())

    year = draw(st.integers(1900, 2100))
    month = draw(st.integers(1, 12))
    day = draw(st.integers(1, 28))
    if calendar is datetime:
        return datetime(year, month, day)
    if calendar is np.datetime64:
        return np.datetime64(datetime(year, month, day))

    return cftime_datetime(year, month, day, calendar=calendar)


@st.composite
def time_interval_strategy(draw, left=None, calendar=None):
    if left is None:
        left = draw(datetime_strategy(calendar=calendar))
    right = left + draw(np_timedelta64_strategy())

    return TimeInterval(left, right)


@pytest.mark.parametrize(
    "left,right",
    [
        (cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 2, calendar="gregorian")),
        (cftime_datetime(2023, 6, 1, calendar="365_day"), cftime_datetime(2023, 6, 2, calendar="365_day")),
        (cftime_datetime(2023, 12, 1, calendar="360_day"), cftime_datetime(2023, 12, 2, calendar="360_day")),
        (datetime(2023, 12, 1), datetime(2023, 12, 2)),
        (np.datetime64(datetime(2023, 12, 1)), np.datetime64(datetime(2023, 12, 2))),
    ],
)
def test_time_interval_initialization(left, right):
    """Test that TimeInterval can be initialized with valid inputs."""
    interval = TimeInterval(left, right)
    assert interval.left == left
    assert interval.right == right

    with pytest.raises(ValueError):
        TimeInterval(right, left)


@given(time_interval_strategy())
def test_time_interval_contains(interval):
    left = interval.left
    right = interval.right
    middle = left + (right - left) / 2

    assert interval.is_all_time_in_interval(left)
    assert interval.is_all_time_in_interval(right)
    assert interval.is_all_time_in_interval(middle)


@given(time_interval_strategy(calendar="365_day"), time_interval_strategy(calendar="365_day"))
def test_time_interval_intersection_commutative(interval1, interval2):
    assert interval1.intersection(interval2) == interval2.intersection(interval1)


@given(time_interval_strategy())
def test_time_interval_intersection_with_self(interval):
    assert interval.intersection(interval) == interval


def test_time_interval_repr():
    """Test the string representation of TimeInterval."""
    interval = TimeInterval(datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 2, 12, 0))
    expected = "TimeInterval(left=datetime.datetime(2023, 1, 1, 12, 0), right=datetime.datetime(2023, 1, 2, 12, 0))"
    assert repr(interval) == expected


@given(time_interval_strategy())
def test_time_interval_equality(interval):
    assert interval == interval


@pytest.mark.parametrize(
    "interval1,interval2,expected",
    [
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 4, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            id="overlapping intervals",
        ),
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 5, calendar="gregorian"), cftime_datetime(2023, 1, 6, calendar="gregorian")
            ),
            None,
            id="non-overlapping intervals",
        ),
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 2, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 2, calendar="gregorian")
            ),
            id="intervals with same start time",
        ),
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            id="intervals with same end time",
        ),
    ],
)
def test_time_interval_intersection(interval1, interval2, expected):
    """Test the intersection of two time intervals."""
    result = interval1.intersection(interval2)
    if expected is None:
        assert result is None
    else:
        assert result.left == expected.left
        assert result.right == expected.right


def test_time_interval_intersection_different_calendars():
    interval1 = TimeInterval(
        cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
    )
    interval2 = TimeInterval(
        cftime_datetime(2023, 1, 1, calendar="365_day"), cftime_datetime(2023, 1, 3, calendar="365_day")
    )
    with pytest.raises(ValueError, match="TimeIntervals are not compatible."):
        interval1.intersection(interval2)


@pytest.mark.parametrize(
    "td,expected",
    [
        pytest.param(np.timedelta64(1, "s"), np.timedelta64(1, "s"), id="noop"),
        pytest.param(timedelta(days=5), np.timedelta64(5, "D"), id="single unit"),
        pytest.param(timedelta(days=5, seconds=30), np.timedelta64(5, "D") + np.timedelta64(30, "s"), id="mixed units"),
        pytest.param(timedelta(days=0), np.timedelta64(0, "s"), id="zero timedelta"),
        pytest.param(
            timedelta(seconds=-2), np.timedelta64(-2, "s"), id="negative timedelta"
        ),  # included because timedelta(seconds=-2) -> timedelta(days=-1, seconds=86398)
    ],
)
def test_maybe_convert_python_timedelta_to_numpy(td, expected):
    result = maybe_convert_python_timedelta_to_numpy(td)
    assert result == expected
