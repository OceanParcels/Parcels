from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from cftime import datetime as cftime_datetime
from hypothesis import given
from hypothesis import strategies as st

from parcels._core.utils.time import TimeInterval

calendar_strategy = st.sampled_from(["gregorian", "proleptic_gregorian", "365_day", "360_day", "julian", "366_day"])
closed_strategy = st.sampled_from(["right", "left", "both", "neither"])


@st.composite
def cftime_datetime_strategy(draw, calendar=None):
    year = draw(st.integers(1900, 2100))
    month = draw(st.integers(1, 12))
    day = draw(st.integers(1, 28))
    if calendar is None:
        calendar = draw(calendar_strategy)
    return cftime_datetime(year, month, day, calendar=calendar)


@st.composite
def cftime_interval_strategy(draw, left=None):
    if left is None:
        left = draw(cftime_datetime_strategy())
    right = left + draw(
        st.timedeltas(
            min_value=timedelta(seconds=1),
            max_value=timedelta(days=100 * 365),
        )
    )
    closed = draw(closed_strategy)
    return TimeInterval(left, right, closed)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
@pytest.mark.parametrize(
    "left,right",
    [
        (cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 2, calendar="gregorian")),
        (cftime_datetime(2023, 6, 1, calendar="365_day"), cftime_datetime(2023, 6, 2, calendar="365_day")),
        (cftime_datetime(2023, 12, 1, calendar="360_day"), cftime_datetime(2023, 12, 2, calendar="360_day")),
    ],
)
def test_time_interval_initialization(left, right, closed):
    """Test that TimeInterval can be initialized with valid inputs."""
    interval = TimeInterval(left, right, closed)
    assert interval.left == left
    assert interval.right == right
    assert interval.closed == closed

    with pytest.raises(ValueError):
        TimeInterval(right, left, closed)


def test_time_interval_invalid_closed():
    """Test that TimeInterval raises ValueError for invalid closed values."""
    left = datetime(2023, 1, 1)
    right = datetime(2023, 1, 2)
    with pytest.raises(ValueError):
        TimeInterval(left, right, closed="invalid")


@given(cftime_interval_strategy())
def test_time_interval_contains(interval):
    left = interval.left
    right = interval.right
    middle = left + (right - left) / 2

    if interval.closed in ["left", "both"]:
        assert left in interval
    if interval.closed in ["right", "both"]:
        assert right in interval

    assert middle in interval


def test_time_interval_repr():
    """Test the string representation of TimeInterval."""
    interval = TimeInterval(datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 2, 12, 0), "both")
    expected = "TimeInterval(left=datetime.datetime(2023, 1, 1, 12, 0), right=datetime.datetime(2023, 1, 2, 12, 0), closed='both')"
    assert repr(interval) == expected


@given(cftime_interval_strategy())
def test_time_interval_equality(interval):
    assert interval == interval
