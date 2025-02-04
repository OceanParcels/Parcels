import cftime
import numpy as np
import pytest

from parcels.tools.converters import TimeConverter, _get_cftime_datetimes

cf_datetime_classes = [getattr(cftime, c) for c in _get_cftime_datetimes()]
cf_datetime_objects = [c(1990, 1, 1) for c in cf_datetime_classes]


@pytest.mark.parametrize(
    "cf_datetime",
    cf_datetime_objects,
)
def test_TimeConverter_cf(cf_datetime):
    assert TimeConverter(cf_datetime).calendar == cf_datetime.calendar
    assert TimeConverter(cf_datetime).time_origin == cf_datetime


def test_TimeConverter_standard():
    dt = np.datetime64("2001-01-01T12:00")
    assert TimeConverter(dt).calendar == "np_datetime64"
    assert TimeConverter(dt).time_origin == dt

    dt = np.timedelta64(1, "s")
    assert TimeConverter(dt).calendar == "np_timedelta64"
    assert TimeConverter(dt).time_origin == dt

    assert TimeConverter(0).calendar is None
    assert TimeConverter(0).time_origin == 0


def test_TimeConverter_reltime_one_day():
    ONE_DAY = 24 * 60 * 60
    first_jan = [c(1990, 1, 1) for c in cf_datetime_classes] + [0]
    second_jan = [c(1990, 1, 2) for c in cf_datetime_classes] + [ONE_DAY]

    for time_origin, time in zip(first_jan, second_jan, strict=True):
        tc = TimeConverter(time_origin)
        assert tc.reltime(time) == ONE_DAY


def test_TimeConverter_timedelta64_float():
    ONE_DAY = 24 * 60 * 60
    tc = TimeConverter(np.timedelta64(0, "s"))
    assert tc.reltime(1 * ONE_DAY) == 1 * ONE_DAY

    tc = TimeConverter(np.timedelta64(0, "D"))
    assert tc.reltime(1 * ONE_DAY) == 1 * ONE_DAY

    tc = TimeConverter(np.timedelta64(0, "ns"))
    assert tc.reltime(1 * ONE_DAY) == 1 * ONE_DAY


@pytest.mark.parametrize(
    "x, y",
    [
        pytest.param(np.datetime64("2001-01-01T12:00"), 0, id="datetime64 float"),
        pytest.param(cftime.DatetimeNoLeap(1990, 1, 1), 0, id="cftime float"),
        pytest.param(cftime.DatetimeNoLeap(1990, 1, 1), cftime.DatetimeAllLeap(1991, 1, 1), id="cftime cftime"),
    ],
)
def test_TimeConverter_reltime_errors(x, y):
    """All of these should raise a ValueError when doing reltime"""
    tc = TimeConverter(x)
    with pytest.raises((ValueError, TypeError)):
        tc.reltime(y)
