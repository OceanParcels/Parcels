from parcels.tools.converters import TimeConverter, _get_cftime_datetimes
import cftime
import numpy as np


def test_TimeConverter():
    cf_datetime_names = _get_cftime_datetimes()
    for cf_datetime in cf_datetime_names:
        date = getattr(cftime, cf_datetime)(1990, 1, 1)
        assert TimeConverter(date).calendar == date.calendar
    assert TimeConverter(None).calendar is None
    date_datetime64 = np.datetime64('2001-01-01T12:00')
    assert TimeConverter(date_datetime64).calendar == "np_datetime64"
