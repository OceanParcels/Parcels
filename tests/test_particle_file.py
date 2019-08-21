from parcels.particlefile import _set_calendar
from parcels.tools.converters import _get_cftime_calendars, _get_cftime_datetimes
import cftime


def test_set_calendar():
    for calendar_name, cf_datetime in zip(_get_cftime_calendars(), _get_cftime_datetimes()):
        date = getattr(cftime, cf_datetime)(1990, 1, 1)
        assert _set_calendar(date.calendar) == date.calendar
    assert _set_calendar('np_datetime64') == 'standard'
