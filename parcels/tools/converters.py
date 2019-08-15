from math import cos, pi
import numpy as np
import cftime
import inspect
from datetime import timedelta as delta

__all__ = ['UnitConverter', 'Geographic', 'GeographicPolar', 'GeographicSquare',
           'GeographicPolarSquare', 'unitconverters_map',
           'TimeConverter']


def _get_cftime_datetimes():
    # Is there a more elegant way to parse these from cftime?
    cftime_calendars = tuple(x[1].__name__ for x in inspect.getmembers(cftime._cftime, inspect.isclass))
    cftime_datetime_names = [ca for ca in cftime_calendars if 'Datetime' in ca]
    return cftime_datetime_names


def _get_cftime_calendars():
    return [getattr(cftime, cf_datetime)(1990, 1, 1).calendar for cf_datetime in _get_cftime_datetimes()]


class TimeConverter(object):
    """ Converter class for dates with different calendars in FieldSets

    :param: time_origin: time origin of the class. Currently supported formats are
            float, integer, numpy.datetime64, and netcdftime.DatetimeNoLeap
    """

    def __init__(self, time_origin=0):
        self.time_origin = 0 if time_origin is None else time_origin
        if isinstance(time_origin, np.datetime64):
            self.calendar = "np_datetime64"
        elif isinstance(time_origin, cftime._cftime.datetime):
            self.calendar = time_origin.calendar
        else:
            self.calendar = None

    def reltime(self, time):
        """Method to compute the difference, in seconds, between a time and the time_origin
        of the TimeConverter

        :param: time: input time
        :return: time - self.time_origin
        """
        time = time.time_origin if isinstance(time, TimeConverter) else time
        if self.calendar == 'np_datetime64':
            return (time - self.time_origin) / np.timedelta64(1, 's')
        elif self.calendar in _get_cftime_calendars():
            if isinstance(time, (list, np.ndarray)):
                return np.array([(t - self.time_origin).total_seconds() for t in time])
            else:
                return (time - self.time_origin).total_seconds()
        elif self.calendar is None:
            return time - self.time_origin
        else:
            raise RuntimeError('Calendar %s not implemented in TimeConverter' % (self.calendar))

    def fulltime(self, time):
        """Method to convert a time difference in seconds to a date, based on the time_origin

        :param: time: input time
        :return: self.time_origin + time
        """
        time = time.time_origin if isinstance(time, TimeConverter) else time
        if self.calendar == 'np_datetime64':
            if isinstance(time, (list, np.ndarray)):
                return [self.time_origin + np.timedelta64(int(t), 's') for t in time]
            else:
                return self.time_origin + np.timedelta64(int(time), 's')
        elif self.calendar in _get_cftime_calendars():
            return self.time_origin + delta(seconds=time)
        elif self.calendar is None:
            return self.time_origin + time
        else:
            raise RuntimeError('Calendar %s not implemented in TimeConverter' % (self.calendar))

    def __repr__(self):
        return "%s" % self.time_origin

    def __eq__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        return self.time_origin == other

    def __ne__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
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


class UnitConverter(object):
    """ Interface class for spatial unit conversion during field sampling
        that performs no conversion.
    """
    source_unit = None
    target_unit = None

    def to_target(self, value, x, y, z):
        return value

    def ccode_to_target(self, x, y, z):
        return "1.0"

    def to_source(self, value, x, y, z):
        return value

    def ccode_to_source(self, x, y, z):
        return "1.0"


class Geographic(UnitConverter):
    """ Unit converter from geometric to geographic coordinates (m to degree) """
    source_unit = 'm'
    target_unit = 'degree'

    def to_target(self, value, x, y, z):
        return value / 1000. / 1.852 / 60.

    def to_source(self, value, x, y, z):
        return value * 1000. * 1.852 * 60.

    def ccode_to_target(self, x, y, z):
        return "(1.0 / (1000.0 * 1.852 * 60.0))"

    def ccode_to_source(self, x, y, z):
        return "(1000.0 * 1.852 * 60.0)"


class GeographicPolar(UnitConverter):
    """ Unit converter from geometric to geographic coordinates (m to degree)
        with a correction to account for narrower grid cells closer to the poles.
    """
    source_unit = 'm'
    target_unit = 'degree'

    def to_target(self, value, x, y, z):
        return value / 1000. / 1.852 / 60. / cos(y * pi / 180)

    def to_source(self, value, x, y, z):
        return value * 1000. * 1.852 * 60. * cos(y * pi / 180)

    def ccode_to_target(self, x, y, z):
        return "(1.0 / (1000. * 1.852 * 60. * cos(%s * M_PI / 180)))" % y

    def ccode_to_source(self, x, y, z):
        return "(1000. * 1.852 * 60. * cos(%s * M_PI / 180))" % y


class GeographicSquare(UnitConverter):
    """ Square distance converter from geometric to geographic coordinates (m2 to degree2) """
    source_unit = 'm2'
    target_unit = 'degree2'

    def to_target(self, value, x, y, z):
        return value / pow(1000. * 1.852 * 60., 2)

    def to_source(self, value, x, y, z):
        return value * pow(1000. * 1.852 * 60., 2)

    def ccode_to_target(self, x, y, z):
        return "pow(1.0 / (1000.0 * 1.852 * 60.0), 2)"

    def ccode_to_source(self, x, y, z):
        return "pow((1000.0 * 1.852 * 60.0), 2)"


class GeographicPolarSquare(UnitConverter):
    """ Square distance converter from geometric to geographic coordinates (m2 to degree2)
        with a correction to account for narrower grid cells closer to the poles.
    """
    source_unit = 'm2'
    target_unit = 'degree2'

    def to_target(self, value, x, y, z):
        return value / pow(1000. * 1.852 * 60. * cos(y * pi / 180), 2)

    def to_source(self, value, x, y, z):
        return value * pow(1000. * 1.852 * 60. * cos(y * pi / 180), 2)

    def ccode_to_target(self, x, y, z):
        return "pow(1.0 / (1000. * 1.852 * 60. * cos(%s * M_PI / 180)), 2)" % y

    def ccode_to_source(self, x, y, z):
        return "pow((1000. * 1.852 * 60. * cos(%s * M_PI / 180)), 2)" % y


unitconverters_map = {'U': GeographicPolar(), 'V': Geographic(),
                      'Kh_zonal': GeographicPolarSquare(),
                      'Kh_meridional': GeographicSquare()}
