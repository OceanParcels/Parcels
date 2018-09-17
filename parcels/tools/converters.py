from math import cos, pi
import numpy as np
import cftime

__all__ = ['UnitConverter', 'Geographic', 'GeographicPolar', 'GeographicSquare',
           'GeographicPolarSquare', 'unitconverters_map',
           'TimeConverter']


class TimeConverter(object):
    """ Intrerface class for converting of TimeOrigins that performs no conversion
    """

    def __init__(self, time_origin=0):
        self.time_origin = time_origin
        if isinstance(time_origin, np.datetime64):
            self.calendar = "standard"
        elif isinstance(time_origin, cftime._cftime.DatetimeNoLeap):
            self.calendar = "NOLEAP"
        else:
            self.calendar = None

    def reltime(self, time):
        time = time.time_origin if isinstance(time, TimeConverter) else time
        if self.calendar == 'standard':
            return (time - self.time_origin) / np.timedelta64(1, 's')
        elif self.calendar == 'NOLEAP':
            return np.array([(t - self.time_origin).total_seconds() for t in time])
        elif self.calendar is None:
            return time - self.time_origin
        else:
            raise RuntimeError('Calendar %s not implemented in TimeConverter' % (self.calendar))

    def fulltime(self, time):
        time = time.time_origin if isinstance(time, TimeConverter) else time
        if self.calendar == 'standard':
            return self.time_origin + np.timedelta64(int(time), 's')
        # elif self.type == 'NOLEAP':

        elif self.calendar is None:
            return time + self.time_origin
        else:
            raise RuntimeError('Calendar %s not implemented in TimeConverter' % (self.calendar))

    def __repr__(self):
        return "%s" % self.time_origin

    def __cmp__(self, other):
        other = other.time_origin if isinstance(other, TimeConverter) else other
        return cmp(self.time_origin, other)


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
