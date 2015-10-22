from scipy.interpolate import RectBivariateSpline
from cached_property import cached_property


__all__ = ['Field']


class Field(object):
    """Class that encapsulates access to field data.

    :param name: Name of the field
    :param data: 2D array of field data
    :param lon: Longitude coordinates of the field
    :param lat: Latitude coordinates of the field
    """

    def __init__(self, name, data, lon, lat):
        self.name = name
        self.data = data
        self.lon = lon
        self.lat = lat

    @cached_property
    def interpolator(self):
        return RectBivariateSpline(self.lat, self.lon, self.data)

    def eval(self, x, y):
        return self.interpolator.ev(x, y)
