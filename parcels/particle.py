__all__ = ['Particle']


class Particle(object):
    """Classe encapsualting the basic attributes of a particle"""

    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def __repr__(self):
        return "P(%f, %f)" % (self.lon, self.lat)

    def advect_rk4(self, grid, dt):
        f = dt / 1000. / 1.852 / 60.
        u1, v1 = grid.eval(self.lon, self.lat)
        lon1, lat1 = (self.lon + u1*.5*f, self.lat + v1*.5*f)
        u2, v2 = grid.eval(lon1, lat1)
        lon2, lat2 = (self.lon + u2*.5*f, self.lat + v2*.5*f)
        u3, v3 = grid.eval(lon2, lat2)
        lon3, lat3 = (self.lon + u3*f, self.lat + v3*f)
        u4, v4 = grid.eval(lon3, lat3)
        self.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f
        self.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f
