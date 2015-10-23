import numpy as np
cimport numpy as np
import cython

__all__ = ['Particle', 'ParticleSet']


class ParticleSet(object):
    """Container class for storing and executing over sets of particles.

    Please note that this currently only supports fixed size partcile sets.

    :param size: Initial size of particle set
    :param grid: Grid object from which to sample velocity"""

    def __init__(self, size, grid):
        self._grid = grid
        self._particles = np.empty(size, dtype=Particle)
        self._npart = 0

    def add_particle(self, p):
        p.xi = np.where(p.lon > self._grid.U.lon)[0][-1]
        p.yi = np.where(p.lat > self._grid.U.lat)[0][-1]

        self._particles[self._npart] = p
        self._npart += 1

    def advect(self, timesteps=1, dt=None):
        print "Parcels::ParticleSet: Advecting %d particles for %d timesteps" \
            % (self._npart, timesteps)
        for t in range(timesteps):
            for p in self._particles:
                p.advect_rk4(self._grid, dt)


cdef class Particle(object):
    """Classe encapsualting the basic attributes of a particle"""

    cdef public np.float32_t lon, lat  # Particle position in (lon, lat)
    cdef public np.int32_t xi, yi      # Current indices on the underlying grid

    def __cinit__(self, np.float32_t lon, np.float32_t lat):
        self.lon = lon
        self.lat = lat

    def __repr__(self):
        return "P(%f, %f)[%d, %d]" % (self.lon, self.lat, self.xi, self.yi)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def advect_rk4(self, grid, np.float32_t dt):
        cdef:
            np.float32_t f, u1, v1, u2, v2, u3, v3, u4, v4
            np.float32_t lon1, lat1, lon2, lat2, lon3, lat3
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
