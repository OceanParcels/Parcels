import numpy as np
cimport numpy as np
import cython
from parcels.jit_module import Kernel, GNUCompiler

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

        # Particle array for JIT kernel
        self._kernel = None
        self._p_dtype = np.dtype([('lon', np.float32), ('lat', np.float32),
                                  ('xi', np.int32), ('yi', np.int32)])
        self._p_array = np.empty(size, dtype=self._p_dtype)

    def add_particle(self, p):
        p.xi = np.where(p.lon > self._grid.U.lon)[0][-1]
        p.yi = np.where(p.lat > self._grid.U.lat)[0][-1]
        self._particles[self._npart] = p

        # Populate the partcile's struct
        self._p_array[self._npart]['lon'] = p.lon
        self._p_array[self._npart]['lat'] = p.lat
        self._p_array[self._npart]['xi'] = p.xi
        self._p_array[self._npart]['yi'] = p.yi

        self._npart += 1

    def generate_jit_kernel(self, filename):
        self._kernel = Kernel(filename)
        self._kernel.generate_code()
        self._kernel.compile(compiler=GNUCompiler())
        self._kernel.load_lib()

    def advect(self, timesteps=1, dt=None):
        print "Parcels::ParticleSet: Advecting %d particles for %d timesteps" \
            % (self._npart, timesteps)
        for t in range(timesteps):
            for p in self._particles:
                p.advect_rk4(self._grid, dt)

    def advect_cython(self, timesteps=1, dt=None):
        print "Parcels::ParticleSet: Advecting %d particles for %d timesteps" \
            % (self._npart, timesteps)
        for t in range(timesteps):
            for p in self._particles:
                p.advect_rk4_cython(self._grid, dt)

        self._kernel.execute(self)


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


    def advect_rk4_cython(self, grid, np.float32_t dt):
        cdef:
            np.float32_t f, u1, v1, u2, v2, u3, v3, u4, v4
            np.float32_t lon1, lat1, lon2, lat2, lon3, lat3
        f = dt / 1000. / 1.852 / 60.
        u1 = interpolate_bilinear(self.lat, self.lon, self.yi, self.xi,
                                  grid.U.lat, grid.U.lon, grid.U.data)
        v1 = interpolate_bilinear(self.lat, self.lon, self.yi, self.xi,
                                  grid.V.lat, grid.V.lon, grid.V.data)
        lon1, lat1 = (self.lon + u1*.5*f, self.lat + v1*.5*f)
        u2 = interpolate_bilinear(lat1, lon1, self.yi, self.xi,
                                  grid.U.lat, grid.U.lon, grid.U.data)
        v2 = interpolate_bilinear(lat1, lon1, self.yi, self.xi,
                                  grid.V.lat, grid.V.lon, grid.V.data)
        lon2, lat2 = (self.lon + u2*.5*f, self.lat + v2*.5*f)
        u3 = interpolate_bilinear(lat2, lon2, self.yi, self.xi,
                                  grid.U.lat, grid.U.lon, grid.U.data)
        v3 = interpolate_bilinear(lat2, lon2, self.yi, self.xi,
                                  grid.V.lat, grid.V.lon, grid.V.data)
        lon3, lat3 = (self.lon + u3*f, self.lat + v3*f)
        u4 = interpolate_bilinear(lat3, lon3, self.yi, self.xi,
                                  grid.U.lat, grid.U.lon, grid.U.data)
        v4 = interpolate_bilinear(lat3, lon3, self.yi, self.xi,
                                  grid.V.lat, grid.V.lon, grid.V.data)

        # Advance particle position in space and on the grid
        self.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f
        self.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f
        self.xi = advance_index(self.lon, self.xi, grid.U.lon)
        self.yi = advance_index(self.lat, self.yi, grid.U.lat)

cdef np.int32_t advance_index(np.float32_t x, np.int32_t i,
                              np.ndarray[np.float32_t, ndim=1, mode="c"] xvals) except? -1:
    while i < xvals.size-1 and x > xvals[i+1]:
        i += 1
    while i > 0 and x < xvals[i]:
        i -= 1
    return i

cdef np.float32_t interpolate_bilinear(np.float32_t x, np.float32_t y,
                                       np.int32_t xi, np.int32_t yi,
                                       np.ndarray[np.float32_t, ndim=1, mode="c"] xvals,
                                       np.ndarray[np.float32_t, ndim=1, mode="c"] yvals,
                                       np.ndarray[np.float32_t, ndim=2, mode="c"] qvals) except? -1:
    """Bilinear interpolation function

    Computes f(x, y), given f(x0, y0), f(x0, y1), f(x1, y0) and f(x1, y1)
    where x0 <= x <= x1 and y0 <= y <= y1

    See https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    cdef np.int32_t i = xi, j = yi
    i = advance_index(x, i, xvals)
    j = advance_index(y, j, yvals)
    return (qvals[i, j] * (xvals[i+1] - x) * (yvals[j+1] - y)
        + qvals[i+1, j] * (x - xvals[i]) * (yvals[j+1] - y)
        + qvals[i, j+1] * (xvals[i+1] - x) * (y - yvals[j])
        + qvals[i+1, j+1] * (x - xvals[i]) * (y - yvals[j])) / ((xvals[i+1] - xvals[i]) * (yvals[j+1] - yvals[j]))
