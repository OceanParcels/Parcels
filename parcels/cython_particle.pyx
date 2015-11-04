import numpy as np
cimport numpy as np
import cython
from parcels.particle import ParticleSet
from parcels.jit_module import Kernel, GNUCompiler

__all__ = ['CythonParticle', 'CythonParticleSet']


class CythonParticleSet(ParticleSet):
    """Container class for storing and executing over sets of particles.

    Please note that this currently only supports fixed size partcile sets.

    :param size: Initial size of particle set
    :param grid: Grid object from which to sample velocity"""

    def advect(self, timesteps=1, dt=None):
        cdef:
            np.int32_t t, tsteps = timesteps
            np.ndarray[np.float32_t, ndim=1, mode="c"] lon_u = self._grid.U.lon,
            np.ndarray[np.float32_t, ndim=1, mode="c"] lat_u = self._grid.U.lat,
            np.ndarray[np.float32_t, ndim=2, mode="c"] U = self._grid.U.data,
            np.ndarray[np.float32_t, ndim=1, mode="c"] lon_v = self._grid.V.lon,
            np.ndarray[np.float32_t, ndim=1, mode="c"] lat_v = self._grid.V.lat,
            np.ndarray[np.float32_t, ndim=2, mode="c"] V = self._grid.V.data
        print "Parcels::CythonParticleSet: Advecting %d particles for %d timesteps" \
            % (self._npart, timesteps)

        for t in range(tsteps):
            for p in self._particles:
                advect_rk4_cython(p, dt, lon_u, lat_u, U, lon_v, lat_v, V)


cdef class CythonParticle(object):
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef advect_rk4_cython(CythonParticle p, np.float32_t dt,
                       np.ndarray[np.float32_t, ndim=1, mode="c"] lon_u,
                       np.ndarray[np.float32_t, ndim=1, mode="c"] lat_u,
                       np.ndarray[np.float32_t, ndim=2, mode="c"] U,
                       np.ndarray[np.float32_t, ndim=1, mode="c"] lon_v,
                       np.ndarray[np.float32_t, ndim=1, mode="c"] lat_v,
                       np.ndarray[np.float32_t, ndim=2, mode="c"] V):
    cdef:
        np.float32_t f, u1, v1, u2, v2, u3, v3, u4, v4
        np.float32_t lon1, lat1, lon2, lat2, lon3, lat3
    f = dt / 1000. / 1.852 / 60.
    u1 = interpolate_bilinear(p.lat, p.lon, p.yi, p.xi, lat_u, lon_u, U)
    v1 = interpolate_bilinear(p.lat, p.lon, p.yi, p.xi, lat_v, lon_v, V)
    lon1, lat1 = (p.lon + u1*.5*f, p.lat + v1*.5*f)
    u2 = interpolate_bilinear(lat1, lon1, p.yi, p.xi, lat_u, lon_u, U)
    v2 = interpolate_bilinear(lat1, lon1, p.yi, p.xi, lat_v, lon_v, V)
    lon2, lat2 = (p.lon + u2*.5*f, p.lat + v2*.5*f)
    u3 = interpolate_bilinear(lat2, lon2, p.yi, p.xi, lat_u, lon_u, U)
    v3 = interpolate_bilinear(lat2, lon2, p.yi, p.xi, lat_v, lon_v, V)
    lon3, lat3 = (p.lon + u3*f, p.lat + v3*f)
    u4 = interpolate_bilinear(lat3, lon3, p.yi, p.xi, lat_u, lon_u, U)
    v4 = interpolate_bilinear(lat3, lon3, p.yi, p.xi, lat_v, lon_v, V)

    # Advance particle position in space and on the grid
    p.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f
    p.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f
    p.xi = advance_index(p.lon, p.xi, lon_u)
    p.yi = advance_index(p.lat, p.yi, lat_u)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int32_t advance_index(np.float32_t x, np.int32_t i,
                              np.ndarray[np.float32_t, ndim=1, mode="c"] xvals) except? -1:
    while x > xvals[i+1]:
        i += 1
    while x < xvals[i]:
        i -= 1
    return i


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.float32_t interpolate_bilinear(np.float32_t x, np.float32_t y,
                                       np.int32_t xi, np.int32_t yi,
                                       np.ndarray[np.float32_t, ndim=1, mode="c"] xvals,
                                       np.ndarray[np.float32_t, ndim=1, mode="c"] yvals,
                                       np.ndarray[np.float32_t, ndim=2, mode="c"] qvals):
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
