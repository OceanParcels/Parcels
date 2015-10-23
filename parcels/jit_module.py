from py import path
import subprocess
from os import environ
import numpy.ctypeslib as npct
from ctypes import c_int, c_float, c_void_p


class Kernel(object):
    """Kernel object that encapsulates auto-generated code.

    :arg filename: Basename for kernel files to generate"""

    def __init__(self, filename):
        self.code = ""
        self.filename = filename
        self.src_file = str(path.local("%s.c" % self.filename))
        self.lib_file = str(path.local("%s.so" % self.filename))
        self.log_file = str(path.local("%s.log" % self.filename))
        self._lib = None

    def generate_code(self, grid):
        parameters = dict(xdim=grid.U.lon.size, ydim=grid.U.lat.size)
        self.code = """
#include <stdio.h>


typedef struct
{
    float lon, lat;
    int xi, yi;
} Particle;


static inline int advance_index(float x, int i, int size, float *xvals)
{
    while (i < size-1 && x > xvals[i+1]) ++i;
    while (i > 0 && x < xvals[i]) --i;
    return i;
}

static inline float interpolate_bilinear(float x, float y, int xi, int yi,
                                        float *xvals, float *yvals,
                                        float qvals[%(ydim)d][%(xdim)d])
{
  int i = xi, j = yi;
  i = advance_index(x, i, %(ydim)d, xvals);
  j = advance_index(y, j, %(xdim)d, yvals);
  return (qvals[i][j] * (xvals[i+1] - x) * (yvals[j+1] - y)
        + qvals[i+1][j] * (x - xvals[i]) * (yvals[j+1] - y)
        + qvals[i][j+1] * (xvals[i+1] - x) * (y - yvals[j])
        + qvals[i+1][j+1] * (x - xvals[i]) * (y - yvals[j]))
        / ((xvals[i+1] - xvals[i]) * (yvals[j+1] - yvals[j]));
}


static inline void runge_kutta4(Particle *p, float dt,
                                float lon_u[%(xdim)d], float lat_u[%(ydim)d],
                                float lon_v[%(xdim)d], float lat_v[%(ydim)d],
                                float u[%(ydim)d][%(xdim)d], float v[%(ydim)d][%(xdim)d])
{
  float f, u1, v1, u2, v2, u3, v3, u4, v4;
  float lon1, lat1, lon2, lat2, lon3, lat3;

  f = dt / 1000. / 1.852 / 60.;
  u1 = interpolate_bilinear(p->lat, p->lon, p->yi, p->xi, lat_u, lon_u, u);
  v1 = interpolate_bilinear(p->lat, p->lon, p->yi, p->xi, lat_v, lon_v, v);
  lon1 = p->lon + u1*.5*f; lat1 = p->lat + v1*.5*f;
  u2 = interpolate_bilinear(lat1, lon1, p->yi, p->xi, lat_u, lon_u, u);
  v2 = interpolate_bilinear(lat1, lon1, p->yi, p->xi, lat_v, lon_v, v);
  lon2 = p->lon + u2*.5*f; lat2 = p->lat + v2*.5*f;
  u3 = interpolate_bilinear(lat2, lon2, p->yi, p->xi, lat_u, lon_u, u);
  v3 = interpolate_bilinear(lat2, lon2, p->yi, p->xi, lat_v, lon_v, v);
  lon3 = p->lon + u3*f; lat3 = p->lat + v3*f;
  u4 = interpolate_bilinear(lat3, lon3, p->yi, p->xi, lat_u, lon_u, u);
  v4 = interpolate_bilinear(lat3, lon3, p->yi, p->xi, lat_v, lon_v, v);

  // Advance particle position in space and on the grid
  p->lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f;
  p->lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f;
  p->xi = advance_index(p->lon, p->xi, %(xdim)d, lon_u);
  p->yi = advance_index(p->lat, p->yi, %(ydim)d, lat_u);
}


void particle_loop(int num_particles, Particle *particles,
                   int timesteps, float dt,
                   float lon_u[%(xdim)d], float lat_u[%(ydim)d],
                   float lon_v[%(xdim)d], float lat_v[%(ydim)d],
                   float u[%(ydim)d][%(xdim)d], float v[%(ydim)d][%(xdim)d])
{
  int p, t;

  for (t = 0; t < timesteps; ++t) {
    for (p = 0; p < num_particles; ++p) {
        runge_kutta4(&(particles[p]), dt, lon_u, lat_u, lon_v, lat_v, u, v);
    }
  }
}

""" % parameters

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        with file(self.src_file, 'w') as f:
            f.write(self.code)
        compiler.compile(self.src_file, self.lib_file, self.log_file)

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, '.')
        self._function = self._lib.particle_loop

    def execute(self, pset, timesteps, dt):
        grid = pset._grid
        self._function(c_int(pset._npart), pset._p_array.ctypes.data_as(c_void_p),
                       c_int(timesteps), c_float(dt),
                       grid.U.lon.ctypes.data_as(c_void_p), grid.U.lat.ctypes.data_as(c_void_p),
                       grid.V.lon.ctypes.data_as(c_void_p), grid.V.lat.ctypes.data_as(c_void_p),
                       grid.U.data.ctypes.data_as(c_void_p), grid.V.data.ctypes.data_as(c_void_p))


class Compiler(object):
    """A compiler object for creating and loading shared libraries.

    :arg cc: C compiler executable (can be overriden by exporting the
        environment variable ``CC``).
    :arg ld: Linker executable (optional, if ``None``, we assume the compiler
        can build object files and link in a single invocation, can be
        overridden by exporting the environment variable ``LDSHARED``).
    :arg cppargs: A list of arguments to the C compiler (optional).
    :arg ldargs: A list of arguments to the linker (optional)."""

    def __init__(self, cc, ld=None, cppargs=[], ldargs=[]):
        self._cc = environ.get('CC', cc)
        self._ld = environ.get('LDSHARED', ld)
        self._cppargs = cppargs
        self._ldargs = ldargs

    def compile(self, src, obj, log):
        cc = [self._cc] + self._cppargs + ['-o', obj, src] + self._ldargs
        with file(log, 'w') as logfile:
            logfile.write("Compiling: %s\n" % " ".join(cc))
            try:
                subprocess.check_call(cc, stdout=logfile, stderr=logfile)
            except OSError:
                err = """OSError during compilation
Please check if compiler exists: %s""" % self._cc
                raise RuntimeError(err)
            except subprocess.CalledProcessError:
                err = """Error during compilation:
Compilation command: %s
Source file: %s
Log file: %s""" % (" ".join(cc), src, logfile.name)
                raise RuntimeError(err)
        print "Compiled:", obj


class GNUCompiler(Compiler):
    """A compiler object for the GNU Linux toolchain.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""
    def __init__(self, cppargs=[], ldargs=[]):
        opt_flags = ['-g', '-O3']
        cppargs = ['-Wall', '-fPIC'] + opt_flags + cppargs
        ldargs = ['-shared'] + ldargs
        super(GNUCompiler, self).__init__("gcc", cppargs=cppargs, ldargs=ldargs)
