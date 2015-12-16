from py import path
import subprocess
import os
import numpy.ctypeslib as npct
from ctypes import c_int, c_float, c_void_p, POINTER


def get_package_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


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

    def generate_code(self, grid, ptype=None):
        parameters = dict(xdim=grid.U.lon.size, ydim=grid.U.lat.size,
                          ptype=ptype.code if ptype else "")
        self.code = """
%(ptype)s

const int GRID_XDIM = %(xdim)d;
const int GRID_YDIM = %(ydim)d;

#include "parcels.h"

/* Outer execution loop for particle computation */
void particle_loop(int num_particles, Particle *particles,
                   int timesteps, float dt,
                   float lon_u[GRID_XDIM], float lat_u[GRID_YDIM],
                   float lon_v[GRID_XDIM], float lat_v[GRID_YDIM],
                   float u[GRID_YDIM][GRID_XDIM], float v[GRID_YDIM][GRID_XDIM])
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
        self._function(c_int(len(pset)), pset._particle_data.ctypes.data_as(c_void_p),
                       c_int(timesteps), c_float(dt),
                       grid.U.lon.ctypes.data_as(POINTER(c_float)),
                       grid.U.lat.ctypes.data_as(POINTER(c_float)),
                       grid.V.lon.ctypes.data_as(POINTER(c_float)),
                       grid.V.lat.ctypes.data_as(POINTER(c_float)),
                       grid.U.data.ctypes.data_as(POINTER(POINTER(c_float))),
                       grid.V.data.ctypes.data_as(POINTER(POINTER(c_float))))


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
        self._cc = os.environ.get('CC', cc)
        self._ld = os.environ.get('LDSHARED', ld)
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
        cppargs = ['-Wall', '-fPIC', '-I%s/include' % get_package_dir()] + opt_flags + cppargs
        ldargs = ['-shared'] + ldargs
        super(GNUCompiler, self).__init__("gcc", cppargs=cppargs, ldargs=ldargs)
