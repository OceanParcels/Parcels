from py import path
import subprocess
from os import environ
import numpy as np


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

    def generate_code(self):
        self.code = """
#include <stdio.h>

typedef struct
{
    float lon, lat;
    int xi, yi;
} Particle;

void particle_kernel(Particle *p)
{
    printf("Particle: P(%f, %f)[%d, %d]\\n", p->lon, p->lat, p->xi, p->yi);
}
"""

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        with file(self.src_file, 'w') as f:
            f.write(self.code)
        compiler.compile(self.src_file, self.lib_file, self.log_file)

    def load_lib(self):
        self._lib = np.ctypeslib.load_library(self.lib_file, '.')
        self._function = self._lib.particle_kernel

    def execute(self, pset):
        self._function(pset._p_array.ctypes.data)


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
