from parcels.codegen import KernelGenerator, LoopGenerator
from py import path
import numpy.ctypeslib as npct
from ctypes import c_int, c_float, c_void_p, POINTER


class Kernel(object):
    """Kernel object that encapsulates auto-generated code.

    :arg filename: Basename for kernel files to generate"""

    def __init__(self, name):
        self.name = name
        self.ccode = None
        self._lib = None

        self.src_file = str(path.local("%s.c" % self.name))
        self.lib_file = str(path.local("%s.so" % self.name))
        self.log_file = str(path.local("%s.log" % self.name))

    def generate_code(self, grid, ptype, pyfunc):
        ccode_kernel = KernelGenerator(grid, ptype).generate(pyfunc)
        self.ccode = LoopGenerator(grid, ptype).generate(self.name, ccode_kernel)

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        with file(self.src_file, 'w') as f:
            f.write(self.ccode)
        compiler.compile(self.src_file, self.lib_file, self.log_file)

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, '.')
        self._function = self._lib.particle_loop

    def execute(self, pset, timesteps, dt):
        grid = pset.grid
        self._function(c_int(len(pset)), pset._particle_data.ctypes.data_as(c_void_p),
                       c_int(timesteps), c_float(dt),
                       grid.U.lon.ctypes.data_as(POINTER(c_float)),
                       grid.U.lat.ctypes.data_as(POINTER(c_float)),
                       grid.V.lon.ctypes.data_as(POINTER(c_float)),
                       grid.V.lat.ctypes.data_as(POINTER(c_float)),
                       grid.U.data.ctypes.data_as(POINTER(POINTER(c_float))),
                       grid.V.data.ctypes.data_as(POINTER(POINTER(c_float))))
