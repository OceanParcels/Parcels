from parcels.codegenerator import KernelGenerator, LoopGenerator
from py import path
import numpy.ctypeslib as npct
from ctypes import c_int, c_float, c_double, c_void_p, byref


class Kernel(object):
    """Kernel object that encapsulates auto-generated code.

    :arg filename: Basename for kernel files to generate"""

    def __init__(self, grid, ptype, pyfunc):
        self.name = "%s%s" % (ptype.name, pyfunc.__name__)
        self.src_file = str(path.local("%s.c" % self.name))
        self.lib_file = str(path.local("%s.so" % self.name))
        self.log_file = str(path.local("%s.log" % self.name))
        self._lib = None

        # Generate the kernel function and add the outer loop
        kernelgen = KernelGenerator(grid, ptype)
        self.ccode = kernelgen.generate(pyfunc)
        self.field_args = kernelgen.field_args

        loopgen = LoopGenerator(grid, ptype)
        self.ccode = loopgen.generate(pyfunc.__name__, self.field_args,
                                      self.ccode)

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        with open(self.src_file, 'w') as f:
            f.write(self.ccode)
        compiler.compile(self.src_file, self.lib_file, self.log_file)

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, '.')
        self._function = self._lib.particle_loop

    def execute(self, pset, timesteps, time, dt):
        grid = pset.grid
        fargs = [byref(f.ctypes_struct) for f in self.field_args.values()]
        self._function(c_int(len(pset)), pset._particle_data.ctypes.data_as(c_void_p),
                       c_int(timesteps), c_double(time), c_float(dt), *fargs)
