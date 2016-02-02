from parcels.codegenerator import KernelGenerator, LoopGenerator
from py import path
import numpy.ctypeslib as npct
from ctypes import c_int, c_float, c_double, c_void_p, byref
from ast import parse, FunctionDef
import inspect
from copy import deepcopy


class Kernel(object):
    """Kernel object that encapsulates auto-generated code.

    :arg filename: Basename for kernel files to generate"""

    def __init__(self, grid, ptype, pyfunc=None, funcname=None,
                 py_ast=None, funcvars=None):
        self.grid = grid
        self.ptype = ptype

        if pyfunc is not None:
            self.funcname = pyfunc.__name__
            self.funcvars = list(pyfunc.__code__.co_varnames)
            # Parse the Python code into an AST
            self.py_ast = parse(inspect.getsource(pyfunc.__code__))
            self.py_ast = self.py_ast.body[0]
            self.pyfunc = pyfunc
        else:
            self.funcname = funcname
            self.py_ast = py_ast
            self.funcvars = funcvars
            # Compile and generate Python function from AST
            py_mod = Module(body=[self.py_ast])
            py_ctx = {}
            exec(compile(py_mod, "<ast>", "exec"), py_ctx)
            self.pyfunc = py_ctx[self.funcname]

        self.name = "%s%s" % (ptype.name, funcname)

        self.src_file = str(path.local("%s.c" % self.name))
        self.lib_file = str(path.local("%s.so" % self.name))
        self.log_file = str(path.local("%s.log" % self.name))
        self._lib = None

        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            kernelgen = KernelGenerator(grid, ptype)
            kernel_ccode = kernelgen.generate(deepcopy(self.py_ast),
                                              self.funcvars)
            self.field_args = kernelgen.field_args
            loopgen = LoopGenerator(grid, ptype)
            self.ccode = loopgen.generate(self.funcname,
                                          self.field_args,
                                          kernel_ccode)

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

    def merge(self, kernel):
        funcname = self.funcname + kernel.funcname
        func_ast = FunctionDef(name=funcname, args=self.py_ast.args,
                               body=self.py_ast.body + kernel.py_ast.body,
                               decorator_list=[], lineno=1, col_offset=0)
        return Kernel(self.grid, self.ptype, pyfunc=None, funcname=funcname,
                      py_ast=func_ast, funcvars=self.funcvars + kernel.funcvars)

    def __add__(self, kernel):
        return self.merge(kernel)

    def __radd__(self, kernel):
        return kernel.merge(self)
