from parcels.codegenerator import KernelGenerator, LoopGenerator
from py import path
import math  # NOQA get flake8 to ignore unused import.
import numpy.ctypeslib as npct
from ctypes import c_int, c_float, c_double, c_void_p, byref
from ast import parse, FunctionDef, Module
import inspect
from copy import deepcopy
import re


re_indent = re.compile(r"^(\s+)")


def fix_indentation(string):
    """Fix indentation to allow in-lined kernel definitions"""
    lines = string.split('\n')
    indent = re_indent.match(lines[0])
    if indent:
        lines = [l.replace(indent.groups()[0], '', 1) for l in lines]
    return "\n".join(lines)


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
            funccode = inspect.getsource(pyfunc.__code__)
            self.py_ast = parse(fix_indentation(funccode))
            self.py_ast = self.py_ast.body[0]
            self.pyfunc = pyfunc
        else:
            self.funcname = funcname
            self.py_ast = py_ast
            self.funcvars = funcvars
            # Extract user context by inspecting the call stack
            stack = inspect.stack()
            try:
                user_ctx = stack[-1][0].f_globals
                user_ctx['math'] = globals()['math']
            except:
                print("Warning: Could not access user context when merging kernels")
                user_ctx = globals()
            finally:
                del stack  # Remove cyclic references
            # Compile and generate Python function from AST
            py_mod = Module(body=[self.py_ast])
            exec(compile(py_mod, "<ast>", "exec"), user_ctx)
            self.pyfunc = user_ctx[self.funcname]

        self.name = "%s%s" % (ptype.name, self.funcname)

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
        if self.ptype.uses_jit:
            fargs = [byref(f.ctypes_struct) for f in self.field_args.values()]
            particle_data = pset._particle_data.ctypes.data_as(c_void_p)
            self._function(c_int(len(pset)), particle_data, c_int(timesteps),
                           c_double(time), c_float(dt), *fargs)
        else:
            for _ in range(int(timesteps)):
                for p in pset.particles:
                    if p.active == 1:
                        self.pyfunc(p, pset.grid, time, dt)
                time += dt

    def execute_adaptive(self, pset, tol, output_time=None, end_time=None):
        for p in pset.particles:
            if output_time is None:
                self.pyfunc(p, pset.grid, end_time, tol)
                return
            while p.time < output_time:
                self.pyfunc(p, pset.grid, output_time, tol)

    def merge(self, kernel):
        funcname = self.funcname + kernel.funcname
        func_ast = FunctionDef(name=funcname, args=self.py_ast.args,
                               body=self.py_ast.body + kernel.py_ast.body,
                               decorator_list=[], lineno=1, col_offset=0)
        return Kernel(self.grid, self.ptype, pyfunc=None, funcname=funcname,
                      py_ast=func_ast, funcvars=self.funcvars + kernel.funcvars)

    def __add__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel(self.grid, self.ptype, pyfunc=kernel)
        return self.merge(kernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel(self.grid, self.ptype, pyfunc=kernel)
        return kernel.merge(self)
