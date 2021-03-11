import re
import _ctypes
import inspect
import numpy.ctypeslib as npct
from time import time as ostime
from os import path
from os import remove
from sys import platform
from sys import version_info
from weakref import finalize
from ast import FunctionDef
from hashlib import md5
from parcels.tools.loggers import logger
from numpy import ndarray

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.tools.global_statics import get_cache_dir

# === import just necessary field classes to perform setup checks === #
from parcels.field import Field
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.grid import GridCode
from parcels.kernels.advection import AdvectionRK4_3D
from parcels.kernels.advection import AdvectionAnalytical
from parcels.tools.statuscodes import OperationCode
from parcels.basekernel import BaseKernel

__all__ = ['BaseKernel']


re_indent = re.compile(r"^(\s+)")


class BaseInteractionKernel(BaseKernel):
    """Base super class for Interaction Kernel objects that encapsulates
    auto-generated code.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None,
                 funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        super(BaseInteractionKernel, self).__init__(
            fieldset=fieldset, ptype=ptype, pyfunc=pyfunc, funcname=funcname,
            funccode=funccode, py_ast=py_ast, funcvars=funcvars,
            c_include=c_include, delete_cfiles=delete_cfiles)

        if pyfunc is not None:
            self._pyfunc = [pyfunc]
            print("here", self.pyfunc)
#             raise ValueError()


    def __del__(self):
        raise NotImplementedError

    @property
    def ptype(self):
        return self._ptype

    @property
    def pyfunc(self):
        return self._pyfunc

    @property
    def fieldset(self):
        return self._fieldset

    @property
    def c_include(self):
        return self._c_include

    @property
    def _cache_key(self):
        raise NotImplementedError

    @staticmethod
    def fix_indentation(string):
        raise NotImplementedError

    def check_fieldsets_in_kernels(self, pyfunc):
        raise NotImplementedError

    def check_kernel_signature_on_version(self):
        raise NotImplementedError

    def remove_lib(self):
        raise NotImplementedError

    def get_kernel_compile_files(self):
        raise NotImplementedError

    def compile(self, compiler):
        raise NotImplementedError

    def load_lib(self):
        raise NotImplementedError

    def merge(self, kernel, kclass):
        raise NotImplementedError

    def __add__(self, kernel):
        if isinstance(kernel, BaseInteractionKernel):
            assert self.__class__ == kernel._class__
            funcname = self.funcname + kernel.funcname
            delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
            pyfunc = self._pyfunc + kernel._pyfunc

            # TODO: Not sure what the func_ast is about.
            new_kernel = self.__class__(
                self.fieldset, self.ptype, pyfunc=pyfunc,
                funcname=funcname, funccode=self.funccode + kernel.funccode,
                py_ast=self.func_ast, funcvars=self.funcvars + kernel.funcvars,
                c_include=self._c_include + kernel.c_include,
                delete_cfiles=delete_cfiles)
            return new_kernel
        raise NotImplementedError

    def __radd__(self, kernel):
        raise NotImplementedError

    @staticmethod
    def cleanup_remove_files(lib_file, all_files_array, delete_cfiles):
        raise NotImplementedError

    @staticmethod
    def cleanup_unload_lib(lib):
        raise NotImplementedError

    def execute_jit(self, pset, endtime, dt):
        raise NotImplementedError

