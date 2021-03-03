import inspect
import math  # noqa
import random  # noqa
from ast import parse
from copy import deepcopy
from ctypes import byref
from ctypes import c_double
from ctypes import c_int
from os import path

import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.baseinteractionkernel import BaseInteractionKernel
from parcels.compilation.codegenerator import KernelGenerator
from parcels.compilation.codegenerator import LoopGenerator
from parcels.field import FieldOutOfBoundError
from parcels.field import FieldOutOfBoundSurfaceError
from parcels.field import TimeExtrapolationError
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
import parcels.rng as ParcelsRandom  # noqa
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode
from parcels.tools.statuscodes import recovery_map as recovery_base_map
from parcels.tools.loggers import logger


__all__ = ['InteractionKernelSOA']


class InteractionKernelSOA(BaseInteractionKernel):
    """InteractionKernel object that encapsulates auto-generated code.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None,
                 funccode=None, py_ast=None, funcvars=None, c_include="", delete_cfiles=True):
        super().__init__(fieldset=fieldset, ptype=ptype, pyfunc=pyfunc, funcname=funcname, funccode=funccode, py_ast=py_ast, funcvars=funcvars, c_include=c_include, delete_cfiles=delete_cfiles)

        raise NotImplementedError

    def execute_jit(self, pset, endtime, dt):
        raise NotImplementedError

    def execute_python(self, pset, endtime, dt):
        raise NotImplementedError

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.)
        super().__del__()

    def __add__(self, kernel):
        raise NotImplementedError

    def __radd__(self, kernel):
        raise NotImplementedError

    def remove_deleted(self, pset, output_file, endtime):
        raise NotImplementedError

    def execute(self, pset, endtime, dt, recovery=None, output_file=None, execute_once=False):
        raise NotImplementedError
