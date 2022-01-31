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
from parcels.particleset.numba_aos import convert_pset_to_tlist,\
    convert_tlist_to_pset
from numba.core.decorators import njit
from parcels.numba.utils import _numba_isclose
try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.kernel.basekernel import BaseKernel
# from parcels.compilation.codegenerator import ArrayKernelGenerator as KernelGenerator
# from parcels.compilation.codegenerator import LoopGenerator
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
import parcels.rng as ParcelsRandom  # noqa
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode
from parcels.tools.statuscodes import recovery_map as recovery_base_map
from parcels.tools.loggers import logger
import numba as nb

__all__ = ['KernelSOA']


class KernelSOA(BaseKernel):
    """Kernel object that encapsulates auto-generated code.

    :arg fieldset: FieldSet object providing the field information
    :arg ptype: PType object for the kernel particle
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None,
                 funccode=None, py_ast=None, funcvars=None):
        super(KernelSOA, self).__init__(
            fieldset=fieldset, ptype=ptype, pyfunc=pyfunc, funcname=funcname,
            funccode=funccode, py_ast=py_ast, funcvars=funcvars)

        # Derive meta information from pyfunc, if not given
        self.check_fieldsets_in_kernels(pyfunc)

        if funcvars is not None:
            self.funcvars = funcvars
        elif hasattr(pyfunc, '__code__'):
            self.funcvars = list(pyfunc.__code__.co_varnames)
        else:
            self.funcvars = None
        self.funccode = funccode or inspect.getsource(pyfunc.__code__)
        # Parse AST if it is not provided explicitly
        self.py_ast = py_ast or parse(BaseKernel.fix_indentation(self.funccode)).body[0]
        if pyfunc is None:
            # Extract user context by inspecting the call stack
            stack = inspect.stack()
            try:
                user_ctx = stack[-1][0].f_globals
                user_ctx['math'] = globals()['math']
                user_ctx['ParcelsRandom'] = globals()['ParcelsRandom']
                user_ctx['random'] = globals()['random']
                user_ctx['StateCode'] = globals()['StateCode']
                user_ctx['OperationCode'] = globals()['OperationCode']
                user_ctx['ErrorCode'] = globals()['ErrorCode']
            except:
                logger.warning("Could not access user context when merging kernels")
                user_ctx = globals()
            finally:
                del stack  # Remove cyclic references
            # Compile and generate Python function from AST
            py_mod = parse("")
            py_mod.body = [self.py_ast]
            exec(compile(py_mod, "<ast>", "exec"), user_ctx)
            self._pyfunc = user_ctx[self.funcname]
        else:
            self._pyfunc = pyfunc

        numkernelargs = self.check_kernel_signature_on_version()

        assert numkernelargs == 3, \
            'Since Parcels v2.0, kernels do only take 3 arguments: particle, fieldset, time !! AND !! Argument order in field interpolation is time, depth, lat, lon.'

        self.name = "%s%s" % (ptype.name, self.funcname)

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        # sign of dt: { [0, 1]: forward simulation; -1: backward simulation }
        sign_dt = np.sign(dt)

        analytical = False
        if 'AdvectionAnalytical' in self._pyfunc.__name__:
            analytical = True
            if not np.isinf(dt):
                logger.warning_once('dt is not used in AnalyticalAdvection, so is set to np.inf')
            dt = np.inf

        if self.fieldset is not None:
            for f in self.fieldset.get_fields():
                if type(f) in [VectorField, NestedField, SummedField]:
                    continue
                f.data = np.array(f.data)

        inner_loop(pset._collection._data, endtime, sign_dt, dt,
                   self.compiled_pyfunc, self._fieldset.numba_fieldset,
                   pset._collection._pbackup, analytical=analytical)

    @property
    def compiled_pyfunc(self):
        if self._compiled_pyfunc is None:
            self._compiled_pyfunc = njit(self._pyfunc)
        return self._compiled_pyfunc

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.)
        super(KernelSOA, self).__del__()

    def __add__(self, kernel):
        if not isinstance(kernel, KernelSOA):
            kernel = KernelSOA(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, KernelSOA)

    def __radd__(self, kernel):
        if not isinstance(kernel, KernelSOA):
            kernel = KernelSOA(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, KernelSOA)

    def remove_deleted(self, pset, output_file, endtime):
        """
        Utility to remove all particles that signalled deletion

        This deletion function is targetted to index-addressable, random-access array-collections.
        """
        # Indices marked for deletion.
        bool_indices = pset.collection.state == OperationCode.Delete
        indices = np.where(bool_indices)[0]
        if len(indices) > 0 and output_file is not None:
            output_file.write(pset, endtime, deleted_only=bool_indices)
        pset.remove_indices(indices)

    def execute(self, pset, endtime, dt, recovery=None, output_file=None, execute_once=False):
        """Execute this Kernel over a ParticleSet for several timesteps"""
        pset.collection.state[:] = StateCode.Evaluate

        if abs(dt) < 1e-6 and not execute_once:
            logger.warning_once("'dt' is too small, causing numerical accuracy limit problems. Please chose a higher 'dt' and rather scale the 'time' axis of the field accordingly. (related issue #762)")

        if recovery is None:
            recovery = {}
        elif ErrorCode.ErrorOutOfBounds in recovery and ErrorCode.ErrorThroughSurface not in recovery:
            recovery[ErrorCode.ErrorThroughSurface] = recovery[ErrorCode.ErrorOutOfBounds]
        recovery_map = recovery_base_map.copy()
        recovery_map.update(recovery)

        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)

        # Remove all particles that signalled deletion
        self.remove_deleted(pset, output_file=output_file, endtime=endtime)   # Generalizable version!

        # Identify particles that threw errors
        n_error = pset.num_error_particles

        while n_error > 0:
            error_pset = pset.error_particles
            # Apply recovery kernel
            for p in error_pset:
                if p.state == OperationCode.StopExecution:
                    return
                if p.state == OperationCode.Repeat:
                    p.reset_state()
                elif p.state == OperationCode.Delete:
                    pass
                elif p.state in recovery_map:
                    recovery_kernel = recovery_map[p.state]
                    p.state(StateCode.Success)
                    recovery_kernel(p, self.fieldset, p.time)
                    if p.isComputed():
                        p.reset_state()
                else:
                    logger.warning_once('Deleting particle {} because of non-recoverable error'.format(p.id))
                    p.delete()

            # Remove all particles that signalled deletion
            self.remove_deleted(pset, output_file=output_file, endtime=endtime)   # Generalizable version!

            # Execute core loop again to continue interrupted particles
            if self.ptype.uses_jit:
                self.execute_jit(pset, endtime, dt)
            else:
                self.execute_python(pset, endtime, dt)

            n_error = pset.num_error_particles


@nb.njit(cache=False)
def inner_loop(pset, endtime, sign_dt, dt,
               compiled_pyfunc,
               numba_fieldset,
               pbackup,
               analytical=False):
    for idx in range(len(pset)):
        evaluate_particle(
            pset, idx, endtime, sign_dt, dt, compiled_pyfunc,
            numba_fieldset, pbackup, analytical=analytical)
    return 0


@nb.njit(cache=False)
def evaluate_particle(pset, idx, endtime, sign_dt, dt, pyfunc,
                      fieldset, pbackup,
                      analytical=False):
    """Inner loop that can be compiled.

    NOTE: User kernels should ALWAYS give back a status code,
    otherwise crashes are expected.

    Execute the kernel evaluation of for an individual particle.
    :arg p: object of (sub-)type (ScipyParticle, JITParticle) or (sub-)type of BaseParticleAccessor
    :arg fieldset: fieldset of the containing ParticleSet (e.g. pset.fieldset)
    :arg analytical: flag indicating the analytical advector or an iterative advection
    :arg endtime: endtime of this overall kernel evaluation step
    :arg dt: computational integration timestep
    """
    # back up variables in case of OperationCodeRepeat
    # Back up is another record array with length 1.
    p = pset[idx]
    pbackup[0] = p
    pdt_prekernels = .0
    # Don't execute particles that aren't started yet
    sign_end_part = np.sign(endtime - p.time)
    # Compute min/max dt for first timestep. Only use endtime-p.time for one timestep
    reset_dt = False
    if abs(endtime - p.time) < abs(p.dt):
        dt_pos = abs(endtime - p.time)
        reset_dt = True
    else:
        dt_pos = abs(p.dt)
        reset_dt = False

    # ==== numerically stable; also making sure that continuously-recovered particles do end successfully,
    # as they fulfil the condition here on entering at the final calculation here. ==== #
    if ((sign_end_part != sign_dt) or _numba_isclose(dt_pos, 0)) and not _numba_isclose(dt, 0):
        if abs(p.time) >= abs(endtime):
            p.state = StateCode.Success
        return p

    while p.state == StateCode.Evaluate or p.state == OperationCode.Repeat or _numba_isclose(dt, 0):
        pdt_prekernels = sign_dt * dt_pos
        p.dt = pdt_prekernels
        state_prev = p.state
        res = pyfunc(p, fieldset, p.time)

        if res is StateCode.Success and p.state != state_prev:
            res = p.state

        if not analytical and res == StateCode.Success and not _numba_isclose(p.dt, pdt_prekernels):
            res = OperationCode.Repeat
            # TODO: Do error handling

        # Handle particle time and time loop
        if res == StateCode.Success or res == OperationCode.Delete:
            # Update time and repeat
            p.time += p.dt
            if reset_dt and p.dt == pdt_prekernels:
                p.dt = dt
            if not np.isnan(p._next_dt):
                p.dt = p._next_dt
                p._next_dt = np.nan
                # NOTE: particle is not an object, so we can't do this:
                # p.update_next_dt()
            if analytical:
                p.dt = np.inf
            if abs(endtime - p.time) < abs(p.dt):
                dt_pos = abs(endtime - p.time)
                reset_dt = True
            else:
                dt_pos = abs(p.dt)
                reset_dt = False

            sign_end_part = np.sign(endtime - p.time)
            if res != OperationCode.Delete and not _numba_isclose(dt_pos, 0) and (sign_end_part == sign_dt):
                res = StateCode.Evaluate
            if sign_end_part != sign_dt:
                dt_pos = 0

            p.state = res
            if _numba_isclose(dt, 0):
                break
        else:
            p.state = res
            # Try again without time update
            pset[idx] = pbackup[0]
            if abs(endtime - p.time) < abs(p.dt):
                dt_pos = abs(endtime - p.time)
                reset_dt = True
            else:
                dt_pos = abs(p.dt)
                reset_dt = False

            sign_end_part = np.sign(endtime - p.time)
            if sign_end_part != sign_dt:
                dt_pos = 0
            break
    return p
