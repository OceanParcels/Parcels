import inspect
import math  # noqa
import random  # noqa
from ast import parse
from collections import defaultdict
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
                 funccode=None, py_ast=None, funcvars=None, c_include="",
                 delete_cfiles=True):
        super().__init__(fieldset=fieldset, ptype=ptype, pyfunc=pyfunc,
                         funcname=funcname, funccode=funccode, py_ast=py_ast,
                         funcvars=funcvars, c_include=c_include,
                         delete_cfiles=delete_cfiles)

        for func in self._pyfunc:
            self.check_fieldsets_in_kernels(func)

        numkernelargs = self.check_kernel_signature_on_version()

        assert numkernelargs[0] == 5 and \
            numkernelargs.count(numkernelargs[0]) == len(numkernelargs), \
            'Interactionkernels take exactly 5 arguments: particle, fieldset, time, neighbours, mutator'

        # At this time, JIT mode is not supported for InteractionKernels,
        # so there is no need for any further "processing" of pyfunc's.

    def execute_jit(self, pset, endtime, dt):
        raise NotImplementedError

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.)
        super().__del__()

    def __add__(self, kernel):
        if not isinstance(kernel, InteractionKernelSOA):
            kernel = InteractionKernelSOA(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, InteractionKernelSOA)

    def __radd__(self, kernel):
        if not isinstance(kernel, InteractionKernelSOA):
            kernel = InteractionKernelSOA(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, InteractionKernelSOA)

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        # back up variables in case of OperationCode.Repeat
        p_var_back = {}

        if self.fieldset is not None:
            for f in self.fieldset.get_fields():
                if type(f) in [VectorField, NestedField, SummedField]:
                    continue
                f.data = np.array(f.data)

        for pyfunc in self._pyfunc:
            pset.compute_neighbor_tree(endtime, dt)
            active_idx = pset._active_particle_idx

            mutator = defaultdict(lambda: [])
            for particle_idx in active_idx:
                p = pset[particle_idx]
                # Don't use particles that are not started.
                if (endtime-p.time)/dt <= -1e-7:
                    continue

                neighbors = pset.neighbors_by_index(particle_idx)
                try:
                    res = pyfunc(p, pset.fieldset, p.time, neighbors, mutator)
                except FieldOutOfBoundError as fse_xy:
                    res = ErrorCode.ErrorOutOfBounds
                    p.exception = fse_xy
                except FieldOutOfBoundSurfaceError as fse_z:
                    res = ErrorCode.ErrorThroughSurface
                    p.exception = fse_z
                except TimeExtrapolationError as fse_t:
                    res = ErrorCode.ErrorTimeExtrapolation
                    p.exception = fse_t

                except Exception as e:
                    res = ErrorCode.Error
                    p.exception = e

            for particle_idx in active_idx:
                p = pset[particle_idx]
                try:
                    for m in mutator[p.id]:
                        m(p)
                except KeyError:
                    pass

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

        if pset.fieldset is not None:
            for g in pset.fieldset.gridset.grids:
                if len(g.load_chunk) > g.chunk_not_loaded:  # not the case if a field in not called in the kernel
                    g.load_chunk = np.where(g.load_chunk == g.chunk_loaded_touched,
                                            g.chunk_deprecated, g.load_chunk)

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
                    p.set_state(StateCode.Evaluate)
                elif p.state == OperationCode.Delete:
                    pass
                elif p.state in recovery_map:
                    recovery_kernel = recovery_map[p.state]
                    p.set_state(StateCode.Success)
                    recovery_kernel(p, self.fieldset, p.time)
                    if p.state == StateCode.Success:
                        p.set_state(StateCode.Evaluate)
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

