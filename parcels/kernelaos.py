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

from parcels.basekernel import BaseKernel
from parcels.compilation.codegenerator import ObjectKernelGenerator as KernelGenerator
from parcels.compilation.codegenerator import ParticleObjectLoopGenerator
from parcels.field import FieldOutOfBoundError
from parcels.field import FieldOutOfBoundSurfaceError
from parcels.field import TimeExtrapolationError
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
import parcels.rng as ParcelsRandom  # noqa
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode  # noqa
from parcels.tools.statuscodes import recovery_map as recovery_base_map
from parcels.tools.loggers import logger


__all__ = ['KernelAOS']


class KernelAOS(BaseKernel):
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
                 funccode=None, py_ast=None, funcvars=None, c_include="", delete_cfiles=True):
        super(KernelAOS, self).__init__(fieldset=fieldset, ptype=ptype, pyfunc=pyfunc, funcname=funcname, funccode=funccode, py_ast=py_ast, funcvars=funcvars, c_include=c_include, delete_cfiles=delete_cfiles)

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

        self.name = "%s%s" % (self._ptype.name, self.funcname)
        # logger.info("KernelAOS::__init__.ptype_name = ({}, {}, {}, {})".format(ptype.name, self._ptype.name, self.funcname, self.name))

        # ======== THIS NEEDS TO BE REFACTORED BASED ON THE TYPE OF PARTICLE BEING USED ======== #
        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            kernelgen = KernelGenerator(self.fieldset, ptype)
            kernel_ccode = kernelgen.generate(deepcopy(self.py_ast), self.funcvars)
            self.field_args = kernelgen.field_args
            self.vector_field_args = kernelgen.vector_field_args
            for f in self.vector_field_args.values():
                Wname = f.W.ccode_name if f.W else 'not_defined'
                for sF_name, sF_component in zip([f.U.ccode_name, f.V.ccode_name, Wname], ['U', 'V', 'W']):
                    if sF_name not in self.field_args:
                        if sF_name != 'not_defined':
                            self.field_args[sF_name] = getattr(f, sF_component)
            self.const_args = kernelgen.const_args
            loopgen = ParticleObjectLoopGenerator(self.fieldset, ptype)
            if path.isfile(c_include):
                with open(c_include, 'r') as f:
                    c_include_str = f.read()
            else:
                c_include_str = c_include
            self.ccode = loopgen.generate(self.funcname, self.field_args, self.const_args,
                                          kernel_ccode, c_include_str)
            # logger.info("KernelAOS: Generated Kernel - compiling C-code now ...")

            src_file_or_files, self.lib_file, self.log_file = self.get_kernel_compile_files()
            # logger.info("KernelAOS: C-code compiled.")
            if type(src_file_or_files) in (list, dict, tuple, np.ndarray):
                self.dyn_srcs = src_file_or_files
            else:
                self.src_file = src_file_or_files

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.)
        super(KernelAOS, self).__del__()

    def __add__(self, kernel):
        if not isinstance(kernel, KernelAOS):
            kernel = KernelAOS(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, KernelAOS)

    def __radd__(self, kernel):
        if not isinstance(kernel, KernelAOS):
            kernel = KernelAOS(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, KernelAOS)

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""

        if pset.fieldset is not None:
            for g in pset.fieldset.gridset.grids:
                g.cstruct = None  # This force to point newly the grids from Python to C
            # Make a copy of the transposed array to enforce
            # C-contiguous memory layout for JIT mode.
            for f in pset.fieldset.get_fields():
                if type(f) in [VectorField, NestedField, SummedField]:
                    continue
                if f in self.field_args.values():
                    f.chunk_data()
                else:
                    for block_id in range(len(f.data_chunks)):
                        f.data_chunks[block_id] = None
                        f.c_data_chunks[block_id] = None

            for g in pset.fieldset.gridset.grids:
                g.load_chunk = np.where(g.load_chunk == g.chunk_loading_requested,
                                        g.chunk_loaded_touched, g.load_chunk)
                if len(g.load_chunk) > g.chunk_not_loaded:  # not the case if a field in not called in the kernel
                    if not g.load_chunk.flags.c_contiguous:
                        g.load_chunk = g.load_chunk.copy()
                if not g.depth.flags.c_contiguous:
                    g.depth = g.depth.copy()
                if not g.lon.flags.c_contiguous:
                    g.lon = g.lon.copy()
                if not g.lat.flags.c_contiguous:
                    g.lat = g.lat.copy()

        fargs = []
        if self.field_args is not None:
            fargs += [byref(f.ctypes_struct) for f in self.field_args.values()]
        if self.const_args is not None:
            fargs += [c_double(f) for f in self.const_args.values()]

        # pdata = pset.particle_data.ctypes.data_as(c_void_p)
        pdata = pset.ctypes_struct
        # logger.info("KernelAOS::execute_jit.pdata = {}".format(pdata))
        # d_array = pset.particle_data.astype(pset.ptype.dtype)
        # logger.info("KernelAOS::execute_jit.d_array = {}".format(d_array[0]))
        if len(fargs) > 0:
            self._function(c_int(len(pset)), pdata, c_double(endtime), c_double(dt), *fargs)
        else:
            self._function(c_int(len(pset)), pdata, c_double(endtime), c_double(dt))

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        if 'AdvectionAnalytical' in self._pyfunc.__name__:
            analytical = True
            if not np.isinf(dt):
                logger.warning_once('dt is not used in AnalyticalAdvection, so is set to np.inf')
            dt = np.inf
        else:
            analytical = False

        # back up variables in case of ErrorCode.Repeat
        p_var_back = {}

        for f in self.fieldset.get_fields():
            if type(f) in [VectorField, NestedField, SummedField]:
                continue
            f.data = np.array(f.data)

        # for f in pset.fieldset.get_fields():
        # for f in self.fieldset.get_fields():
        #     if type(f) in [VectorField, NestedField, SummedField]:
        #         continue
        #     if f in pset.fieldset.get_fields():
        #         f.chunk_data()
        #     else:
        #         for block_id in range(len(f.data_chunks)):
        #             f.data_chunks[block_id] = None
        #             f.c_data_chunks[block_id] = None

        # for p in pset:
        prev_i = -1
        i = -1
        # for i, p in enumerate(pset.iterator()):
        # for i, p in enumerate(pset):
        for p in pset:
            i += 1
            logger.info("iteration {} - endtime: {} - p.id={} p.time={} p.coord=({}, {}, {})".format(i, endtime, p.id, p.time, p.lon, p.lat, p.depth))
            ptype = p.getPType()
            if prev_i == i:
                raise IndexError
            prev_i = i
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
            if ((sign_end_part != sign_dt) or np.isclose(dt_pos, 0)) and not np.isclose(dt, 0):
                if abs(p.time) >= abs(endtime):
                    p.set_state(StateCode.Success)
                continue

            while p.state in [StateCode.Evaluate, OperationCode.Repeat] or np.isclose(dt, 0):
                if i == 0:
                    logger.info("P at time={}: {}".format(endtime, p))

                # for var in pset.collection.ptype.variables:
                for var in ptype.variables:
                    p_var_back[var.name] = getattr(p, var.name)
                try:
                    pdt_prekernels = sign_dt * dt_pos
                    p.dt = pdt_prekernels
                    state_prev = p.state
                    res = self.pyfunc(p, pset.fieldset, p.time)
                    if res is None:
                        res = StateCode.Success

                    if res is StateCode.Success and p.state != state_prev:
                        res = p.state

                    if not analytical and res == StateCode.Success and not np.isclose(p.dt, pdt_prekernels):
                        res = OperationCode.Repeat

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

                # Handle particle time and time loop
                if res in [StateCode.Success, OperationCode.Delete]:
                    # Update time and repeat
                    p.time += p.dt
                    if reset_dt and p.dt == pdt_prekernels:
                        p.dt = dt
                    p.update_next_dt()
                    if analytical:
                        p.dt = np.inf
                    if abs(endtime - p.time) < abs(p.dt):
                        dt_pos = abs(endtime - p.time)
                        reset_dt = True
                    else:
                        dt_pos = abs(p.dt)
                        reset_dt = False

                    sign_end_part = np.sign(endtime - p.time)
                    if res != OperationCode.Delete and not np.isclose(dt_pos, 0) and (sign_end_part == sign_dt):
                        res = StateCode.Evaluate
                    if sign_end_part != sign_dt:
                        dt_pos = 0

                    p.set_state(res)
                    if np.isclose(dt, 0):
                        break
                else:
                    p.set_state(res)
                    # Try again without time update
                    # for var in pset.collection.ptype.variables:
                    for var in ptype.variables:
                        if var.name not in ['dt', 'state']:
                            setattr(p, var.name, p_var_back[var.name])
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

    def remove_deleted(self, pset, output_file, endtime):
        """Utility to remove all particles that signalled deletion"""
        logger.info("KernelAOS::remove_deleted - collect indices ...")
        indices = [i for i, p in enumerate(pset) if p.state == OperationCode.Delete]
        logger.info("KernelAOS::remove_deleted - {} indices collect.".format(len(indices)))
        if len(indices) > 0 and output_file is not None:
            output_file.write(pset, endtime, deleted_only=indices)
        pset.remove_indices(indices)

    def execute(self, pset, endtime, dt, recovery=None, output_file=None, execute_once=False):
        """Execute this Kernel over a ParticleSet for several timesteps"""
        logger.info("KernelAOS::execute() - executing with endtime = {}".format(endtime))
        logger.info("KernelAOS::execute() - resetting the state ...")
        for p in pset:
            p.reset_state()
        logger.info("KernelAOS::execute() - state reset.")

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

        logger.info("KernelAOS::execute() - executing kernel function ...")
        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)
        logger.info("KernelAOS::execute() - kernel function returned.")

        logger.info("KernelAOS::execute() - removing deleted particles ...")
        # Remove all particles that signalled deletion
        self.remove_deleted(pset, output_file=output_file, endtime=endtime)
        logger.info("KernelAOS::execute() - deleted particles removed.")

        # Identify particles that threw errors
        logger.info("KernelAOS::execute() - collecting erroneous particles ...")
        for p in pset:
            logger.info("P({}) - state: {}".format(p.id, p.state))
        error_particles = [p for p in pset if p.state not in [StateCode.Success, StateCode.Evaluate]]
        logger.info("KernelAOS::execute() - {} erroneous particles collected.".format(len(error_particles)))
        # error_particles = [p for p in pset.iterator() if p.state not in [StateCode.Success, StateCode.Evaluate]]

        cleanup_iter = 0
        while len(error_particles) > 0:
            logger.info("KernelAOS::execute() - while-loop - iteration {}".format(cleanup_iter))
            # Apply recovery kernel
            for p in error_particles:
                if p.state == OperationCode.StopExecution:
                    return
                if p.state == OperationCode.Repeat:
                    # p.reset_state()
                    p.set_state(StateCode.Evaluate)
                elif p.state == OperationCode.Delete:
                    pass
                elif p.state in recovery_map:
                    recovery_kernel = recovery_map[p.state]
                    # p.state = StateCode.Success
                    p.set_state(StateCode.Success)
                    recovery_kernel(p, self.fieldset, p.time)
                    if p.state == StateCode.Success:  # (p.isComputed()):
                        # p.reset_state()
                        p.set_state(StateCode.Evaluate)
                else:
                    logger.warning_once('Deleting particle {} because of non-recoverable error'.format(p.id))
                    p.delete()
                    raise NotImplementedError
                logger.info("KernelAOS - while-loop - new pstate of P({}): {}".format(p.id, p.state))

            # Remove all particles that signalled deletion
            logger.info("KernelAOS::execute() - # particles {} - removing deleted particles in clean-up loop {} ...".format(len(pset), cleanup_iter))
            self.remove_deleted(pset, output_file=output_file, endtime=endtime)
            logger.info("KernelAOS::execute() - # particles {} - deleted particles in clean-up loop {} removed.".format(len(pset), cleanup_iter))

            # Execute core loop again to continue interrupted particles
            logger.info("KernelAOS::execute() - executing kernel function in clean-up loop {} ...".format(cleanup_iter))
            if self.ptype.uses_jit:
                self.execute_jit(pset, endtime, dt)
            else:
                self.execute_python(pset, endtime, dt)
            logger.info("KernelAOS::execute() - kernel function in clean-up loop {} returned.".format(cleanup_iter))

            logger.info("KernelAOS::execute() - collecting erroneous particles in clean-up loop {} ...".format(cleanup_iter))
            for p in pset:
                logger.info("P({}) - state: {}".format(p.id, p.state))
            error_particles = [p for p in pset if p.state not in [StateCode.Success, StateCode.Evaluate]]
            logger.info("KernelAOS::execute() - {} erroneous particles in clean-up loop {} collected.".format(len(error_particles), cleanup_iter))
            # error_particles = [p for p in pset.iterator() if p.state not in [StateCode.Success, StateCode.Evaluate]]
            cleanup_iter += 1
