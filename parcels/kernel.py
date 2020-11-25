import _ctypes
import inspect
import math  # noqa
import random  # noqa
import re
import time
from ast import FunctionDef
from ast import parse
from copy import deepcopy
from ctypes import byref
from ctypes import c_double
from ctypes import c_int
from hashlib import md5
from os import path
from os import remove
from sys import platform
from sys import version_info
from weakref import finalize

import numpy as np
import numpy.ctypeslib as npct
try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.codegenerator import KernelGenerator
from parcels.codegenerator import LoopGenerator
from parcels.compiler import get_cache_dir
from parcels.field import Field
from parcels.field import FieldOutOfBoundError
from parcels.field import FieldOutOfBoundSurfaceError
from parcels.field import TimeExtrapolationError
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
from parcels.grid import GridCode
from parcels.kernels.advection import AdvectionRK4_3D
from parcels.kernels.advection import AdvectionAnalytical
import parcels.rng as ParcelsRandom  # noqa
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode
from parcels.tools.statuscodes import recovery_map as recovery_base_map
from parcels.tools.loggers import logger


__all__ = ['Kernel']


re_indent = re.compile(r"^(\s+)")


def fix_indentation(string):
    """Fix indentation to allow in-lined kernel definitions"""
    lines = string.split('\n')
    indent = re_indent.match(lines[0])
    if indent:
        lines = [line.replace(indent.groups()[0], '', 1) for line in lines]
    return "\n".join(lines)


class Kernel(object):
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
        self.fieldset = fieldset
        self.ptype = ptype
        self._lib = None
        self.delete_cfiles = delete_cfiles
        self._cleanup_files = None
        self._cleanup_lib = None
        self.c_include = c_include

        # Derive meta information from pyfunc, if not given
        self.funcname = funcname or pyfunc.__name__
        if pyfunc is AdvectionRK4_3D:
            warning = False
            if isinstance(fieldset.W, Field) and fieldset.W.creation_log != 'from_nemo' and \
               fieldset.W._scaling_factor is not None and fieldset.W._scaling_factor > 0:
                warning = True
            if type(fieldset.W) in [SummedField, NestedField]:
                for f in fieldset.W:
                    if f.creation_log != 'from_nemo' and f._scaling_factor is not None and f._scaling_factor > 0:
                        warning = True
            if warning:
                logger.warning_once('Note that in AdvectionRK4_3D, vertical velocity is assumed positive towards increasing z.\n'
                                    '         If z increases downward and w is positive upward you can re-orient it downwards by setting fieldset.W.set_scaling_factor(-1.)')
        elif pyfunc is AdvectionAnalytical:
            if ptype.uses_jit:
                raise NotImplementedError('Analytical Advection only works in Scipy mode')
            if fieldset.U.interp_method != 'cgrid_velocity':
                raise NotImplementedError('Analytical Advection only works with C-grids')
            if fieldset.U.grid.gtype not in [GridCode.CurvilinearZGrid, GridCode.RectilinearZGrid]:
                raise NotImplementedError('Analytical Advection only works with Z-grids in the vertical')

        if funcvars is not None:
            self.funcvars = funcvars
        elif hasattr(pyfunc, '__code__'):
            self.funcvars = list(pyfunc.__code__.co_varnames)
        else:
            self.funcvars = None
        self.funccode = funccode or inspect.getsource(pyfunc.__code__)
        # Parse AST if it is not provided explicitly
        self.py_ast = py_ast or parse(fix_indentation(self.funccode)).body[0]
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
            self.pyfunc = user_ctx[self.funcname]
        else:
            self.pyfunc = pyfunc

        if version_info[0] < 3:
            numkernelargs = len(inspect.getargspec(self.pyfunc).args)
        else:
            numkernelargs = len(inspect.getfullargspec(self.pyfunc).args)

        assert numkernelargs == 3, \
            'Since Parcels v2.0, kernels do only take 3 arguments: particle, fieldset, time !! AND !! Argument order in field interpolation is time, depth, lat, lon.'

        self.name = "%s%s" % (ptype.name, self.funcname)

        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            kernelgen = KernelGenerator(fieldset, ptype)
            kernel_ccode = kernelgen.generate(deepcopy(self.py_ast),
                                              self.funcvars)
            self.field_args = kernelgen.field_args
            self.vector_field_args = kernelgen.vector_field_args
            fieldset = self.fieldset
            for f in self.vector_field_args.values():
                Wname = f.W.ccode_name if f.W else 'not_defined'
                for sF_name, sF_component in zip([f.U.ccode_name, f.V.ccode_name, Wname], ['U', 'V', 'W']):
                    if sF_name not in self.field_args:
                        if sF_name != 'not_defined':
                            self.field_args[sF_name] = getattr(f, sF_component)
            self.const_args = kernelgen.const_args
            loopgen = LoopGenerator(fieldset, ptype)
            if path.isfile(c_include):
                with open(c_include, 'r') as f:
                    c_include_str = f.read()
            else:
                c_include_str = c_include
            self.ccode = loopgen.generate(self.funcname, self.field_args, self.const_args,
                                          kernel_ccode, c_include_str)
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                basename = path.join(get_cache_dir(), self._cache_key) if mpi_rank == 0 else None
                basename = mpi_comm.bcast(basename, root=0)
                basename = basename + "_%d" % mpi_rank
            else:
                basename = path.join(get_cache_dir(), "%s_0" % self._cache_key)

            self.src_file = "%s.c" % basename
            self.lib_file = "%s.%s" % (basename, 'dll' if platform == 'win32' else 'so')
            self.log_file = "%s.log" % basename

    @property
    def _cache_key(self):
        field_keys = "-".join(["%s:%s" % (name, field.units.__class__.__name__)
                               for name, field in self.field_args.items()])
        key = self.name + self.ptype._cache_key + field_keys + ('TIME:%f' % time.time())
        return md5(key.encode('utf-8')).hexdigest()

    def remove_lib(self):
        # Unload the currently loaded dynamic linked library to be secure
        if self._lib is not None:
            _ctypes.FreeLibrary(self._lib._handle) if platform == 'win32' else _ctypes.dlclose(self._lib._handle)
            del self._lib
            self._lib = None

        # deactivate the cleanup finalizers for the current set of files
        if self._cleanup_files is not None:
            self._cleanup_files.detach()

        if self._cleanup_lib is not None:
            self._cleanup_lib.detach()

        # If file already exists, pull new names. This is necessary on a Windows machine, because
        # Python's ctype does not deal in any sort of manner well with dynamic linked libraries on this OS.
        if path.isfile(self.lib_file):
            [remove(s) for s in [self.src_file, self.lib_file, self.log_file]]
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                basename = path.join(get_cache_dir(), self._cache_key) if mpi_rank == 0 else None
                basename = mpi_comm.bcast(basename, root=0)
                basename = basename + "_%d" % mpi_rank
            else:
                basename = path.join(get_cache_dir(), "%s_0" % self._cache_key)

            self.src_file = "%s.c" % basename
            self.lib_file = "%s.%s" % (basename, 'dll' if platform == 'win32' else 'so')
            self.log_file = "%s.log" % basename

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        with open(self.src_file, 'w') as f:
            f.write(self.ccode)
        compiler.compile(self.src_file, self.lib_file, self.log_file)
        logger.info("Compiled %s ==> %s" % (self.name, self.lib_file))
        self._cleanup_files = finalize(self, cleanup_remove_files, self.delete_cfiles, self.src_file, self.lib_file, self.log_file)

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, '.')
        self._function = self._lib.particle_loop
        self._cleanup_lib = finalize(self, cleanup_unload_lib, self._lib)

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        if len(pset) > 0 and pset.collection._data['xi'].ndim == 2 and pset.fieldset is not None:
            assert pset.fieldset.gridset.size == pset.collection._data['xi'].shape[1], \
                'FieldSet has different number of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'

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
                g.load_chunk = np.where(g.load_chunk == 1, 2, g.load_chunk)
                if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
                    if not g.load_chunk.flags.c_contiguous:
                        g.load_chunk = g.load_chunk.copy()
                if not g.depth.flags.c_contiguous:
                    g.depth = g.depth.copy()
                if not g.lon.flags.c_contiguous:
                    g.lon = g.lon.copy()
                if not g.lat.flags.c_contiguous:
                    g.lat = g.lat.copy()

        fargs = [byref(f.ctypes_struct) for f in self.field_args.values()]
        fargs += [c_double(f) for f in self.const_args.values()]
        particle_data = byref(pset.ctypes_struct)
        return self._function(c_int(len(pset)), particle_data,
                              c_double(endtime), c_double(dt), *fargs)

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        if 'AdvectionAnalytical' in self.pyfunc.__name__:
            analytical = True
            if not np.isinf(dt):
                logger.warning_once('dt is not used in AnalyticalAdvection, so is set to np.inf')
            dt = np.inf
        else:
            analytical = False

        # back up variables in case of OperationCode.Repeat
        p_var_back = {}

        if self.fieldset is not None:
            for f in self.fieldset.get_fields():
                if type(f) in [VectorField, NestedField, SummedField]:
                    continue
                f.data = np.array(f.data)

        for p in pset:
            # Don't execute particles that aren't started yet
            sign_end_part = np.sign(endtime - p.time)
            # Compute min/max dt for first timestep
            dt_pos = min(abs(p.dt), abs(endtime - p.time))

            # ==== numerically stable; also making sure that continuously-recovered particles do end successfully,
            # as they fulfil the condition here on entering at the final calculation here. ==== #
            if ((sign_end_part != sign_dt) or np.isclose(dt_pos, 0)) and not np.isclose(dt, 0):
                if abs(p.time) >= abs(endtime):
                    p.set_state(StateCode.Success)
                continue

            while p.state in [StateCode.Evaluate, OperationCode.Repeat] or np.isclose(dt, 0):

                for var in pset.collection.ptype.variables:
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
                    p.update_next_dt()
                    if analytical:
                        p.dt = np.inf
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

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
                    for var in pset.collection.ptype.variables:
                        if var.name not in ['dt', 'state']:
                            setattr(p, var.name, p_var_back[var.name])
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

                    sign_end_part = np.sign(endtime - p.time)
                    if sign_end_part != sign_dt:
                        dt_pos = 0
                    break

    def execute(self, pset, endtime, dt, recovery=None, output_file=None, execute_once=False):
        """Execute this Kernel over a ParticleSet for several timesteps"""
        for p in pset:
            p.set_state(StateCode.Evaluate)

        if abs(dt) < 1e-6 and not execute_once:
            logger.warning_once("'dt' is too small, causing numerical accuracy limit problems. Please chose a higher 'dt' and rather scale the 'time' axis of the field accordingly. (related issue #762)")

        def remove_deleted(pset):
            """Utility to remove all particles that signalled deletion"""
            indices = pset.collection._data['state'] == OperationCode.Delete
            if np.count_nonzero(indices) > 0 and output_file is not None:
                output_file.write(pset, endtime, deleted_only=indices)
            pset.remove_booleanvector(indices)

        if recovery is None:
            recovery = {}
        elif ErrorCode.ErrorOutOfBounds in recovery and ErrorCode.ErrorThroughSurface not in recovery:
            recovery[ErrorCode.ErrorThroughSurface] = recovery[ErrorCode.ErrorOutOfBounds]
        recovery_map = recovery_base_map.copy()
        recovery_map.update(recovery)

        if pset.fieldset is not None:
            for g in pset.fieldset.gridset.grids:
                if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
                    g.load_chunk = np.where(g.load_chunk == 2, 3, g.load_chunk)

        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)

        # Remove all particles that signalled deletion
        remove_deleted(pset)

        # Identify particles that threw errors
        n_error = -1
        while n_error != 0:
            error_pset = pset.error_particles
            # Apply recovery kernel
            # Effectively loop over all particles that don't have sucess of
            # evaluate codes.
            n_error = 0
            for particle in error_pset:
                if particle.state == OperationCode.StopExecution:
                    return
                if particle.state == OperationCode.Repeat:
                    particle.set_state(StateCode.Evaluate)
                elif particle.state in recovery_map:
                    recovery_kernel = recovery_map[particle.state]
                    particle.set_state(StateCode.Success)
                    recovery_kernel(particle, self.fieldset, particle.time)
                    if particle.state == StateCode.Success:
                        particle.set_state(StateCode.Evaluate)
                else:
                    logger.warning_once(
                        'Deleting particle because of bug in #749 and #737')
                    particle.delete()
                n_error += 1

            if n_error > 0:
                remove_deleted(pset)
                # Execute core loop again to continue interrupted particles
                if self.ptype.uses_jit:
                    self.execute_jit(pset, endtime, dt)
                else:
                    self.execute_python(pset, endtime, dt)

    def merge(self, kernel):
        funcname = self.funcname + kernel.funcname
        func_ast = FunctionDef(name=funcname, args=self.py_ast.args,
                               body=self.py_ast.body + kernel.py_ast.body,
                               decorator_list=[], lineno=1, col_offset=0)
        delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
        return Kernel(self.fieldset, self.ptype, pyfunc=None,
                      funcname=funcname, funccode=self.funccode + kernel.funccode,
                      py_ast=func_ast, funcvars=self.funcvars + kernel.funcvars,
                      c_include=self.c_include + kernel.c_include,
                      delete_cfiles=delete_cfiles)

    def __add__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self)


def cleanup_remove_files(delete_cfiles, src_file, lib_file, log_file):
    if path.isfile(lib_file) and delete_cfiles:
        [remove(s) for s in [src_file, lib_file, log_file]]


def cleanup_unload_lib(lib):
    # Clean-up the in-memory dynamic linked libraries.
    # This is not really necessary, as these programs are not that large, but with the new random
    # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
    if lib is not None:
        _ctypes.FreeLibrary(lib._handle) if platform == 'win32' else _ctypes.dlclose(lib._handle)
