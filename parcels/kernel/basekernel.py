import re
import _ctypes
import inspect
import numpy.ctypeslib as npct
from time import time as ostime
from os import path
from os import remove
from sys import platform
from sys import version_info
from ast import FunctionDef
from hashlib import md5
from parcels.tools.loggers import logger
import numpy as np
from numpy import ndarray

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.tools.global_statics import get_cache_dir

# === import just necessary field classes to perform setup checks === #
from parcels.field import Field
from parcels.field import VectorField
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.grid import GridCode
from parcels.field import FieldOutOfBoundError
from parcels.field import FieldOutOfBoundSurfaceError
from parcels.field import TimeExtrapolationError
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode
from parcels.application_kernels.advection import AdvectionRK4_3D
from parcels.application_kernels.advection import AdvectionAnalytical

__all__ = ['BaseKernel']


re_indent = re.compile(r"^(\s+)")


class BaseKernel(object):
    """Base super class for base Kernel objects that encapsulates auto-generated code.

    :arg fieldset: FieldSet object providing the field information (possibly None)
    :arg ptype: PType object for the kernel particle
    :arg pyfunc: (aggregated) Kernel function
    :arg funcname: function name
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """
    _pyfunc = None
    _fieldset = None
    _ptype = None
    funcname = None

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        self._fieldset = fieldset
        self.field_args = None
        self.const_args = None
        self._ptype = ptype
        self._lib = None
        self.delete_cfiles = delete_cfiles
        self._cleanup_files = None
        self._cleanup_lib = None
        self._c_include = c_include

        # Derive meta information from pyfunc, if not given
        self._pyfunc = None
        self.funcname = funcname or pyfunc.__name__
        self.name = "%s%s" % (ptype.name, self.funcname)
        self.ccode = ""
        self.funcvars = funcvars
        self.funccode = funccode
        self.py_ast = py_ast
        self.dyn_srcs = []
        self.static_srcs = []
        self.src_file = None
        self.lib_file = None
        self.log_file = None

        # Generate the kernel function and add the outer loop
        if self._ptype.uses_jit:
            src_file_or_files, self.lib_file, self.log_file = self.get_kernel_compile_files()
            if type(src_file_or_files) in (list, dict, tuple, ndarray):
                self.dyn_srcs = src_file_or_files
            else:
                self.src_file = src_file_or_files

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        try:
            self.remove_lib()
        except:
            pass
        self._fieldset = None
        self.field_args = None
        self.const_args = None
        self.funcvars = None
        self.funccode = None

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
        field_keys = ""
        if self.field_args is not None:
            field_keys = "-".join(
                ["%s:%s" % (name, field.units.__class__.__name__) for name, field in self.field_args.items()])
        key = self.name + self.ptype._cache_key + field_keys + ('TIME:%f' % ostime())
        return md5(key.encode('utf-8')).hexdigest()

    @staticmethod
    def fix_indentation(string):
        """Fix indentation to allow in-lined kernel definitions"""
        lines = string.split('\n')
        indent = re_indent.match(lines[0])
        if indent:
            lines = [line.replace(indent.groups()[0], '', 1) for line in lines]
        return "\n".join(lines)

    def check_fieldsets_in_kernels(self, pyfunc):
        """
        function checks the integrity of the fieldset with the kernels.
        This function is to be called from the derived class when setting up the 'pyfunc'.
        """
        if self.fieldset is not None:
            if pyfunc is AdvectionRK4_3D:
                warning = False
                if isinstance(self._fieldset.W, Field) and self._fieldset.W.creation_log != 'from_nemo' and \
                   self._fieldset.W._scaling_factor is not None and self._fieldset.W._scaling_factor > 0:
                    warning = True
                if type(self._fieldset.W) in [SummedField, NestedField]:
                    for f in self._fieldset.W:
                        if f.creation_log != 'from_nemo' and f._scaling_factor is not None and f._scaling_factor > 0:
                            warning = True
                if warning:
                    logger.warning_once('Note that in AdvectionRK4_3D, vertical velocity is assumed positive towards increasing z.\n'
                                        '  If z increases downward and w is positive upward you can re-orient it downwards by setting fieldset.W.set_scaling_factor(-1.)')
            elif pyfunc is AdvectionAnalytical:
                if self._ptype.uses_jit:
                    raise NotImplementedError('Analytical Advection only works in Scipy mode')
                if self._fieldset.U.interp_method != 'cgrid_velocity':
                    raise NotImplementedError('Analytical Advection only works with C-grids')
                if self._fieldset.U.grid.gtype not in [GridCode.CurvilinearZGrid, GridCode.RectilinearZGrid]:
                    raise NotImplementedError('Analytical Advection only works with Z-grids in the vertical')

    def check_kernel_signature_on_version(self):
        """
        returns numkernelargs
        """
        numkernelargs = 0
        if self._pyfunc is not None:
            if version_info[0] < 3:
                numkernelargs = len(inspect.getargspec(self._pyfunc).args)
            else:
                numkernelargs = len(inspect.getfullargspec(self._pyfunc).args)
        return numkernelargs

    def remove_lib(self):
        if self._lib is not None:
            BaseKernel.cleanup_unload_lib(self._lib)
            del self._lib
            self._lib = None

        all_files_array = []
        if self.src_file is None:
            if self.dyn_srcs is not None:
                [all_files_array.append(fpath) for fpath in self.dyn_srcs]
        else:
            if self.src_file is not None:
                all_files_array.append(self.src_file)
        if self.log_file is not None:
            all_files_array.append(self.log_file)
        if self.lib_file is not None and all_files_array is not None and self.delete_cfiles is not None:
            BaseKernel.cleanup_remove_files(self.lib_file, all_files_array, self.delete_cfiles)

        # If file already exists, pull new names. This is necessary on a Windows machine, because
        # Python's ctype does not deal in any sort of manner well with dynamic linked libraries on this OS.
        if self._ptype.uses_jit:
            src_file_or_files, self.lib_file, self.log_file = self.get_kernel_compile_files()
            if type(src_file_or_files) in (list, dict, tuple, ndarray):
                self.dyn_srcs = src_file_or_files
            else:
                self.src_file = src_file_or_files

    def get_kernel_compile_files(self):
        """
        Returns the correct src_file, lib_file, log_file for this kernel
        """
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            cache_name = self._cache_key  # only required here because loading is done by Kernel class instead of Compiler class
            dyn_dir = get_cache_dir() if mpi_rank == 0 else None
            dyn_dir = mpi_comm.bcast(dyn_dir, root=0)
            basename = cache_name if mpi_rank == 0 else None
            basename = mpi_comm.bcast(basename, root=0)
            basename = basename + "_%d" % mpi_rank
        else:
            cache_name = self._cache_key  # only required here because loading is done by Kernel class instead of Compiler class
            dyn_dir = get_cache_dir()
            basename = "%s_0" % cache_name
        lib_path = "lib" + basename
        src_file_or_files = None
        if type(basename) in (list, dict, tuple, ndarray):
            src_file_or_files = ["", ] * len(basename)
            for i, src_file in enumerate(basename):
                src_file_or_files[i] = "%s.c" % path.join(dyn_dir, src_file)
        else:
            src_file_or_files = "%s.c" % path.join(dyn_dir, basename)
        lib_file = "%s.%s" % (path.join(dyn_dir, lib_path), 'dll' if platform == 'win32' else 'so')
        log_file = "%s.log" % path.join(dyn_dir, basename)
        return src_file_or_files, lib_file, log_file

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        all_files_array = []
        if self.src_file is None:
            if self.dyn_srcs is not None:
                for dyn_src in self.dyn_srcs:
                    with open(dyn_src, 'w') as f:
                        f.write(self.ccode)
                    all_files_array.append(dyn_src)
                compiler.compile(self.dyn_srcs, self.lib_file, self.log_file)
        else:
            if self.src_file is not None:
                with open(self.src_file, 'w') as f:
                    f.write(self.ccode)
                if self.src_file is not None:
                    all_files_array.append(self.src_file)
                compiler.compile(self.src_file, self.lib_file, self.log_file)
        if len(all_files_array) > 0:
            logger.info("Compiled %s ==> %s" % (self.name, self.lib_file))
            if self.log_file is not None:
                all_files_array.append(self.log_file)

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, '.')
        self._function = self._lib.particle_loop

    def merge(self, kernel, kclass):
        funcname = self.funcname + kernel.funcname
        func_ast = None
        if self.py_ast is not None:
            func_ast = FunctionDef(name=funcname, args=self.py_ast.args, body=self.py_ast.body + kernel.py_ast.body,
                                   decorator_list=[], lineno=1, col_offset=0)
        delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
        return kclass(self.fieldset, self.ptype, pyfunc=None,
                      funcname=funcname, funccode=self.funccode + kernel.funccode,
                      py_ast=func_ast, funcvars=self.funcvars + kernel.funcvars,
                      c_include=self._c_include + kernel.c_include,
                      delete_cfiles=delete_cfiles)

    def __add__(self, kernel):
        if not isinstance(kernel, BaseKernel):
            kernel = BaseKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, BaseKernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, BaseKernel):
            kernel = BaseKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, BaseKernel)

    @staticmethod
    def cleanup_remove_files(lib_file, all_files_array, delete_cfiles):
        if lib_file is not None:
            if path.isfile(lib_file):  # and delete_cfiles
                [remove(s) for s in [lib_file, ] if path is not None and path.exists(s)]
            if delete_cfiles and len(all_files_array) > 0:
                [remove(s) for s in all_files_array if path is not None and path.exists(s)]

    @staticmethod
    def cleanup_unload_lib(lib):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        if lib is not None:
            try:
                _ctypes.FreeLibrary(lib._handle) if platform == 'win32' else _ctypes.dlclose(lib._handle)
            except:
                logger.warning_once("compiled library already freed.")

    def remove_deleted(self, pset, output_file, endtime):
        """
        Utility to remove all particles that signalled deletion.

        This version is generally applicable to all structures and collections
        """
        indices = [i for i, p in enumerate(pset) if p.state == OperationCode.Delete]
        if len(indices) > 0 and output_file is not None:
            output_file.write(pset, endtime, deleted_only=indices)
        pset.remove_indices(indices)

    def load_fieldset_jit(self, pset):
        """
        Updates the loaded fields of pset's fieldset according to the chunk information within their grids
        """
        if pset.fieldset is not None:
            for g in pset.fieldset.gridset.grids:
                g.cstruct = None  # This force to point newly the grids from Python to C
            # Make a copy of the transposed array to enforce
            # C-contiguous memory layout for JIT mode.
            for f in pset.fieldset.get_fields():
                if type(f) in [VectorField, NestedField, SummedField]:
                    continue
                if f.data.dtype != np.float32:
                    raise RuntimeError('Field %s data needs to be float32 in JIT mode' % f.name)
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

    def evaluate_particle(self, p, endtime, sign_dt, dt, analytical=False):
        """
        Execute the kernel evaluation of for an individual particle.
        :arg p: object of (sub-)type (ScipyParticle, JITParticle) or (sub-)type of BaseParticleAccessor
        :arg fieldset: fieldset of the containing ParticleSet (e.g. pset.fieldset)
        :arg analytical: flag indicating the analytical advector or an iterative advection
        :arg endtime: endtime of this overall kernel evaluation step
        :arg dt: computational integration timestep
        """
        variables = self._ptype.variables
        # back up variables in case of OperationCode.Repeat
        p_var_back = {}
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
            return p

        while p.state in [StateCode.Evaluate, OperationCode.Repeat] or np.isclose(dt, 0):
            for var in variables:
                p_var_back[var.name] = getattr(p, var.name)
            try:
                pdt_prekernels = sign_dt * dt_pos
                p.dt = pdt_prekernels
                state_prev = p.state
                res = self._pyfunc(p, self._fieldset, p.time)
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
                for var in variables:
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
        return p

    def execute_jit(self, pset, endtime, dt):
        pass

    def execute_python(self, pset, endtime, dt):
        pass

    def execute(self, pset, endtime, dt, recovery=None, output_file=None, execute_once=False):
        pass
