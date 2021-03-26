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

from parcels.kernel.basekernel import BaseKernel
from parcels.compilation.codegenerator import ArrayKernelGenerator as KernelGenerator
from parcels.compilation.codegenerator import LoopGenerator
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
import parcels.rng as ParcelsRandom  # noqa
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode
from parcels.tools.statuscodes import recovery_map as recovery_base_map
from parcels.tools.loggers import logger


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
                 funccode=None, py_ast=None, funcvars=None, c_include="", delete_cfiles=True):
        super(KernelSOA, self).__init__(fieldset=fieldset, ptype=ptype, pyfunc=pyfunc, funcname=funcname, funccode=funccode, py_ast=py_ast, funcvars=funcvars, c_include=c_include, delete_cfiles=delete_cfiles)

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
            if path.isfile(self._c_include):
                with open(self._c_include, 'r') as f:
                    c_include_str = f.read()
            else:
                c_include_str = self._c_include
            self.ccode = loopgen.generate(self.funcname, self.field_args, self.const_args,
                                          kernel_ccode, c_include_str)

            src_file_or_files, self.lib_file, self.log_file = self.get_kernel_compile_files()
            if type(src_file_or_files) in (list, dict, tuple, np.ndarray):
                self.dyn_srcs = src_file_or_files
            else:
                self.src_file = src_file_or_files

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        self.load_fieldset_jit(pset)

        fargs = [byref(f.ctypes_struct) for f in self.field_args.values()]
        fargs += [c_double(f) for f in self.const_args.values()]
        particle_data = byref(pset.ctypes_struct)
        return self._function(c_int(len(pset)), particle_data,
                              c_double(endtime), c_double(dt), *fargs)

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

        for p in pset:
            self.evaluate_particle(p, endtime, sign_dt, dt, analytical=analytical)

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
                    p.reset_state()
                elif p.state == OperationCode.Delete:
                    pass
                elif p.state in recovery_map:
                    recovery_kernel = recovery_map[p.state]
                    p.set_state(StateCode.Success)
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
