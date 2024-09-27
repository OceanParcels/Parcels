import inspect
import warnings
from collections import defaultdict

import numpy as np

from parcels._compat import MPI
from parcels.field import NestedField, VectorField
from parcels.kernel import BaseKernel
from parcels.tools.statuscodes import StatusCode

__all__ = ["InteractionKernel"]


class InteractionKernel(BaseKernel):
    """InteractionKernel object that encapsulates auto-generated code.

    InteractionKernels do not implement ways to catch or recover from
    errors caused during execution of the kernel function(s).
    It is strongly recommended not to sample from fields inside an
    InteractionKernel.
    """

    def __init__(
        self,
        fieldset,
        ptype,
        pyfunc=None,
        funcname=None,
        funccode=None,
        py_ast=None,
        funcvars=None,
        c_include="",
        delete_cfiles: bool = True,
    ):
        if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
            raise NotImplementedError(
                "InteractionKernels are not supported in an MPI environment. Please run your simulation outside MPI."
            )

        if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
            raise NotImplementedError(
                "InteractionKernels are not supported in an MPI environment. Please run your simulation outside MPI."
            )

        if pyfunc is not None:
            if isinstance(pyfunc, list):
                funcname = "".join([func.__name__ for func in pyfunc])
            else:
                funcname = pyfunc.__name__

        super().__init__(
            fieldset=fieldset,
            ptype=ptype,
            pyfunc=pyfunc,
            funcname=funcname,
            funccode=funccode,
            py_ast=py_ast,
            funcvars=funcvars,
            c_include=c_include,
            delete_cfiles=delete_cfiles,
        )

        if pyfunc is not None:
            if isinstance(pyfunc, list):
                funcname = "".join([func.__name__ for func in pyfunc])
            else:
                funcname = pyfunc.__name__

        if pyfunc is not None:
            if isinstance(pyfunc, list):
                self._pyfunc = pyfunc
            else:
                self._pyfunc = [pyfunc]

        if self._ptype.uses_jit:
            raise NotImplementedError(
                "JIT mode is not supported for InteractionKernels. Please run your simulation in SciPy mode."
            )

        for func in self._pyfunc:
            self.check_fieldsets_in_kernels(func)

        numkernelargs = self.check_kernel_signature_on_version()

        assert numkernelargs[0] == 5 and numkernelargs.count(numkernelargs[0]) == len(
            numkernelargs
        ), "Interactionkernels take exactly 5 arguments: particle, fieldset, time, neighbors, mutator"

        # At this time, JIT mode is not supported for InteractionKernels,
        # so there is no need for any further "processing" of pyfunc's.

    @property
    def _cache_key(self):
        raise NotImplementedError

    def check_fieldsets_in_kernels(self, pyfunc):
        # Currently, the implemented interaction kernels do not impose
        # any requirements on the fieldset
        pass

    def check_kernel_signature_on_version(self):
        """
        Returns numkernelargs.
        Adaptation of this method in the BaseKernel that works with
        lists of functions.
        """
        numkernelargs = []
        if self._pyfunc is not None and isinstance(self._pyfunc, list):
            for func in self._pyfunc:
                numkernelargs.append(len(inspect.getfullargspec(func).args))
        return numkernelargs

    def remove_lib(self):
        # Currently, no libs are generated/linked, so nothing has to be
        # removed
        pass

    def get_kernel_compile_files(self):
        raise NotImplementedError

    def compile(self, compiler):
        raise NotImplementedError

    def load_lib(self):
        raise NotImplementedError

    def merge(self, kernel, kclass):
        assert self.__class__ == kernel.__class__
        funcname = self.funcname + kernel.funcname
        # delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
        pyfunc = self._pyfunc + kernel._pyfunc
        return kclass(self._fieldset, self._ptype, pyfunc=pyfunc, funcname=funcname)

    def __add__(self, kernel):
        if not isinstance(kernel, InteractionKernel):
            kernel = InteractionKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, InteractionKernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, InteractionKernel):
            kernel = InteractionKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, InteractionKernel)

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.)
        super().__del__()

    @staticmethod
    def cleanup_remove_files(lib_file, all_files_array, delete_cfiles):
        raise NotImplementedError

    @staticmethod
    def cleanup_unload_lib(lib):
        raise NotImplementedError

    def execute_jit(self, pset, endtime, dt):
        raise NotImplementedError(
            "JIT mode is not supported for InteractionKernels. Please run your simulation in SciPy mode."
        )

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python.

        InteractionKernels do not implement ways to catch or recover from
        errors caused during execution of the kernel function(s).
        It is strongly recommended not to sample from fields inside an
        InteractionKernel.
        """
        if self.fieldset is not None:
            for f in self.fieldset.get_fields():
                if isinstance(f, (VectorField, NestedField)):
                    continue
                f.data = np.array(f.data)

        reset_particle_idx = []
        for pyfunc in self._pyfunc:
            pset._compute_neighbor_tree(endtime, dt)
            active_idx = pset._active_particle_idx

            mutator = defaultdict(lambda: [])

            # Loop only over particles that are in a positive state and have started.
            for particle_idx in active_idx:
                p = pset[particle_idx]
                # Don't use particles that are not started.
                if (endtime - p.time) / dt <= -1e-7:
                    continue
                elif (endtime - p.time) / dt < 1:
                    p.dt = endtime - p.time
                    reset_particle_idx.append(particle_idx)

                neighbors = pset._neighbors_by_index(particle_idx)
                try:
                    res = pyfunc(p, pset.fieldset, p.time, neighbors, mutator)
                except Exception as e:
                    res = StatusCode.Error
                    p.exception = e

                # InteractionKernels do not implement a way to recover
                # from errors.
                if res != StatusCode.Success:
                    warnings.warn(
                        "Some InteractionKernel was not completed succesfully, likely because a Particle threw an error that was not captured.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            for particle_idx in active_idx:
                p = pset[particle_idx]
                try:
                    for mutator_func, args in mutator[p.id]:
                        mutator_func(p, *args)
                except KeyError:
                    pass
            for particle_idx in reset_particle_idx:
                pset[particle_idx].dt = dt

    def execute(self, pset, endtime, dt, output_file=None):
        """Execute this Kernel over a ParticleSet for several timesteps.

        InteractionKernels do not implement ways to catch or recover from
        errors caused during execution of the kernel function(s).
        It is strongly recommended not to sample from fields inside an
        InteractionKernel.
        """
        pset.particledata.state[:] = StatusCode.Evaluate

        if abs(dt) < 1e-6:
            warnings.warn(
                "'dt' is too small, causing numerical accuracy limit problems. Please chose a higher 'dt' and rather scale the 'time' axis of the field accordingly. (related issue #762)",
                RuntimeWarning,
                stacklevel=2,
            )

        if pset.fieldset is not None:
            for g in pset.fieldset.gridset.grids:
                if len(g._load_chunk) > g._chunk_not_loaded:  # not the case if a field in not called in the kernel
                    g._load_chunk = np.where(
                        g._load_chunk == g._chunk_loaded_touched, g._chunk_deprecated, g._load_chunk
                    )

        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            # This should never happen, as it is already checked in the
            # initialization.
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)

        # Remove all particles that signalled deletion
        self.remove_deleted(pset)  # Generalizable version!

        # Identify particles that threw errors
        n_error = pset._num_error_particles

        while n_error > 0:
            error_pset = pset._error_particles
            # Check for StatusCodes
            for p in error_pset:
                if p.state == StatusCode.StopExecution:
                    return
                if p.state == StatusCode.Repeat:
                    p.state = StatusCode.Evaluate
                elif p.state == StatusCode.Delete:
                    pass
                else:
                    warnings.warn(
                        f"Deleting particle {p.id} because of non-recoverable error",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    p.delete()

            # Remove all particles that signalled deletion
            self.remove_deleted(pset)  # Generalizable version!

            # Execute core loop again to continue interrupted particles
            if self.ptype.uses_jit:
                self.execute_jit(pset, endtime, dt)
            else:
                self.execute_python(pset, endtime, dt)

            n_error = pset._num_error_particles
