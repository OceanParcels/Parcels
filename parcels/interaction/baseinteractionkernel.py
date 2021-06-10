import inspect
from sys import version_info

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.kernel.basekernel import BaseKernel

__all__ = ['BaseInteractionKernel']


class BaseInteractionKernel(BaseKernel):
    """Base super class for Interaction Kernel objects that encapsulates
    auto-generated code.

    InteractionKernels do not implement ways to catch or recover from
    errors caused during execution of the kernel function(s).
    It is strongly recommended not to sample from fields inside an
    InteractionKernel.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None,
                 funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
            raise NotImplementedError("InteractionKernels are not supported in an MPI environment. Please run your simulation outside MPI.")

        if pyfunc is not None:
            if isinstance(pyfunc, list):
                funcname = ''.join([func.__name__ for func in pyfunc])
            else:
                funcname = pyfunc.__name__

        super(BaseInteractionKernel, self).__init__(
            fieldset=fieldset, ptype=ptype, pyfunc=pyfunc, funcname=funcname,
            funccode=funccode, py_ast=py_ast, funcvars=funcvars,
            c_include=c_include, delete_cfiles=delete_cfiles)

        if pyfunc is not None:
            if isinstance(pyfunc, list):
                self._pyfunc = pyfunc
            else:
                self._pyfunc = [pyfunc]

        if self._ptype.uses_jit:
            raise NotImplementedError("JIT mode is not supported for"
                                      " InteractionKernels. Please run your"
                                      " simulation in SciPy mode.")

    @property
    def _cache_key(self):
        raise NotImplementedError

    @staticmethod
    def fix_indentation(string):
        raise NotImplementedError

    def check_fieldsets_in_kernels(self, pyfunc):
        # Currently, the implemented interaction kernels do not impose
        # any requirements on the fieldset
        pass

    def check_kernel_signature_on_version(self):
        """
        returns numkernelargs
        Adaptation of this method in the BaseKernel that works with
        lists of functions.
        """
        numkernelargs = []
        if self._pyfunc is not None and isinstance(self._pyfunc, list):
            for func in self._pyfunc:
                if version_info[0] < 3:
                    numkernelargs.append(
                        len(inspect.getargspec(func).args)
                    )
                else:
                    numkernelargs.append(
                        len(inspect.getfullargspec(func).args)
                    )
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
        return kclass(self._fieldset, self._ptype, pyfunc=pyfunc,
                      funcname=funcname)

    def __add__(self, kernel):
        if not isinstance(kernel, BaseInteractionKernel):
            kernel = BaseInteractionKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, BaseInteractionKernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, BaseInteractionKernel):
            kernel = BaseInteractionKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, BaseInteractionKernel)

    @staticmethod
    def cleanup_remove_files(lib_file, all_files_array, delete_cfiles):
        raise NotImplementedError

    @staticmethod
    def cleanup_unload_lib(lib):
        raise NotImplementedError

    def execute_jit(self, pset, endtime, dt):
        raise NotImplementedError("JIT mode is not supported for"
                                  " InteractionKernels. Please run your"
                                  " simulation in SciPy mode.")
