import numpy as np
from ctypes import byref, c_double, c_int  # NOQA

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode  # NOQA
from parcels.kernel.basekernel import BaseKernel
from parcels.tools.loggers import logger  # NOQA
from parcels.tools.performance_logger import TimingLog

__all__ = ['BaseBenchmarkKernel']


class BaseBenchmarkKernel(BaseKernel):
    perform_benchmark = False

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, funcvars=None, py_ast=None,
                 c_include="", delete_cfiles=True, use_benchmark=False):
        super(BaseBenchmarkKernel, self).__init__(fieldset, ptype, pyfunc, funcname, funccode, funcvars, py_ast,
                                                  c_include, delete_cfiles)
        self.perform_benchmark = use_benchmark
        self._compute_timings = TimingLog()
        self._io_timings = TimingLog()
        self._mem_io_timings = TimingLog()

    def __del__(self):
        super(BaseBenchmarkKernel, self).__del__()

    @property
    def io_timings(self):
        return self._io_timings

    @property
    def mem_io_timings(self):
        return self._mem_io_timings

    @property
    def compute_timings(self):
        return self._compute_timings

    def remove_deleted(self, pset, output_file, endtime):
        """
        Utility to remove all particles that signalled deletion

        This version is generally applicable to all structures and collections
        :arg pset: host ParticleSet
        :arg output_file: instance of ParticleFile object of the host ParticleSet where deleted objects are to be written to on deletion
        :arg endtime: timestamp at which the particles are to be deleted
        """
        super(BaseBenchmarkKernel, self).remove_deleted(pset=pset, output_file=output_file, endtime=endtime)

    def benchmark_remove_deleted(self, pset, output_file, endtime):
        self._mem_io_timings.start_timing()
        self.remove_deleted(pset=pset, output_file=output_file, endtime=endtime)
        self._mem_io_timings.stop_timing()
        self._mem_io_timings.accumulate_timing()
        # self._mem_io_timings.advance_iteration()

    def load_fieldset_jit(self, pset):
        """
        Updates the loaded fields of pset's fieldset according to the chunk information within their grids
        :arg pset: host ParticleSet
        """
        if pset.fieldset is not None:
            self._io_timings.start_timing()
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
            self._io_timings.stop_timing()
            self._io_timings.accumulate_timing()

            self._mem_io_timings.start_timing()
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
            self._mem_io_timings.stop_timing()
            self._mem_io_timings.accumulate_timing()

    def execute_jit(self, pset, endtime, dt):
        """
        Invokes JIT engine to perform the core update loop
        :arg pset: particle set to calculate
        :arg endtime: timestamp to calculate
        :arg dt: delta-t to be calculated
        """
        raise NotImplementedError

    def execute_python(self, pset, endtime, dt):
        """
        Performs the core update loop via Python
        :arg pset: particle set to calculate
        :arg endtime: timestamp to calculate
        :arg dt: delta-t to be calculated
        """
        raise NotImplementedError

    def execute(self, pset, endtime, dt, recovery=None, output_file=None, execute_once=False):
        """
        Execute this Kernel over a ParticleSet for several timesteps
        :arg pset: host ParticleSet
        :arg endtime: endtime of this overall kernel evaluation step
        :arg dt: computational integration timestep
        :arg recovery: dict of recovery code -> recovery function
        :arg output_file: instance of ParticleFile object of the host ParticleSet where deleted objects are to be written to on deletion
        :arg execute_once: boolean, telling if to execute once (True) or computing the kernel iteratively
        """
        raise NotImplementedError
