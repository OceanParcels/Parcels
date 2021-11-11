import numpy as np
# import numpy.ctypeslib as npct
try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.tools.loggers import logger
from parcels.tools.performance_logger import TimingLog

__all__ = ['BenchmarkKernel']

class BenchmarkKernel:

    def __init__(self):
        self._compute_timings = TimingLog()
        self._io_timings = TimingLog()
        self._mem_io_timings = TimingLog()

    @property
    def io_timings(self):
        return self._io_timings

    @property
    def mem_io_timings(self):
        return self._mem_io_timings

    @property
    def compute_timings(self):
        return self._compute_timings

