import time as time_module
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None

class TimingLog():
    stime = 0
    etime = 0
    mtime = 0
    _samples = []
    _times_steps = []
    _iter = 0

    def __init__(self):
        self.stime = 0
        self.etime = 0
        self.mtime = 0
        self._samples = []
        self._times_steps = []
        self._iter = 0

    @property
    def timing(self):
        return self._times_steps

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._samples)

    def get_values(self):
        return self._times_steps

    def get_value(self, index):
        return self._times_steps[index]

    def start_timing(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                # self.stime = MPI.Wtime()
                # self.stime = time_module.perf_counter()
                self.stime = time_module.process_time()
        else:
            self.stime = time_module.process_time()

    def stop_timing(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                # self.etime = MPI.Wtime()
                # self.etime = time_module.perf_counter()
                self.etime = time_module.process_time()
        else:
            self.etime = time_module.process_time()

    def accumulate_timing(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                self.mtime += (self.etime-self.stime)
            else:
                self.mtime = 0
        else:
            self.mtime += (self.etime-self.stime)

    def advance_iteration(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                self._times_steps.append(self.mtime)
                self._samples.append(self._iter)
                self._iter += 1
            self.mtime = 0
        else:
            self._times_steps.append(self.mtime)
            self._samples.append(self._iter)
            self._iter += 1
            self.mtime = 0

    def add_aux_measure(self, value):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                self.mtime += value
            else:
                self.mtime += 0
        else:
            self.mtime += value

    def sum(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                result = np.array(self._times_steps).sum()
                return result
            else:
                return 0
        else:
            result = np.array(self._times_steps).sum()

    def reset(self):
        if self._times_steps is not None and len(self._times_steps) > 0:
            del self._times_steps[:]
        if self._samples is not None and len(self._samples) > 0:
            del self._samples[:]
        self.stime = 0
        self.etime = 0
        self.mtime = 0
        self._samples = []
        self._times_steps = []
        self._iter = 0


class ParamLogging():
    _samples = []
    _params = []
    _iter = 0

    def __init__(self):
        self._samples = []
        self._params = []
        self._iter = 0

    @property
    def samples(self):
        return self._samples

    @property
    def params(self):
        return self._params

    def get_params(self):
        return self._params

    def get_param(self, index):
        return self._params[index]

    def __len__(self):
        return len(self._samples)

    def advance_iteration(self, param):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                self._params.append(param)
                self._samples.append(self._iter)
                self._iter += 1
        else:
            self._params.append(param)
            self._samples.append(self._iter)
            self._iter += 1
