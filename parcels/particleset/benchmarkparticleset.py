import time as time_module
from datetime import datetime
from datetime import timedelta as delta
import psutil
import os
import sys
from platform import system as system_name
from resource import getrusage, RUSAGE_SELF
import numpy as np
# import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.tools.loggers import logger
from parcels.tools.performance_logger import TimingLog, ParamLogging, Asynchronous_ParamLogging



__all__ = ['BenchmarkParticleSet']

def measure_mem():
    process = psutil.Process(os.getpid())
    pmem = process.memory_info()
    pmem_total = pmem.shared + pmem.text + pmem.data + pmem.lib
    # print("psutil - res-set: {}; res-shr: {} res-text: {}, res-data: {}, res-lib: {}; res-total: {}".format(pmem.rss, pmem.shared, pmem.text, pmem.data, pmem.lib, pmem_total))
    return pmem_total

def measure_mem_rss():
    process = psutil.Process(os.getpid())
    pmem = process.memory_info()
    pmem_total = pmem.shared + pmem.text + pmem.data + pmem.lib
    # print("psutil - res-set: {}; res-shr: {} res-text: {}, res-data: {}, res-lib: {}; res-total: {}".format(pmem.rss, pmem.shared, pmem.text, pmem.data, pmem.lib, pmem_total))
    return pmem.rss

def measure_mem_usage():
    rsc = getrusage(RUSAGE_SELF)
    print("RUSAGE - Max. RES set-size: {}; shr. mem size: {}; ushr. mem size: {}".format(rsc.ru_maxrss, rsc.ru_ixrss, rsc.ru_idrss))
    if system_name() == "Linux":
        return rsc.ru_maxrss*1024
    return rsc.ru_maxrss

USE_ASYNC_MEMLOG = False
USE_RUSE_SYNC_MEMLOG = False  # can be faulty

class BenchmarkParticleSet:

    def __init__(self):
        self.total_log = TimingLog()
        self.compute_log = TimingLog()
        self.io_log = TimingLog()
        self.mem_io_log = TimingLog()
        self.plot_log = TimingLog()
        self.nparticle_log = ParamLogging()
        self.mem_log = ParamLogging()
        self.async_mem_log = Asynchronous_ParamLogging()
        self.process = psutil.Process(os.getpid())

    def set_async_memlog_interval(self, interval):
        self.async_mem_log.measure_interval = interval

    def plot_and_log(self, total_times=None, compute_times=None, io_times=None, plot_times=None, memory_used=None, nparticles=None, target_N=1, imageFilePath="", odir=os.getcwd(), xlim_range=None, ylim_range=None):
        # == do something with the log-arrays == #
        if total_times is None or type(total_times) not in [list, dict, np.ndarray]:
            total_times = self.total_log.get_values()
        if not isinstance(total_times, np.ndarray):
            total_times = np.array(total_times)
        if compute_times is None or type(compute_times) not in [list, dict, np.ndarray]:
            compute_times = self.compute_log.get_values()
        if not isinstance(compute_times, np.ndarray):
            compute_times = np.array(compute_times)
        mem_io_times = None
        if io_times is None or type(io_times) not in [list, dict, np.ndarray]:
            io_times = self.io_log.get_values()
            mem_io_times = self.mem_io_log.get_values()
        if not isinstance(io_times, np.ndarray):
            io_times = np.array(io_times)
        if mem_io_times is not None:
            mem_io_times = np.array(mem_io_times)
            io_times += mem_io_times
        if plot_times is None or type(plot_times) not in [list, dict, np.ndarray]:
            plot_times = self.plot_log.get_values()
        if not isinstance(plot_times, np.ndarray):
            plot_times = np.array(plot_times)
        if memory_used is None or type(memory_used) not in [list, dict, np.ndarray]:
            memory_used = self.mem_log.get_params()
        if not isinstance(memory_used, np.ndarray):
            memory_used = np.array(memory_used)
        if nparticles is None or type(nparticles) not in [list, dict, np.ndarray]:
            nparticles = []
        if not isinstance(nparticles, np.ndarray):
            nparticles = np.array(nparticles, dtype=np.int32)

        memory_used_async = None
        if USE_ASYNC_MEMLOG:
            memory_used_async = np.array(self.async_mem_log.get_params(), dtype=np.int64)

        t_scaler = 1. * 10./1.0
        npart_scaler = 1.0 / 1000.0
        mem_scaler = 1.0 / (1024 * 1024 * 1024)
        plot_t = (total_times * t_scaler).tolist()
        plot_ct = (compute_times * t_scaler).tolist()
        plot_iot = (io_times * t_scaler).tolist()
        plot_drawt = (plot_times * t_scaler).tolist()
        plot_npart = (nparticles * npart_scaler).tolist()
        plot_mem = []
        if memory_used is not None and len(memory_used) > 1:
            plot_mem = (memory_used * mem_scaler).tolist()

        plot_mem_async = None
        if USE_ASYNC_MEMLOG:
            plot_mem_async = (memory_used_async * mem_scaler).tolist()

        do_iot_plot = True
        do_drawt_plot = False
        do_mem_plot = True
        do_mem_plot_async = True
        do_npart_plot = True
        assert (len(plot_t) == len(plot_ct))
        if len(plot_t) != len(plot_iot):
            print("plot_t and plot_iot have different lengths ({} vs {})".format(len(plot_t), len(plot_iot)))
            do_iot_plot = False
        if len(plot_t) != len(plot_drawt):
            print("plot_t and plot_drawt have different lengths ({} vs {})".format(len(plot_t), len(plot_iot)))
            do_drawt_plot = False
        if len(plot_t) != len(plot_mem):
            print("plot_t and plot_mem have different lengths ({} vs {})".format(len(plot_t), len(plot_mem)))
            do_mem_plot = False
        if len(plot_t) != len(plot_npart):
            print("plot_t and plot_npart have different lengths ({} vs {})".format(len(plot_t), len(plot_npart)))
            do_npart_plot = False
        x = np.arange(start=0, stop=len(plot_t))

        fig, ax = plt.subplots(1, 1, figsize=(21, 12))
        ax.plot(x, plot_t, 's-', label="total time_spent [100ms]")
        ax.plot(x, plot_ct, 'o-', label="compute-time spent [100ms]")
        if do_iot_plot:
            ax.plot(x, plot_iot, 'o-', label="io-time spent [100ms]")
        if do_drawt_plot:
            ax.plot(x, plot_drawt, 'o-', label="draw-time spent [100ms]")
        if (memory_used is not None) and do_mem_plot:
            ax.plot(x, plot_mem, '.-', label="memory_used (cumulative) [1 GB]")
        if USE_ASYNC_MEMLOG:
            if (memory_used_async is not None) and do_mem_plot_async:
                ax.plot(x, plot_mem_async, 'x-', label="memory_used [async] (cum.) [1GB]")
        if do_npart_plot:
            ax.plot(x, plot_npart, '-', label="sim. particles [# 1000]")
        if xlim_range is not None:
            plt.xlim(list(xlim_range))  # [0, 730]
        if ylim_range is not None:
            plt.ylim(list(ylim_range))  # [0, 120]
        plt.legend()
        ax.set_xlabel('iteration')
        plt.savefig(os.path.join(odir, imageFilePath), dpi=600, format='png')

        sys.stdout.write("cumulative total runtime: {}\n".format(total_times.sum()))
        sys.stdout.write("cumulative compute time: {}\n".format(compute_times.sum()))
        sys.stdout.write("cumulative I/O time: {}\n".format(io_times.sum()))
        sys.stdout.write("cumulative plot time: {}\n".format(plot_times.sum()))

        csv_file = os.path.splitext(imageFilePath)[0]+".csv"
        with open(os.path.join(odir, csv_file), 'w') as f:
            nparticles_t0 = 0
            nparticles_tN = 0
            if nparticles is not None:
                nparticles_t0 = nparticles[0]
                nparticles_tN = nparticles[-1]
            ncores = 1
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                ncores = mpi_comm.Get_size()
            header_string = "target_N, start_N, final_N, avg_N, ncores, avg_kt_total[s], avg_kt_compute[s], avg_kt_io[s], avg_kt_plot[s], cum_t_total[s], cum_t_compute[s], com_t_io[s], cum_t_plot[s], max_mem[MB]\n"
            f.write(header_string)
            data_string = "{}, {}, {}, {}, {}, ".format(target_N, nparticles_t0, nparticles_tN, nparticles.mean(), ncores)
            data_string += "{:2.10f}, {:2.10f}, {:2.10f}, {:2.10f}, ".format(total_times.mean(), compute_times.mean(), io_times.mean(), plot_times.mean())
            max_mem_sync = 0
            if memory_used is not None and len(memory_used) > 1:
                memory_used = np.floor(memory_used / (1024*1024))
                memory_used = memory_used.astype(dtype=np.uint32)
                max_mem_sync = memory_used.max()
            max_mem_async = 0
            if USE_ASYNC_MEMLOG:
                if memory_used_async is not None and len(memory_used_async) > 1:
                    memory_used_async = np.floor(memory_used_async / (1024*1024))
                    memory_used_async = memory_used_async.astype(dtype=np.int64)
                    max_mem_async = memory_used_async.max()
            max_mem = max(max_mem_sync, max_mem_async)
            data_string += "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {}".format(total_times.sum(), compute_times.sum(), io_times.sum(), plot_times.sum(), max_mem)
            f.write(data_string)
