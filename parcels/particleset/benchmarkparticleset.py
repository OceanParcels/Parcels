import time as time_module
from datetime import datetime
from datetime import timedelta as delta
import psutil
import os
import sys
from platform import system as system_name
import matplotlib.pyplot as plt
from resource import getrusage, RUSAGE_SELF
import numpy as np
# import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.tools.loggers import logger
from parcels.tools.performance_logger import TimingLog, ParamLogging, Asynchronous_ParamLogging
from parcels.particleset.baseparticleset import BaseParticleSet
from parcels.kernel.benchmarkkernel import BaseBenchmarkKernel
# from parcels.particle import JITParticle
from parcels.application_kernels.advection import AdvectionRK4



__all__ = ['BaseBenchmarkParticleSet']

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

class BaseBenchmarkParticleSet(BaseParticleSet):
    perform_benchmark = False

    def __init__(self, fieldset=None, pclass=None, lon=None, lat=None,
                 depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        super(BaseBenchmarkParticleSet, self).__init__(fieldset, pclass, lon, lat, depth, time, repeatdt, lonlatdepth_dtype, pid_orig, **kwargs)
        self.perform_benchmark = kwargs.pop('use_benchmark', False)
        self.total_log = TimingLog()
        self.compute_log = TimingLog()
        self.io_log = TimingLog()
        self.mem_io_log = TimingLog()
        self.plot_log = TimingLog()
        self.nparticle_log = ParamLogging()
        self.mem_log = ParamLogging()
        self.async_mem_log = Asynchronous_ParamLogging()
        self.process = psutil.Process(os.getpid())

    def __del__(self):
        super(BaseBenchmarkParticleSet, self).__del__()

    def set_async_memlog_interval(self, interval):
        self.async_mem_log.measure_interval = interval

    def execute(self, pyfunc=AdvectionRK4, pyfunc_inter=None, endtime=None, runtime=None, dt=1.,
                moviedt=None, recovery=None, output_file=None, movie_background_field=None,
                verbose_progress=None, postIterationCallbacks=None, callbackdt=None):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.Kernel` object.
                       Kernels can be concatenated using the + operator
        :param endtime: End time for the timestepping loop.
                        It is either a datetime object or a positive double.
        :param runtime: Length of the timestepping loop. Use instead of endtime.
                        It is either a timedelta object or a positive double.
        :param dt: Timestep interval to be passed to the kernel.
                   It is either a timedelta object or a double.
                   Use a negative value for a backward-in-time simulation.
        :param moviedt:  Interval for inner sub-timestepping (leap), which dictates
                         the update frequency of animation.
                         It is either a timedelta object or a positive double.
                         None value means no animation.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param recovery: Dictionary with additional `:mod:parcels.tools.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
        :param movie_background_field: field plotted as background in the movie if moviedt is set.
                                       'vector' shows the velocity as a vector field.
        :param verbose_progress: Boolean for providing a progress bar for the kernel execution loop.
        :param postIterationCallbacks: (Optional) Array of functions that are to be called after each iteration (post-process, non-Kernel)
        :param callbackdt: (Optional, in conjecture with 'postIterationCallbacks) timestep inverval to (latestly) interrupt the running kernel and invoke post-iteration callbacks from 'postIterationCallbacks'
        """
        if not self.perform_benchmark:
            super(BaseBenchmarkParticleSet, self).execute(pyfunc, pyfunc_inter, endtime, runtime, dt, moviedt, recovery, output_file,
                                                          movie_background_field, verbose_progress, postIterationCallbacks, callbackdt)
            return
        self._create_runtime_kernel_(pyfunc, pyfunc_inter)
        # Convert all time variables to seconds
        endtime, runtime, dt, outputdt, moviedt, callbackdt = self._check_times_(endtime, runtime, dt, moviedt=moviedt, callbackdt=callbackdt, output_file=output_file)

        assert runtime is None or runtime >= 0, 'runtime must be positive'
        assert outputdt is None or outputdt >= 0, 'outputdt must be positive'
        assert moviedt is None or moviedt >= 0, 'moviedt must be positive'

        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')

        _starttime, endtime, runtime, dt, execute_once = self._get_time_bounds_(endtime, runtime, dt)
        # First write output_file, because particles could have been added
        if output_file:
            output_file.write(self, _starttime)
        if moviedt:
            self.show(field=movie_background_field, show_time=_starttime, animation=True)

        moviedt, callbackdt, next_prelease, next_output, next_movie, next_callback, next_input = self._get_dt_bounds(_starttime, dt, outputdt, moviedt, callbackdt)
        time = _starttime
        tol = 1e-12

        pbar = None
        walltime_start = None
        if verbose_progress is None:
            walltime_start = time_module.time()
        if verbose_progress:
            pbar = self._create_progressbar_(_starttime, endtime)

        mem_used_start = 0
        if USE_ASYNC_MEMLOG:
            self.async_mem_log.measure_func = measure_mem
            mem_used_start = measure_mem()

        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:
            self.total_log.start_timing()
            if USE_ASYNC_MEMLOG:
                self.async_mem_log.measure_start_value = mem_used_start
                self.async_mem_log.start_partial_measurement()
            if verbose_progress is None and time_module.time() - walltime_start > 10:
                # Showing progressbar if runtime > 10 seconds
                if output_file:
                    logger.info('Temporary output files are stored in %s.' % output_file.tempwritedir_base)
                    logger.info('You can use "parcels_convert_npydir_to_netcdf %s" to convert these '
                                'to a NetCDF file during the run.' % output_file.tempwritedir_base)
                pbar = self._create_progressbar_(_starttime, endtime)
                verbose_progress = True

            if dt > 0:
                next_time = min(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            else:
                next_time = max(next_prelease, next_input, next_output, next_movie, next_callback, endtime)

            if not isinstance(self.kernel, BaseBenchmarkKernel):
                self.compute_log.start_timing()
            time = self._execute_kernel_(next_time, dt, recovery, output_file=output_file, execute_once=execute_once)
            if abs(time-next_prelease) < tol:
                # creating new particles equals a memory-io operation
                if not isinstance(self.kernel, BaseBenchmarkKernel):
                    self.compute_log.stop_timing()
                    self.compute_log.accumulate_timing()

                self.mem_io_log.start_timing()
                self._add_periodic_release_particles_(time, dt)
                self.mem_io_log.stop_timing()
                self.mem_io_log.accumulate_timing()
                next_prelease += self.repeatdt * np.sign(dt)
            else:
                if not isinstance(self.kernel, BaseBenchmarkKernel):
                    self.compute_log.stop_timing()
                else:
                    pass
            if isinstance(self.kernel, BaseBenchmarkKernel):
                self.compute_log.add_aux_measure(self.kernel.compute_timings.sum())
                self.kernel.compute_timings.reset()
                self.io_log.add_aux_measure(self.kernel.io_timings.sum())
                self.kernel.io_timings.reset()
                self.mem_io_log.add_aux_measure(self.kernel.mem_io_timings.sum())
                self.kernel.mem_io_timings.reset()
            self.compute_log.accumulate_timing()
            self.nparticle_log.advance_iteration(self.size)

            if abs(time - next_output) < tol or execute_once:
                self.io_log.start_timing()
                self._write_field_data_(output_file)
                self.io_log.stop_timing()
                self.io_log.accumulate_timing()
            if abs(time - next_output) < tol:
                self.io_log.start_timing()
                self._write_particle_data_(output_file,  time)
                self.io_log.stop_timing()
                self.io_log.accumulate_timing()
                next_output += outputdt * np.sign(dt)
            if abs(time-next_movie) < tol:
                self.plot_log.start_timing()
                self.show(field=movie_background_field, show_time=time, animation=True)
                self.plot_log.stop_timing()
                self.plot_log.accumulate_timing()
                next_movie += moviedt * np.sign(dt)
            if abs(time-next_callback) < tol:
                self.mem_io_log.start_timing()
                if postIterationCallbacks is not None:
                    for extFunc in postIterationCallbacks:
                        extFunc()
                self.mem_io_log.stop_timing()
                self.mem_io_log.accumulate_timing()
                next_callback += callbackdt * np.sign(dt)
            if time != endtime:
                self.io_log.start_timing()
                next_input = self.fieldset.computeTimeChunk(time, dt)
                self.io_log.stop_timing()
                self.io_log.accumulate_timing()
            if dt == 0:
                break
            if verbose_progress:
                self.plot_log.start_timing()
                pbar.update(abs(time - _starttime))
                self.plot_log.stop_timing()
                self.plot_log.accumulate_timing()
            self.total_log.stop_timing()
            self.total_log.accumulate_timing()
            mem_B_used_total = 0
            if USE_RUSE_SYNC_MEMLOG:
                mem_B_used_total = measure_mem_usage()
            else:
                mem_B_used_total = measure_mem_rss()
            self.mem_log.advance_iteration(mem_B_used_total)
            if USE_ASYNC_MEMLOG:
                self.async_mem_log.stop_partial_measurement()  # does 'advance_iteration' internally

            self.compute_log.advance_iteration()
            self.io_log.advance_iteration()
            self.mem_io_log.advance_iteration()
            self.plot_log.advance_iteration()
            self.total_log.advance_iteration()

        if output_file:
            self.io_log.start_timing()
            output_file.write(self, time)
            self.io_log.stop_timing()
            self.io_log.accumulate_timing()
        if verbose_progress:
            self.plot_log.start_timing()
            pbar.finish()
            self.plot_log.stop_timing()
            self.plot_log.accumulate_timing()

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
