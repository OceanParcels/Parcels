from parcels import FieldSet, Field, ParticleSet_Benchmark, JITParticle, AdvectionRK4, ErrorCode, Variable
from datetime import timedelta as delta
from glob import glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
import xarray as xr
import warnings
import math
import sys
import os
import gc
from  argparse import ArgumentParser
import fnmatch
import time as ostime
#import dask
warnings.simplefilter("ignore", category=xr.SerializationWarning)

try:
    from mpi4py import MPI
except:
    MPI = None


def create_galapagos_fieldset(datahead, periodic_wrap, use_stokes):
    # dask.config.set({'array.chunk-size': '16MiB'})
    ddir = os.path.join(datahead,"NEMO-MEDUSA/ORCA0083-N006/")
    ufiles = sorted(glob(ddir+'means/ORCA0083-N06_20[00-10]*d05U.nc'))
    vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
    meshfile = glob(ddir+'domain/coordinates.nc')
    nemo_files = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
                 'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}
    nemo_variables = {'U': 'uo', 'V': 'vo'}
    nemo_dimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
    period = delta(days=366) if periodic_wrap else False
    extrapolation = False if periodic_wrap else True
    # ==== Because the stokes data is a different grid, we actually need to define the chunking ==== #
    # fieldset_nemo = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, field_chunksize='auto')
    nemo_chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 100, 'x': 100}
    fieldset_nemo = FieldSet.from_nemo(nemo_files, nemo_variables, nemo_dimensions, field_chunksize=nemo_chs, time_periodic=period, allow_time_extrapolation=extrapolation)

    if wstokes:
        stokes_files = sorted(glob(datahead+"/WaveWatch3data/CFSR/WW3-*_uss.nc"))
        stokes_variables = {'U': 'uuss', 'V': 'vuss'}
        stokes_dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
        stokes_chs = {'time': 1, 'latitude': 16, 'longitude': 32}
        fieldset_stokes = FieldSet.from_netcdf(stokes_files, stokes_variables, stokes_dimensions, field_chunksize=stokes_chs, time_periodic=period, allow_time_extrapolation=extrapolation)
        fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

        fieldset = FieldSet(U=fieldset_nemo.U+fieldset_stokes.U, V=fieldset_nemo.V+fieldset_stokes.V)
        fU = fieldset.U[0]
    else:
        fieldset = fieldset_nemo
        fU = fieldset.U

    return fieldset, fU


def plot(total_times = [], compute_times = [], io_times = [], memory_used = [], nparticles = [], imageFilePath = ""):
    plot_t = []
    plot_ct = []
    plot_iot = []
    plot_npart = []
    cum_t = 0
    cum_ct = 0
    cum_iot = 0
    t_scaler = 1. * 10./1.0
    npart_scaler = 1.0 / 1000.0
    for i in range(0, len(total_times)):
        #if i==0:
        #    plot_t.append( (total_times[i]-global_t_0)*t_scaler )
        #    cum_t += (total_times[i]-global_t_0)
        #else:
        #    plot_t.append( (total_times[i]-total_times[i-1])*t_scaler )
        #    cum_t += (total_times[i]-total_times[i-1])
        plot_t.append( total_times[i]*t_scaler )
        cum_t += (total_times[i])

    for i in range(0, len(compute_times)):
        plot_ct.append(compute_times[i] * t_scaler)
        cum_ct += compute_times[i]
    for i in range(0, len(io_times)):
        plot_iot.append(io_times[i] * t_scaler)
        cum_iot += io_times[i]
    for i in range(0, len(nparticles)):
        plot_npart.append(nparticles[i] * npart_scaler)

    if memory_used is not None:
        #mem_scaler = (1*10)/(1024*1024*1024)
        mem_scaler = 1 / (1024 * 1024 * 1024)
        plot_mem = []
        for i in range(0, len(memory_used)):
            plot_mem.append(memory_used[i] * mem_scaler)

    # do_ct_plot = True
    do_iot_plot = True
    do_mem_plot = True
    do_npart_plot = True
    assert (len(plot_t) == len(plot_ct))
    # assert (len(plot_t) == len(plot_iot))
    if len(plot_t) != len(plot_iot):
        print("plot_t and plot_iot have different lengths ({} vs {})".format(len(plot_t), len(plot_iot)))
        do_iot_plot = False
    # assert (len(plot_t) == len(plot_mem))
    if len(plot_t) != len(plot_mem):
        print("plot_t and plot_mem have different lengths ({} vs {})".format(len(plot_t), len(plot_mem)))
        do_mem_plot = False
    # assert (len(plot_t) == len(plot_npart))
    if len(plot_t) != len(plot_npart):
        print("plot_t and plot_npart have different lengths ({} vs {})".format(len(plot_t), len(plot_npart)))
        do_npart_plot = False
    x = []
    for i in itertools.islice(itertools.count(), 0, len(plot_t)):
        x.append(i)

    fig, ax = plt.subplots(1, 1, figsize=(21, 12))
    ax.plot(x, plot_t, 'o-', label="total time_spent [100ms]")
    ax.plot(x, plot_ct, 'o-', label="compute time_spent [100ms]")
    # == this is still the part that breaks - as they are on different time scales, possibly leave them out ? == #
    if do_iot_plot:
        ax.plot(x, plot_iot, 'o-', label="io time_spent [100ms]")
    if (memory_used is not None) and do_mem_plot:
        #ax.plot(x, plot_mem, 'x-', label="memory_used (cumulative) [100 MB]")
        ax.plot(x, plot_mem, 'x-', label="memory_used (cumulative) [1 GB]")
    if do_npart_plot:
        ax.plot(x, plot_npart, '-', label="sim. particles [# 1000]")
    plt.xlim([0, 730])
    plt.ylim([0, 120])
    plt.legend()
    ax.set_xlabel('iteration')
    plt.savefig(os.path.join(odir, imageFilePath), dpi=600, format='png')

    sys.stdout.write("cumulative total runtime: {}\n".format(cum_t))
    sys.stdout.write("cumulative compute time: {}\n".format(cum_ct))
    sys.stdout.write("cumulative I/O time: {}\n".format(cum_iot))


def perIterGC():
    gc.collect()


class GalapagosParticle(JITParticle):
    age = Variable('age', initial=0.)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=14.0*86400.0)  # np.finfo(np.float64).max


def Age(particle, fieldset, time):
    if particle.state == ErrorCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def periodicBC(particle, fieldSet, time):
    dlon = -89.0 + 91.8
    dlat = 0.7 + 1.4
    if particle.lon > -89.0:
        particle.lon -= dlon
    if particle.lon < -91.8:
        particle.lon += dlon
    if particle.lat > 0.7:
        particle.lat -= dlat
    if particle.lat < -1.4:
        particle.lat += dlat


if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection around an idealised peninsula")
    parser.add_argument("-s", "--stokes", dest="stokes", action='store_true', default=False, help="use Stokes' field data")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    # parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 365)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=str, default="1*365", help="runtime in days (default: 1*365)")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    args = parser.parse_args()

    time_in_days = int(float(eval(args.time_in_days)))
    wstokes = args.stokes

    headdir = ""
    odir = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    dirread_mesh = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments/galapagos".format("ckehl")
        odir = os.path.join(headdir,"BENCHres")
        datahead = "/data/oceanparcels/input_data"
        ddir_head = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
    elif fnmatch.fnmatchcase(os.uname()[1], "int?.*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        headdir = "/scratch/shared/{}/experiments/galapagos".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "/BENCHres")
        datahead = "/projects/0/topios/hydrodynamic_data"
        ddir_head = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
    else:
        headdir = "/var/scratch/galapagos"
        odir = os.path.join(headdir, "BENCHres")
        datahead = "/data"
        ddir_head = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')




    fieldset, fu = create_galapagos_fieldset(datahead, True, wstokes)
    fname = os.path.join(odir,"galapagosparticles_bwd_wstokes_v2.nc") if wstokes else os.path.join(odir,"galapagosparticles_bwd_v2.nc")

    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    startlon, startlat = np.meshgrid(np.arange(galapagos_extent[0], galapagos_extent[1], 0.2),
                                     np.arange(galapagos_extent[2], galapagos_extent[3], 0.2))

    pset = ParticleSet_Benchmark(fieldset=fieldset, pclass=GalapagosParticle, lon=startlon, lat=startlat, time=fU.grid.time[-1], repeatdt=delta(days=7))
    outfile = pset.ParticleFile(name=fname, outputdt=delta(days=1))
    kernel = pset.Kernel(AdvectionRK4)+pset.Kernel(Age)+pset.Kernel(periodicBC)

    starttime = 0
    endtime = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # starttime = MPI.Wtime()
            starttime = ostime.process_time()
    else:
        #starttime = ostime.time()
        starttime = ostime.process_time()

    pset.execute(kernel, dt=delta(hours=-1), output_file=outfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # endtime = MPI.Wtime()
            endtime = ostime.process_time()
    else:
        # endtime = ostime.time()
        endtime = ostime.process_time()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            size_Npart = len(pset.nparticle_log.params)
            if size_Npart>0:
                sys.stdout.write("final # particles: {}\n".format(pset.nparticle_log.params[size_Npart-1]))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
            avg_time = np.mean(np.array(pset.total_log.times_steps, dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
    else:
        size_Npart = len(pset.nparticle_log.params)
        if size_Npart > 0:
            sys.stdout.write("final # particles: {}\n".format(pset.nparticle_log.params[size_Npart - 1]))
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
        avg_time = np.mean(np.array(pset.total_log.times_steps, dtype=np.float64))
        sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

    outfile.close()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() == 0:
            # plot(perflog.samples, perflog.times_steps, perflog.memory_steps, pset.compute_log.times_steps, pset.io_log.times_steps, os.path.join(odir, args.imageFileName))
            plot(pset.total_log.times_steps, pset.compute_log.times_steps, pset.io_log.times_steps, pset.mem_log.params, pset.nparticle_log.params, os.path.join(odir, args.imageFileName))
    else:
        # plot(perflog.samples, perflog.times_steps, perflog.memory_steps, pset.compute_log.times_steps, pset.io_log.times_steps, os.path.join(odir, args.imageFileName))
        plot(pset.total_log.times_steps, pset.compute_log.times_steps, pset.io_log.times_steps, pset.mem_log.params, pset.nparticle_log.params, os.path.join(odir, args.imageFileName))
