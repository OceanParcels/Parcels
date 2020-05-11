"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4_3D
from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, ErrorCode
from parcels import rng as random
from parcels.field import VectorField, NestedField, SummedField
from parcels import plotTrajectoriesFile_loadedField
from datetime import timedelta as delta
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import fnmatch
# import dask as da
# import dask.array as daArray
from glob import glob
import time as ostime
import math
import matplotlib.pyplot as plt
import sys
import os
import psutil
import gc
try:
    from mpi4py import MPI
except:
    MPI = None

with_GC = False

import warnings
import xarray as xr
warnings.simplefilter("ignore", category=xr.SerializationWarning)

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0
odir = ""

class PerformanceLog():
    samples = []
    times_steps = []
    memory_steps = []
    Nparticles_step = []
    _iter = 0
    pset = None

    def advance(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            process = psutil.Process(os.getpid())
            mem_B_used = process.memory_info().rss
            mem_B_used_total = mpi_comm.reduce(mem_B_used, op=MPI.SUM, root=0)
            Nparticles_global = 0
            if self.pset is not None:
                Nparticles_local = len(self.pset)
                Nparticles_global = mpi_comm.reduce(Nparticles_local, op=MPI.SUM, root=0)
            if mpi_rank == 0:
                #self.times_steps.append(MPI.Wtime())
                self.times_steps.append(ostime.process_time())
                self.memory_steps.append(mem_B_used_total)
                if self.pset is not None:
                    self.Nparticles_step.append(Nparticles_global)
                self.samples.append(self._iter)
                self._iter+=1
        else:
            process = psutil.Process(os.getpid())
            #self.times_steps.append(ostime.time())
            self.times_steps.append(ostime.process_time())
            self.memory_steps.append(process.memory_info().rss)
            if self.pset is not None:
                self.Nparticles_step.append(len(self.pset))
            self.samples.append(self._iter)
            self._iter+=1

def plot(x, times, memory_used, nparts, imageFilePath):
    plot_t = []
    #t_scaler = 1. * 10./1.0
    t_scaler = 1. * 5. / 1.0
    for i in range(len(times)):
        if i==0:
            plot_t.append( (times[i]-global_t_0)*t_scaler )
        else:
            plot_t.append( (times[i]-times[i-1])*t_scaler )
    mem_scaler = (1*10)/(1024*1024*1024)
    plot_mem = []
    for i in range(len(memory_used)):
        plot_mem.append(memory_used[i] * mem_scaler)
    npart_scaler = 1.0 / 1000.0
    plot_npart = []
    for i in range(len(nparts)):
        plot_npart.append(nparts[i] * npart_scaler)


    fig, ax = plt.subplots(1, 1, figsize=(21, 12))
    #ax.plot(x, plot_t, 'o-', label="time_spent [100ms]")
    ax.plot(x, plot_t, 'o-', label="time_spent [500ms]")
    ax.plot(x, plot_mem, 'x-', label="memory_used (cumulative) [100 MB]")
    ax.plot(x, plot_npart, '-', label="sim. particles [# 1000]")
    plt.xlim([0, 730])
    #plt.ylim([0, 120])
    plt.ylim([0, 250])
    plt.legend()
    ax.set_xlabel('iteration')
    plt.savefig(os.path.join(odir, imageFilePath), dpi=600, format='png')


def create_CMEMS_fieldset(datahead, periodic_wrap):
    ddir = os.path.join(datahead, "CMEMS/GLOBAL_REANALYSIS_PHY_001_030/")
    files = sorted(glob(ddir+"mercatorglorys12v1_gl12_mean_201607*.nc"))
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    # chs = False
    chs = 'auto'
    if periodic_wrap:
        return FieldSet.from_netcdf(files, variables, dimensions, field_chunksize=chs, time_periodic=delta(days=1))
    else:
        return FieldSet.from_netcdf(files, variables, dimensions, field_chunksize=chs, allow_time_extrapolation=True)


class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)

class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)

age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}

def periodicBC(particle, fieldSet, time):
    if particle.lon > 180.0:
        particle.lon -= 360.0
    if particle.lon < -180.0:
        particle.lon += 360.0
    if particle.lat > 90.0:
        particle.lat -= 170.0
    if particle.lat < -80.0:
        particle.lat += 170.0

def initialize(particle, fieldset, time):
    if particle.initialized_dynamic < 1:
        particle.life_expectancy = time + random.uniform(.0, fieldset.life_expectancy) * math.sqrt(3.0/2.0)
        particle.initialized_dynamic = 1

def Age(particle, fieldset, time):
    if particle.state == ErrorCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def RenewParticle(particle, fieldset, time):
    particle.lat = np.random.rand() * 360.0 -180.0
    particle.lon = np.random.rand() * 160.0 -80.0

def perIterGC():
    gc.collect()

if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--release", dest="release", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    parser.add_argument("-rt", "--releasetime", dest="repeatdt", type=int, default=720, help="repeating release rate of added particles in Minutes (default: 720min = 12h)")
    parser.add_argument("-a", "--aging", dest="aging", action='store_true', default=False, help="Removed aging particles dynamically (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-x", "--xarray", dest="use_xarray", action='store_true', default=False, help="use xarray as data backend")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False, help="animate the particle trajectories during the run or not (default: False).")
    parser.add_argument("-V", "--visualize", dest="visualize", action='store_true', default=False, help="Visualize particle trajectories at the end (default: False). Requires -w in addition to take effect.")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-sN", "--start_n_particles", dest="start_nparticles", type=str, default="96", help="(optional) number of particles generated per release cycle (if --rt is set) (default: 96)")
    parser.add_argument("-m", "--mode", dest="compute_mode", choices=['jit','scipy'], default="jit", help="computation mode = [JIT, SciPp]")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    args = parser.parse_args()

    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    backwardSimulation = args.backwards
    repeatdtFlag=args.release
    repeatRateMinutes=args.repeatdt
    time_in_days = args.time_in_days
    use_xarray = args.use_xarray
    agingParticles = args.aging
    with_GC = args.useGC

    headdir = ""
    odir = ""
    dirread_pal = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments".format("ckehl")
        odir = headdir
        datahead = "/data/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
    elif fnmatch.fnmatchcase(os.uname()[1], "int?.*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        headdir = "/scratch/shared/{}/experiments/parcels".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
    else:
        headdir = "/var/scratch/experiments"
        odir = headdir
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')

    Nparticle = int(float(eval(args.nparticles)))
    start_N_particles = int(float(eval(args.start_nparticles)))
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            if agingParticles and not repeatdtFlag:
                sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * math.sqrt(3.0/2.0))))
            else:
                sys.stdout.write("N: {}\n".format(Nparticle))
    else:
        if agingParticles and not repeatdtFlag:
            sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * math.sqrt(3.0/2.0))))
        else:
            sys.stdout.write("N: {}\n".format(Nparticle))

    dt_minutes = 60
    nowtime = datetime.now()
    random.seed(nowtime.microsecond)

    func_time = []
    mem_used_GB = []

    np.random.seed(nowtime.microsecond)
    fieldset = create_CMEMS_fieldset(datahead, periodic_wrap=periodicFlag)

    if args.compute_mode is 'scipy':
        Nparticle = 2**10

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            #global_t_0 = MPI.Wtime()
            global_t_0 = ostime.process_time()
    else:
        #global_t_0 = ostime.time()
        global_t_0 = ostime.process_time()

    simStart = None
    for f in fieldset.get_fields():
        if type(f) in [VectorField, NestedField, SummedField]:  # or not f.grid.defer_load
            continue
        else:
            if backwardSimulation:
                simStart=f.grid.time_full[-1]
            else:
                simStart = f.grid.time_full[0]
            break

    if agingParticles:
        if not repeatdtFlag:
            Nparticle = int(Nparticle * math.sqrt(3.0/2.0))
        fieldset.add_constant('life_expectancy', delta(days=time_in_days).total_seconds())
    if repeatdtFlag:
        addParticleN = Nparticle/2.0
        refresh_cycle = (delta(days=time_in_days).total_seconds() / (addParticleN/start_N_particles)) / math.sqrt(3/2)
        repeatRateMinutes = int(refresh_cycle/60.0) if repeatRateMinutes == 720 else repeatRateMinutes

    if backwardSimulation:
        # ==== backward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart)
    else:
        # ==== forward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart)


    output_file = None
    out_fname = "benchmark_CMEMS"
    if args.write_out:
        if MPI and (MPI.COMM_WORLD.Get_size()>1):
            out_fname += "_MPI"
        else:
            out_fname += "_noMPI"
        out_fname += "_n"+str(Nparticle)
        if backwardSimulation:
            out_fname += "_bwd"
        else:
            out_fname += "_fwd"
        if repeatdtFlag:
            out_fname += "_add"
        if agingParticles:
            out_fname += "_age"
        output_file = pset.ParticleFile(name=os.path.join(odir,out_fname+".nc"), outputdt=delta(hours=24))
    delete_func = RenewParticle
    if args.delete_particle:
        delete_func=DeleteParticle

    perflog = PerformanceLog()
    perflog.pset = pset
    postProcessFuncs = [perflog.advance,]

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            #starttime = MPI.Wtime()
            starttime = ostime.process_time()
    else:
        #starttime = ostime.time()
        starttime = ostime.process_time()
    kernels = pset.Kernel(AdvectionRK4,delete_cfiles=True)
    kernels += pset.Kernel(periodicBC, delete_cfiles=True)
    if agingParticles:
        kernels += pset.Kernel(initialize, delete_cfiles=True)
        kernels += pset.Kernel(Age, delete_cfiles=True)
    if with_GC:
        postProcessFuncs.append(perIterGC)
    if backwardSimulation:
        # ==== backward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
    else:
        # ==== forward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            #endtime = MPI.Wtime()
            endtime = ostime.process_time()
    else:
        #endtime = ostime.time()
        endtime = ostime.process_time()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            dt_time = []
            for i in range(len(perflog.times_steps)):
                if i==0:
                    dt_time.append( (perflog.times_steps[i]-global_t_0) )
                else:
                    dt_time.append( (perflog.times_steps[i]-perflog.times_steps[i-1]) )
            if len(perflog.Nparticles_step)>0:
                sys.stdout.write("final # particles: {}".format(perflog.Nparticles_step[len(perflog.Nparticles_step)-1]))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
            avg_time = np.mean(np.array(dt_time, dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
    else:
        dt_time = []
        for i in range(len(perflog.times_steps)):
            if i == 0:
                dt_time.append((perflog.times_steps[i] - global_t_0))
            else:
                dt_time.append((perflog.times_steps[i] - perflog.times_steps[i - 1]))
        if len(perflog.Nparticles_step) > 0:
            sys.stdout.write("final # particles: {}".format(perflog.Nparticles_step[len(perflog.Nparticles_step)-1]))
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
        avg_time = np.mean(np.array(dt_time, dtype=np.float64))
        sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

    if args.write_out:
        output_file.close()
        if args.visualize:
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                if mpi_comm.Get_rank() == 0:
                    plotTrajectoriesFile_loadedField(os.path.join(odir, out_fname+".nc"),
                                                     tracerfield=fieldset.U)
            else:
                plotTrajectoriesFile_loadedField(os.path.join(odir, out_fname+".nc"),tracerfield=fieldset.U)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() == 0:
            plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.Nparticles_step, os.path.join(odir, imageFileName))
    else:
        plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.Nparticles_step, os.path.join(odir, imageFileName))
