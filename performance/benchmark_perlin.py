"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4
from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, RectilinearZGrid, ErrorCode
from parcels.field import Field, VectorField, NestedField, SummedField
from parcels import plotTrajectoriesFile_loadedField
from datetime import timedelta as delta
import math
from argparse import ArgumentParser
#import datetime
import numpy as np
import xarray as xr
import pytest
import psutil
import gc
import os
import time as ostime
import matplotlib.pyplot as plt
from parcels.tools import perlin3d

import sys
try:
    from mpi4py import MPI
except:
    MPI = None
with_GC = False


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0
Nparticle = int(math.pow(2,10)) # equals to Nparticle = 1024
#Nparticle = int(math.pow(2,11)) # equals to Nparticle = 2048
#Nparticle = int(math.pow(2,12)) # equals to Nparticle = 4096
#Nparticle = int(math.pow(2,13)) # equals to Nparticle = 8192
#Nparticle = int(math.pow(2,14)) # equals to Nparticle = 16384
#Nparticle = int(math.pow(2,15)) # equals to Nparticle = 32768
#Nparticle = int(math.pow(2,16)) # equals to Nparticle = 65536
#Nparticle = int(math.pow(2,17)) # equals to Nparticle = 131072
#Nparticle = int(math.pow(2,18)) # equals to Nparticle = 262144
#Nparticle = int(math.pow(2,19)) # equals to Nparticle = 524288

noctaves=3
#noctaves=4 # formerly
perlinres=(1,32,8)
shapescale=(4,8,8)
#shapescale=(8,6,6) # formerly
perlin_persistence=0.3
a = 10000 * 1e3
b = 10000 * 1e3
scalefac = 2.0

# Idea for 4D: perlin3D creates a time-consistent 3D field
# Thus, we can use skimage to create shifted/rotated/morphed versions
# for the depth domain, so that perlin4D = [depth][time][lat][lon].
# then, we can do a transpose-op in numpy, to get [time][depth][lat][lon]

# we need to modify the kernel.execute / pset.execute so that it returns from the JIT
# in a given time WITHOUT writing to disk via outfie => introduce a pyloop_dt

class PerformanceLog():
    samples = []
    times_steps = []
    memory_steps = []
    fds_steps = []
    _iter = 0

    def advance(self):
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            process = psutil.Process(os.getpid())
            mem_B_used = process.memory_info().rss
            fds_open = len(process.open_files())
            mem_B_used_total = mpi_comm.reduce(mem_B_used, op=MPI.SUM, root=0)
            fds_open_total = mpi_comm.reduce(fds_open, op=MPI.SUM, root=0)
            if mpi_rank == 0:
                self.times_steps.append(ostime.time())
                self.memory_steps.append(mem_B_used_total)
                self.fds_steps.append(fds_open_total)
                self.samples.append(self._iter)
                self._iter+=1
        else:
            process = psutil.Process(os.getpid())
            self.times_steps.append(ostime.time())
            self.memory_steps.append(process.memory_info().rss)
            self.fds_steps.append(len(process.open_files()))
            self.samples.append(self._iter)
            self._iter+=1

def plot(x, times, memory_used, nfiledescriptors, imageFilePath):
    plot_t = []
    t_scaler = 1. * 10./1.0
    for i in range(len(times)):
        if i==0:
            plot_t.append( (times[i]-global_t_0)*t_scaler )
        else:
            plot_t.append( (times[i]-times[i-1])*t_scaler )
    #mem_scaler = (1*10)/(1024*1024*1024)
    mem_scaler = 1 / (1024 * 1024 * 1024)
    plot_mem = []
    for i in range(len(memory_used)):
        #if i==0:
        #    plot_mem.append((memory_used[i]-global_m_0)*mem_scaler)
        #else:
        #    plot_mem.append((memory_used[i] - memory_used[i-1]) * mem_scaler)
        plot_mem.append(memory_used[i] * mem_scaler)

    fig, ax = plt.subplots(1, 1, figsize=(21, 12))
    ax.plot(x, plot_t, 'o-', label="time_spent [100ms]")
    ax.plot(x, plot_mem, 'x-', label="memory_used [100 MB]")
    #ax.plot(x, nfiledescriptors, '.-', label="open_files [#]")
    plt.xlim([0, 730])
    plt.ylim([0, 100])
    plt.legend()
    ax.set_xlabel('iteration')
    # ax.set_ylabel('Time spent in pset.execute() [s]')
    # ax.set_ylabel('Time spent [s]')
    plt.savefig(os.path.join(odir, imageFilePath), dpi=600, format='png')

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def RenewParticle(particle, fieldset, time):
    particle.lat = np.random.rand() * a
    particle.lon = np.random.rand() * b

def perlin_fieldset_from_numpy(periodic_wrap=False):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247
    """
    img_shape = (perlinres[0]*shapescale[0], int(math.pow(2,noctaves))*perlinres[1]*shapescale[1], int(math.pow(2,noctaves))*perlinres[2]*shapescale[2])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, img_shape[1], dtype=np.float32)
    #sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(0, b, img_shape[2], dtype=np.float32)
    #sys.stdout.write("lat field: {}\n".format(lat.size))
    totime = img_shape[0]*24.0*60.0*60.0
    time = np.linspace(0., totime, img_shape[0], dtype=np.float64)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    U = np.transpose(U, (0,2,1))
    #U = np.swapaxes(U, 1, 2)
    #sys.stdout.write("U field shape: {} - [tdim][ydim][xdim]=[{}][{}][{}]\n".format(U.shape, time.shape[0], lat.shape[0], lon.shape[0]))
    V = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    V = np.transpose(V, (0,2,1))
    #V = np.swapaxes(V, 1, 2)

    #P = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    #P = np.transpose(P, (0,2,1))
    ##P = np.swapaxes(P, 1, 2)

    data = {'U': U, 'V': V}
    dimensions = {'time': time, 'lon': lon, 'lat': lat}
    if periodic_wrap:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, time_periodic=delta(days=1))
    else:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, allow_time_extrapolation=True)


def perlin_fieldset_from_xarray(periodic_wrap=False):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247
    """
    img_shape = (perlinres[0]*shapescale[0], int(math.pow(2,noctaves))*perlinres[1]*shapescale[1], int(math.pow(2,noctaves))*perlinres[2]*shapescale[2])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, img_shape[1], dtype=np.float32)
    lat = np.linspace(0, b, img_shape[2], dtype=np.float32)
    totime = img_shape[0] * 24.0 * 60.0 * 60.0
    time = np.linspace(0., totime, img_shape[0], dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    U = np.transpose(U, (0,2,1))
    V = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    V = np.transpose(V, (0,2,1))
    #P = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac

    dimensions = {'time': time, 'lon': lon, 'lat': lat}
    dims = ('time', 'lat', 'lon')
    data = {'Uxr': xr.DataArray(U, coords=dimensions, dims=dims),
            'Vxr': xr.DataArray(V, coords=dimensions, dims=dims)}   #,'Pxr': xr.DataArray(P, coords=dimensions, dims=dims)
    ds = xr.Dataset(data)

    variables = {'U': 'Uxr', 'V': 'Vxr'}
    dimensions = {'time': 'time', 'lat': 'lat', 'lon': 'lon'}
    if periodic_wrap:
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', time_periodic=delta(days=1))
    else:
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', allow_time_extrapolation=True)

def perIterGC():
    gc.collect()

class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    agetime = Variable('agetime', dtype=np.float64, initial=-1.0)
    #def __init__(self, *args, **kwargs):
    #    type(self).age.initial=0.
    #    type(self).agetime.initial=-1.0
    #    super(AgeParticle_JIT, self).__init__(*args, **kwargs)

class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    agetime = Variable('agetime', dtype=np.float64, initial=-1.0)
    #def __init__(self, *args, **kwargs):
    #    type(self).age.initial=0.
    #    type(self).agetime.initial=-1.0
    #    super(AgeParticle_SciPy, self).__init__(*args, **kwargs)

age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}

def Age(particle, fieldset, time):
    if math.fabs(time-particle.agetime) > math.fabs(1.e10 * particle.dt):
        particle.age = particle.age + math.fabs(particle.dt)
        particle.agetime = time

        if particle.age > (60.0*60.0*24.0*14):
            particle.delete()

#def OldAge(particle, fieldset, time):
#    if particle.age > (60.0*60.0*24.0*62.0):
#        particle.delete()
#    else:
#        particle.age = particle.age + math.fabs(particle.dt)
#    #particle.age = particle.age + math.fabs(particle.dt)
#    #if particle.age > (60.0 * 60.0 * 24.0 * 62.0):
#    #    particle.delete()

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
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2e6", help="number of particles to generate and advect (default: 2e6)")
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
    Nparticle = int(float(eval(args.nparticles)))
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            sys.stdout.write("N: {}\n".format(Nparticle))
    else:
        sys.stdout.write("N: {}\n".format(Nparticle))

    dt_minutes = 60
    #dt_minutes = 20

    odir = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:
        odir = "/scratch/{}/experiments".format(os.environ['USER'])
    else:
        odir = "/var/scratch/experiments"

    func_time = []
    mem_used_GB = []

    np.random.seed(0)
    fieldset = None
    if use_xarray:
        fieldset = perlin_fieldset_from_xarray(periodic_wrap=periodicFlag)
    else:
        fieldset = perlin_fieldset_from_numpy(periodic_wrap=periodicFlag)

    if args.compute_mode is 'scipy':
        Nparticle = 2**6

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            global_t_0 = ostime.time()
    else:
        global_t_0 = ostime.time()

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

    if backwardSimulation:
        # ==== backward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(96, 1) * a, lat=np.random.rand(96, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(96, 1) * a, lat=np.random.rand(96, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)
    else:
        # ==== forward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(96, 1) * a, lat=np.random.rand(96, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(96, 1) * a, lat=np.random.rand(96, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)

    output_file = None
    if args.write_out:
        output_file = pset.ParticleFile(name=os.path.join(odir,"test_mem_behaviour.nc"), outputdt=delta(hours=12))
    delete_func = RenewParticle
    if args.delete_particle:
        delete_func=DeleteParticle

    perflog = PerformanceLog()
    postProcessFuncs = [perflog.advance,]

    starttime = ostime.time()
    kernels = pset.Kernel(AdvectionRK4,delete_cfiles=True)
    if agingParticles:
        kernels +=  pset.Kernel(Age,delete_cfiles=True)
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
    endtime = ostime.time()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            dt_time = []
            for i in range(len(perflog.times_steps)):
                if i==0:
                    dt_time.append( (perflog.times_steps[i]-global_t_0) )
                else:
                    dt_time.append( (perflog.times_steps[i]-perflog.times_steps[i-1]) )
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
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
        avg_time = np.mean(np.array(dt_time, dtype=np.float64))
        sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

    if args.write_out:
        output_file.close()
        if args.visualize:
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                if mpi_comm.Get_rank() == 0:
                    plotTrajectoriesFile_loadedField(os.path.join(odir, "test_mem_behaviour.nc"),
                                                     tracerfield=fieldset.U)
            else:
                plotTrajectoriesFile_loadedField(os.path.join(odir,"test_mem_behaviour.nc"),tracerfield=fieldset.U)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() == 0:
            plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.fds_steps, os.path.join(odir, imageFileName))
    else:
        plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.fds_steps, os.path.join(odir, imageFileName))


