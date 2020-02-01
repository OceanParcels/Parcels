from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from datetime import timedelta as delta
from argparse import ArgumentParser
import numpy as np
import dask as da
import dask.array as daArray
from glob import glob
import time as ostime
import matplotlib.pyplot as plt
import os
import parcels
import psutil

import gc
import sys
try:
    from mpi4py import MPI
except:
    MPI = None

with_GC = False
with_ChunkInfoPrint = False

global_t_0 = 0
global_m_0 = 0
global_samples = []
samplenr = 0
global_memory_step = []
global_time_steps = []
global_fds_step = []


class IterationCounter():
    _iter = 0

    @classmethod
    def advance(self):
        old_iter = self._iter
        self._iter += 1
        return old_iter


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
                self._iter += 1
        else:
            process = psutil.Process(os.getpid())
            self.times_steps.append(ostime.time())
            self.memory_steps.append(process.memory_info().rss)
            self.fds_steps.append(len(process.open_files()))
            self.samples.append(self._iter)
            self._iter += 1


def set_cmems_fieldset(cs, deferLoadFlag=True, periodicFlag=False):
    ddir_head = "/data/oceanparcels/input_data"
    ddir = os.path.join(ddir_head, "CMEMS/GLOBAL_REANALYSIS_PHY_001_030/")
    files = sorted(glob(ddir+"mercatorglorys12v1_gl12_mean_201607*.nc"))
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}

    if cs not in ['auto', False]:
        # cs = (1, cs, cs) # == OLD initialization as tuple. Advised to rather use a dictionary. ==#
        cs = {dimensions['time']: 1, dimensions['lon']: cs, dimensions['lat']: cs}
    if periodicFlag:
        return FieldSet.from_netcdf(files, variables, dimensions, time_periodic=delta(days=30), deferred_load=deferLoadFlag, field_chunksize=cs)
    else:
        return FieldSet.from_netcdf(files, variables, dimensions, allow_time_extrapolation=True, deferred_load=deferLoadFlag, field_chunksize=cs)


def print_field_info(fieldset):
    for f in fieldset.get_fields():
        if type(f) in [parcels.VectorField, parcels.NestedField, parcels.SummedField] or not f.grid.defer_load:
            continue
        if isinstance(f.data, daArray.core.Array):
            sys.stdout.write("Array of Field[name={}] is dask.Array\n".format(f.name))
            sys.stdout.write(
                "Chunk info of Field[name={}]: field.nchunks={}; shape(field.data.nchunks)={}; field.data.numblocks={}; shape(f.data)={}\n".format(
                    f.name, f.nchunks, (len(f.data.chunks[0]), len(f.data.chunks[1]), len(f.data.chunks[2])),
                    f.data.numblocks, f.data.shape))
        sys.stdout.write("Chunk info of Grid[field.name={}]: g.chunk_info={}; g.load_chunk={}; len(g.load_chunk)={}\n".format(
            f.name, f.grid.chunk_info, f.grid.load_chunk, len(f.grid.load_chunk)))


def plot(x, times, memory_used, nfiledescriptors, imageFilePath):
    plot_t = []
    for i in range(len(times)):
        if i == 0:
            plot_t.append(times[i]-global_t_0)
        else:
            plot_t.append(times[i]-times[i-1])
    mem_scaler = (1*10)/(1024*1024*1024)
    plot_mem = []
    for i in range(len(memory_used)):
        plot_mem.append(memory_used[i] * mem_scaler)

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.plot(x, plot_t, 'o-', label="time_spent [s]")
    ax.plot(x, plot_mem, 'x-', label="memory_used [100 MB]")
    ax.plot(x, nfiledescriptors, '.-', label="open_files [#]")
    plt.ylim([0, 50])
    plt.legend()
    ax.set_xlabel('iteration')
    plt.savefig(os.path.join(odir, imageFilePath), dpi=300, format='png')


def LogMemTimeFds():
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        process = psutil.Process(os.getpid())
        mem_B_used = process.memory_info().rss
        fds_open = len(psutil.Process().open_files())
        mem_B_used_total = mpi_comm.reduce(mem_B_used, op=MPI.SUM, root=0)
        fds_open_total = mpi_comm.reduce(fds_open, op=MPI.SUM, root=0)
        if mpi_rank == 0:
            global_time_steps.append(ostime.time())
            global_memory_step.append(mem_B_used_total)
            global_fds_step.append(fds_open_total)
            global_samples.append(IterationCounter.advance())
    else:
        process = psutil.Process(os.getpid())
        global_time_steps.append(ostime.time())
        global_memory_step.append(process.memory_info().rss)
        global_fds_step.append(len(psutil.Process().open_files()))


def perIterGC():
    gc.collect()


if __name__ == '__main__':
    field_chunksize = 1
    do_chunking = False
    auto_chunking = False
    imageFileName = ""
    parser = ArgumentParser(description="Example of particle advection around an idealised peninsula")
    parser.add_argument("-f", "--fieldsize", dest="fieldsize", type=int, default=1, help="size of each field chunk")
    parser.add_argument("-c", "--do-chunking", dest="do_chunking", action='store_true', default=False, help="enable/disable field chunking")
    parser.add_argument("-a", "--auto-chunking", dest="auto_chunking", action='store_true', default=False, help="enable/disable auto-chunking")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-d", "--defer", dest="defer", action='store_false', default=True, help="enable/disable running with deferred load (default: True)")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--repeatdt", dest="repeatdt", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    args = parser.parse_args()

    auto_chunking = args.auto_chunking
    do_chunking = args.do_chunking
    if auto_chunking:
        do_chunking = True
    elif do_chunking:
        field_chunksize = args.fieldsize
    imageFileName = args.imageFileName
    deferLoadFlag = args.defer
    periodicFlag = args.periodic
    backwardSimulation = args.backwards
    repeatdtFlag = args.repeatdt

    odir = "/scratch/{}/experiments".format(os.environ['USER'])
    func_time = []
    mem_used_GB = []
    open_fds = []
    auto_field_size = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        if mpi_rank == 0:
            print("MPI - # workers: {}\n".format(mpi_size))
            print("Dask global config - array.chunk-size: {}\n".format(da.config.get('array.chunk-size')))
    else:
        print("Dask global config - array.chunk-size: {}\n".format(da.config.get('array.chunk-size')))

    if not do_chunking:
        fieldset = set_cmems_fieldset(False, deferLoadFlag, periodicFlag)
    elif auto_chunking:
        fieldset = set_cmems_fieldset('auto', deferLoadFlag, periodicFlag)
    else:
        fieldset = set_cmems_fieldset(field_chunksize, deferLoadFlag, periodicFlag)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        process = psutil.Process(os.getpid())
        if mpi_rank == 0:
            global_t_0 = ostime.time()
            global_m_0 = process.memory_info().rss
    else:
        process = psutil.Process(os.getpid())
        global_t_0 = ostime.time()
        global_m_0 = process.memory_info().rss

    if with_ChunkInfoPrint:
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_comm.Barrier()
            if mpi_comm.Get_rank() > 0:
                pass
            else:
                print_field_info(fieldset)
        else:
            print_field_info()

    simStart = None
    for f in fieldset.get_fields():
        if type(f) in [parcels.VectorField, parcels.NestedField, parcels.SummedField]:
            continue
        else:
            if backwardSimulation:
                simStart = f.grid.time_full[-1]
            else:
                simStart = f.grid.time_full[0]
            break

    if backwardSimulation:
        # ==== backward simulation ==== #
        if repeatdtFlag:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * 1e-5, lat=np.random.rand(96, 1) * 1e-5, time=simStart, repeatdt=delta(hours=1))
        else:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * 1e-5, lat=np.random.rand(96, 1) * 1e-5, time=simStart)
    else:
        # ==== forward simulation ==== #
        if repeatdtFlag:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * 1e-5, lat=np.random.rand(96, 1) * 1e-5, repeatdt=delta(hours=1))
        else:
            pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.random.rand(96, 1) * 1e-5, lat=np.random.rand(96, 1) * 1e-5)

    perflog = PerformanceLog()
    postProcessFuncs = [perflog.advance, ]
    if with_GC:
        postProcessFuncs.append(perIterGC)

    if backwardSimulation:
        # ==== backward simulation ==== #
        pset.execute(AdvectionRK4, runtime=delta(days=33), dt=delta(hours=-1), postIterationFunctions=postProcessFuncs)
    else:
        # ==== forward simulation ==== #
        pset.execute(AdvectionRK4, runtime=delta(days=33), dt=delta(hours=1), postIterationFunctions=postProcessFuncs)

    if auto_chunking:
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                for f in fieldset.get_fields():
                    if type(f) in [parcels.VectorField, parcels.NestedField,
                                   parcels.SummedField] or not f.grid.defer_load:
                        continue
                    chunk_info = f.grid.chunk_info
                    auto_field_size = chunk_info[chunk_info[0] + 1]
                    break
        else:
            for f in fieldset.get_fields():
                if type(f) in [parcels.VectorField, parcels.NestedField, parcels.SummedField] or not f.grid.defer_load:
                    continue
                chunk_info = f.grid.chunk_info
                auto_field_size = chunk_info[chunk_info[0]+1]
                break

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        if mpi_comm.Get_rank() > 0:
            pass
        else:
            plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.fds_steps, os.path.join(odir, imageFileName))
    else:
        plot(perflog.samples, perflog.times_steps, perflog.memory_steps, perflog.fds_steps, os.path.join(odir, imageFileName))
