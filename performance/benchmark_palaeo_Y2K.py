# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:31:22 2017

@author: nooteboom
"""
import itertools

from parcels import (FieldSet, JITParticle, AdvectionRK4_3D,
                     Field, ErrorCode, ParticleFile, Variable)
# from parcels import ParticleSet
# from parcels.particleset_vectorized_benchmark import ParticleSet_Benchmark
from parcels.particleset_node_benchmark import ParticleSet_Benchmark
from parcels.tools import idgen
from parcels.tools import GenerateID_Service, SpatioTemporalIdGenerator, SequentialIdGenerator
from argparse import ArgumentParser
from datetime import timedelta as delta
from datetime import datetime
import numpy as np
import math
from glob import glob
import sys
import pandas as pd

import os
import time as ostime
import matplotlib.pyplot as plt
import fnmatch

# import dask
import gc

try:
    from mpi4py import MPI
except:
    MPI = None

import warnings
import xarray as xr
warnings.simplefilter("ignore", category=xr.SerializationWarning)

global_t_0 = 0
odir = ""


def set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile, mesh_mask='/scratch/ckehl/experiments/palaeo-parcels/NEMOdata/domain/coordinates.nc', periodicFlag=False):
    filenames = { 'U': {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':ufiles},
                'V' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':vfiles},
                'W' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'depth': [ufiles[0]],
                        'data':wfiles},  
                'S' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'data':tfiles},   
                'T' : {'lon': mesh_mask,
                        'lat': mesh_mask,
                        'data':tfiles},
                'NO3':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [pfiles[0]],
                       'data':pfiles},
                'ICE':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'data':ifiles},
                'ICEPRES':{'lon': mesh_mask,
                           'lat': mesh_mask,
                           'data':ifiles},
                'CO2':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'data':dfiles},
                'PP':{'lon': mesh_mask,
                       'lat': mesh_mask,
                       'depth': [dfiles[0]],
                       'data':dfiles},
                }
    if mesh_mask:
        filenames['mesh_mask'] = mesh_mask
    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'T': 'sst',
                 'S': 'sss',
                 'NO3': 'DIN',                 
                 'PP': 'TPP3',                
                 'ICE': 'sit',
                 'ICEPRES': 'ice_pres',
                 'CO2': 'TCO2' }

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthu', 'time': 'time_counter'},  #
                  'T': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'S': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'NO3': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'deptht', 'time': 'time_counter'},
                  'ICE': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'ICEPRES': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'CO2': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'PP': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'deptht', 'time': 'time_counter'},
                  }
    bfiles = {'lon': mesh_mask, 'lat': mesh_mask, 'data': [bfile, ]}
    bvariables = ('B', 'Bathymetry')
    bdimensions = {'lon': 'glamf', 'lat': 'gphif'}
    bchs = False

    # ==== depth-split need to be 75 so all (2D- and 3D) fields are chunked the same way
    chs = {'time_counter': 1, 'depthu': 80, 'depthv': 80, 'depthw': 80, 'deptht': 80, 'y': 200, 'x': 200}
    # chs = {'U': {'depthu': 80, 'depthv': 80, 'depthw': 80, 'deptht': 80, 'y': 64, 'x': 128, 'time_counter': 1},
    #        'V': {'depthu': 80, 'depthv': 80, 'depthw': 80, 'deptht': 80, 'y': 64, 'x': 128, 'time_counter': 1},
    #        'W': {'depthu': 80, 'depthv': 80, 'depthw': 80, 'deptht': 80, 'y': 64, 'x': 128, 'time_counter': 1},
    #        'T':   {'x': 128, 'y': 64, 'deptht': 80, 'time_counter': 1},  # tfiles
    #        'S':   {'x': 128, 'y': 64, 'deptht': 80, 'time_counter': 1},  # tfiles
    #        'NO3': {'x': 128, 'y': 64, 'deptht': 80, 'time_counter': 1},  # pfiles
    #        'PP':  {'x': 128, 'y': 64, 'deptht': 80, 'time_counter': 1},  # dfiles
    #        'ICE':     False,  # ifiles
    #        'ICEPRES': False,  # ifiles
    #        'CO2': {'x': 128, 'y': 64, 'deptht': 80, 'time_counter': 1},  # dfiles
    #        }

    # dask.config.set({'array.chunk-size': '6MiB'})
    # chs = 'auto'
    nchs = {
        'U':       {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthu', 25), 'time': ('time_counter', 1)},  # ufiles
        'V':       {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthv', 25), 'time': ('time_counter', 1)},  # vfiles
        'W':       {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthw', 25), 'time': ('time_counter', 1)},  # wfiles
        'T':       {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},  # tfiles
        'S':       {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},  # tfiles
        'NO3':     {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('deptht', 25), 'time': ('time_counter', 1)},  # pfiles
        'ICE':     {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},  # ifiles
        'ICEPRES': {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},  # ifiles
        'CO2':     {'lon': ('x', 128), 'lat': ('y', 96), 'time': ('time_counter', 1)},  # dfiles
        'PP':      {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('deptht', 25), 'time': ('time_counter', 1)},  # dfiles
    }

    if mesh_mask: # and isinstance(bfile, list) and len(bfile) > 0:
        if not periodicFlag:
            try:
                fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True, field_chunksize=chs)
                # Tfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
                # Sfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
                # NO3field = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
                # PPfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
                # ICEfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
                # ICEPRESfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
                # CO2field = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
                Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
            except (SyntaxError, ):
                fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True, chunksize=nchs)
                Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', chunksize=bchs)
        else:
            try:
                fieldset = FieldSet.from_nemo(filenames, variables, dimensions, time_periodic=delta(days=366), field_chunksize=chs)
                Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', field_chunksize=bchs)
            except (SyntaxError, ):
                fieldset = FieldSet.from_nemo(filenames, variables, dimensions, time_periodic=delta(days=366), chunksize=nchs)
                Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='cgrid_tracer', chunksize=bchs)
        fieldset.add_field(Bfield, 'B')
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10
        return fieldset
    else:
        filenames.pop('B')
        variables.pop('B')
        dimensions.pop('B')
        if not periodicFlag:
            try:
                # fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=False, field_chunksize=chs)
                fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, field_chunksize=chs)
            except (SyntaxError, ):
                fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, chunksize=nchs)
        else:
            try:
                fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, time_periodic=delta(days=366), field_chunksize=chs)
            except (SyntaxError, ):
                fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, time_periodic=delta(days=366), chunksize=nchs)
        fieldset.U.vmax = 10
        fieldset.V.vmax = 10
        fieldset.W.vmax = 10
        return fieldset
        

def periodicBC(particle, fieldSet, time):
    if particle.lon > 180:
        particle.lon -= 360
    if particle.lon < -180:
        particle.lon += 360

def Sink(particle, fieldset, time):
    if(particle.depth>fieldset.dwellingdepth):
        particle.depth = particle.depth + fieldset.sinkspeed * particle.dt
    elif(particle.depth<=fieldset.dwellingdepth and particle.depth>1):
        particle.depth = fieldset.surface
        particle.temp = fieldset.T[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.salin = fieldset.S[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.PP = fieldset.PP[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.NO3 = fieldset.NO3[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.ICE = fieldset.ICE[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.ICEPRES = fieldset.ICEPRES[time+particle.dt, fieldset.surface, particle.lat, particle.lon] 
        particle.CO2 = fieldset.CO2[time+particle.dt, fieldset.surface, particle.lat, particle.lon]  
        particle.delete()

def Age(particle, fieldset, time):
    if particle.state == ErrorCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    # if particle.age > fieldset.maxage:
    #     particle.delete()

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def perIterGC():
    gc.collect()

def initials(particle, fieldset, time):
    if particle.age==0.:
        particle.depth = fieldset.B[time, fieldset.surface, particle.lat, particle.lon]
        if(particle.depth  > 5800.):
            particle.age = (particle.depth - 5799.)*fieldset.sinkspeed
            particle.depth = 5799.        
        particle.lon0 = particle.lon
        particle.lat0 = particle.lat
        particle.depth0 = particle.depth


class DinoParticle(JITParticle):
    temp = Variable('temp', dtype=np.float32, initial=np.nan)
    age = Variable('age', dtype=np.float32, initial=0.)
    salin = Variable('salin', dtype=np.float32, initial=np.nan)
    lon0 = Variable('lon0', dtype=np.float32, initial=0.)
    lat0 = Variable('lat0', dtype=np.float32, initial=0.)
    depth0 = Variable('depth0',dtype=np.float32, initial=0.)
    PP = Variable('PP',dtype=np.float32, initial=np.nan)
    NO3 = Variable('NO3',dtype=np.float32, initial=np.nan)
    ICE = Variable('ICE',dtype=np.float32, initial=np.nan)
    ICEPRES = Variable('ICEPRES',dtype=np.float32, initial=np.nan)
    CO2 = Variable('CO2',dtype=np.float32, initial=np.nan)

if __name__ == "__main__":
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-sp", "--sinking_speed", dest="sp", type=float, default=11.0, help="set the simulation sinking speed in [m/day] (default: 11.0)")
    parser.add_argument("-dd", "--dwelling_depth", dest="dd", type=float, default=10.0, help="set the dwelling depth (i.e. ocean surface depth) in [m] (default: 10.0)")
    # parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 365)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=str, default="1*365", help="runtime in days (default: 1*365)")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-pr", "--profiling", dest="profiling", action='store_true', default=False, help="tells that the profiling of the script is activates")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_false', default=True, help="write data in outfile (default: true)")
    args = parser.parse_args()

    sp = args.sp # The sinkspeed m/day
    dd = args.dd  # The dwelling depth
    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    time_in_days = int(float(eval(args.time_in_days)))
    time_in_years = int(time_in_days/366.0)
    with_GC = args.useGC

    outdt = 24
    cbdt = 24  # np.infty

    # ==== this is not a good choice for long-running simulations (e.g. 10y+) - needs to be adapted to scale ==== #
    # ==== also, it is a backward simulation, so the high-value should be first.                             ==== #
    idgen = GenerateID_Service(SequentialIdGenerator)
    idgen.setTimeLine(0, delta(days=time_in_days).total_seconds())
    # idgen.setTimeLine(delta(days=time_in_days).total_seconds(), 0)

    branch = "nodes"
    computer_env = "local/unspecified"
    scenario = "palaeo-parcels"
    headdir = ""
    odir = ""
    dirread_pal = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments/palaeo-parcels".format("ckehl")
        odir = os.path.join(headdir,"BENCHres")
        dirread_pal = os.path.join(headdir,'NEMOdata')
        datahead = "/data/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        computer_env = "Gemini"
    # elif fnmatch.fnmatchcase(os.uname()[1], "int?.*"):  # Cartesius
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments/palaeo-parcels".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "BENCHres")
        dirread_pal = os.path.join(headdir,'NEMOdata')
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA_BGC/ORCA0083-N006/')
        computer_env = "Cartesius"
    else:
        headdir = "/var/scratch/nooteboom"
        odir = os.path.join(headdir, "BENCHres")
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
        dirread_top_bgc = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')


    # dirread_pal = '/projects/0/palaeo-parcels/NEMOdata/'

    outfile = 'grid_dd' + str(int(dd))
    outfile += '_sp' + str(int(sp))
    if periodicFlag:
        outfile += '_p'
    if time_in_years != 1:
        outfile += '_' + str(time_in_years) + 'y'
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_size = mpi_comm.Get_size()
        outfile += '_n' + str(mpi_size)
    if args.profiling:
        outfile += '_prof'
    if with_GC:
        outfile += '_wGC'
    else:
        outfile += '_woGC'
    dirwrite = os.path.join(odir, "sp%d_dd%d" % (int(sp),int(dd)))
    if not os.path.exists(dirwrite):
        os.mkdir(dirwrite)

    latsz = np.array(pd.read_csv(os.path.join(headdir,"TF_locationsSurfaceSamples_forPeter.csv")).Latitude.tolist())
    lonsz = np.array(pd.read_csv(os.path.join(headdir,"TF_locationsSurfaceSamples_forPeter.csv")).Longitude.tolist())
    numlocs = np.logical_and(latsz<1000, lonsz<1000)
    latsz = latsz[numlocs]
    lonsz = lonsz[numlocs]

    assert ~(np.isnan(latsz)).any(), 'locations should not contain any NaN values'
    dep = dd * np.ones(latsz.shape)

    # timesz = np.array([datetime(2000, 12, 25) - delta(days=x) for x in range(0,int(365),3)])
    timesz = np.array([datetime(2000, 12, 31) - delta(days=x) for x in range(0, time_in_days, 3)])
    # timesz = np.array([datetime(2000, 12, 25) - delta(days=x) for x in range(0, time_in_days, 3)])
    times = np.empty(shape=(0))
    depths = np.empty(shape=(0))
    lons = np.empty(shape=(0))
    lats = np.empty(shape=(0))
    for i in range(len(timesz)):
        lons = np.append(lons,lonsz)
        lats = np.append(lats, latsz)
        depths = np.append(depths, np.zeros(len(lonsz), dtype=np.float32))
        times = np.append(times, np.full(len(lonsz),timesz[i]))
    del timesz
    del lonsz
    del latsz

    print("running {} on {} (uname: {}) - branch '{}' - (target) N: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, lons.shape[0], sys.argv[1:]))


    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # global_t_0 = MPI.Wtime()
            global_t_0 = ostime.process_time()
    else:
        # global_t_0 = ostime.time()
        global_t_0 = ostime.process_time()

    ufiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_2000????d05U.nc'))
    vfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_2000????d05V.nc'))
    wfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_2000????d05W.nc'))
    tfiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_2000????d05T.nc'))
    pfiles = sorted(glob(dirread_top_bgc + 'means/ORCA0083-N06_2000????d05P.nc'))
    dfiles = sorted(glob(dirread_top_bgc + 'means/ORCA0083-N06_2000????d05D.nc'))
    ifiles = sorted(glob(dirread_top + 'means/ORCA0083-N06_2000????d05I.nc'))
    bfile = dirread_top + 'domain/bathymetry_ORCA12_V3.3.nc'

    fieldset = set_nemo_fieldset(ufiles, vfiles, wfiles, tfiles, pfiles, dfiles, ifiles, bfile, os.path.join(dirread_pal, "domain/coordinates.nc"), periodicFlag=periodicFlag)
    fieldset.add_periodic_halo(zonal=True) 
    fieldset.add_constant('dwellingdepth', np.float(dd))
    fieldset.add_constant('sinkspeed', sp/86400.)
    fieldset.add_constant('maxage', 300000.*86400)
    fieldset.add_constant('surface', 2.5)

    print("|lon| = {}; |lat| = {}; |depth| = {}, |times| = {}".format(lonsz.shape[0], latsz.shape[0], dep.shape[0], times.shape[0]))

    # ==== Set min/max depths in the fieldset ==== #
    fs_depths = fieldset.U.depth
    idgen.setDepthLimits(np.min(fs_depths), np.max(fs_depths))

    pset = ParticleSet_Benchmark.from_list(fieldset=fieldset, pclass=DinoParticle, lon=lons.tolist(), lat=lats.tolist(), depth=depths.tolist(), time = times)

    """ Kernal + Execution"""
    postProcessFuncs = []
    if with_GC:
        postProcessFuncs.append(perIterGC)
    output_fpath = None
    pfile = None
    if args.write_out:
        output_fpath = os.path.join(dirwrite, outfile)
        pfile = pset.ParticleFile(output_fpath, outputdt=outdt, convert_at_end=True, write_ondelete=True)
    kernels = pset.Kernel(initials) + Sink + Age  + pset.Kernel(AdvectionRK4_3D) + Age

    starttime = 0
    endtime = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # starttime = ostime.time()
            # starttime = MPI.Wtime()
            starttime = ostime.process_time()
    else:
        # starttime = ostime.time()
        starttime = ostime.process_time()
    # pset.execute(kernels, runtime=delta(days=365*9), dt=delta(minutes=-20), output_file=pfile, verbose_progress=False,
    # recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs)
    # postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12)
    pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(hours=-12), output_file=pfile, verbose_progress=False, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs, callbackdt=cbdt)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # endtime = ostime.time()
            # endtime = MPI.Wtime()
            endtime = ostime.process_time()
    else:
        #endtime = ostime.time()
        endtime = ostime.process_time()

    if args.write_out:
        pfile.close()

    size_Npart = len(pset.nparticle_log)
    Npart = pset.nparticle_log.get_param(size_Npart - 1)
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()
        Npart = mpi_comm.reduce(Npart, op=MPI.SUM, root=0)
        if mpi_comm.Get_rank() == 0:
            if size_Npart>0:
                sys.stdout.write("final # particles: {}\n".format( Npart ))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
            avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
    else:
        if size_Npart > 0:
            sys.stdout.write("final # particles: {}\n".format( Npart ))
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
        avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
        sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        # mpi_comm.Barrier()
        Nparticles = mpi_comm.reduce(np.array(pset.nparticle_log.get_params()), op=MPI.SUM, root=0)
        Nmem = mpi_comm.reduce(np.array(pset.mem_log.get_params()), op=MPI.SUM, root=0)
        if mpi_comm.Get_rank() == 0:
            pset.plot_and_log(memory_used=Nmem, nparticles=Nparticles, target_N=1, imageFilePath=imageFileName, odir=odir)
    else:
        pset.plot_and_log(target_N=1, imageFilePath=imageFileName, odir=odir)

    print('Execution finished')
    exit(0)



