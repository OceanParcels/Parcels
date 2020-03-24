import math
from argparse import ArgumentParser
from datetime import timedelta as delta
from glob import glob
from os import path

import numpy as np
import pytest
import  dask

from parcels import AdvectionRK4
from parcels import FieldSet
from parcels import JITParticle
from parcels import ParticleFile
from parcels import ParticleSet
from parcels import ScipyParticle

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}

def fieldset_from_nemo_3D(chunk_mode):
    data_path = path.join(path.dirname(__file__), 'NemoNorthSeaORCA025-N006_data/')
    ufiles = sorted(glob(data_path + 'ORCA*U.nc'))
    vfiles = sorted(glob(data_path + 'ORCA*V.nc'))
    wfiles = sorted(glob(data_path + 'ORCA*W.nc'))
    mesh_mask = data_path + 'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles}}

    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}
    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = {'U': {'depthu': 75, 'y': 16, 'x': 16},
               'V': {'depthv': 75, 'y': 16, 'x': 16},
               'W': {'depthw': 75, 'y': 16, 'x': 16}}

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
    return fieldset


def fieldset_from_globcurrent(chunk_mode):
    filename = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                             '200201*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = {'U': {'lat': 16, 'lon': 16},
               'V': {'lat': 16, 'lon': 16}}

    fieldset = FieldSet.from_netcdf(filename, variables, dimensions, field_chunksize=chs)
    return fieldset

# ==== undefined as long as we have no POP example data ==== #
def test_pop():
    pass


def compute_nemo_particle_advection(field_set, mode, lonp, latp):

    def periodicBC(particle, fieldSet, time):
        if particle.lon > 15.0:
            particle.lon -= 15.0
        if particle.lon < 0:
            particle.lon += 15.0
        if particle.lat > 60.0:
            particle.lat -= 11.0
        if particle.lat < 49.0:
            particle.lat += 11.0

    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile("nemo_particles_chunk", pset, outputdt=delta(days=1))
    kernels = pset.Kernel(AdvectionRK4) + periodicBC
    pset.execute(kernels, runtime=delta(days=4), dt=delta(hours=6), output_file=pfile)
    return pset


def compute_globcurrent_particle_advection(field_set, mode, lonp, latp):
    pset = ParticleSet(field_set, pclass=ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile("globcurrent_particles_chunk", pset, outputdt=delta(hours=2))
    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5), output_file=pfile)
    return pset


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific'])  # Only testing jit as scipy is very slow
def test_nemo_3D(mode, chunk_mode):
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '2MiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_nemo_3D(chunk_mode)
    # Now run particles as normal
    npart = 20
    lonp = 1.9 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    pset = compute_nemo_particle_advection(field_set, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    if chunk_mode == False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(201.0/16.0)) * int(math.ceil(151.0/16.0))))


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific'])  # Only testing jit as scipy is very slow
def test_globcurrent_2D(mode, chunk_mode):
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '32KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_globcurrent(chunk_mode)
    lonp = [25]
    latp = [-35]
    pset = compute_globcurrent_particle_advection(field_set, mode, lonp, latp)
    # Nemo sample file dimensions: time=UNLIMITED, lat=41, lon=81
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode == False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(41.0/16.0)) * int(math.ceil(81.0/16.0))))
    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)
