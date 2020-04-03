import math
from argparse import ArgumentParser
from datetime import timedelta as delta
from glob import glob
from os import path

import numpy as np
import pytest
import dask

from parcels import AdvectionRK4
from parcels import FieldSet
from parcels import JITParticle
from parcels import ParticleFile
from parcels import ParticleSet
from parcels import ScipyParticle
from parcels import ErrorCode

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def run_nemo_curvilinear(mode, outfile):
    """Function that shows how to read in curvilinear grids, in this case from NEMO"""
    data_path = path.join(path.dirname(__file__), 'NemoCurvilinear_data/')

    filenames = {'U': {'lon': data_path + 'mesh_mask.nc4',
                       'lat': data_path + 'mesh_mask.nc4',
                       'data': data_path + 'U_purely_zonal-ORCA025_grid_U.nc4'},
                 'V': {'lon': data_path + 'mesh_mask.nc4',
                       'lat': data_path + 'mesh_mask.nc4',
                       'data': data_path + 'V_purely_zonal-ORCA025_grid_V.nc4'}}
    variables = {'U': 'U', 'V': 'V'}
    dimensions = {'lon': 'glamf', 'lat': 'gphif'}
    field_chunksize = {'y': 2, 'x': 2}
    field_set = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=field_chunksize)
    assert field_set.U.field_chunksize == field_chunksize

    # Now run particles as normal
    npart = 20
    lonp = 30 * np.ones(npart)
    latp = [i for i in np.linspace(-70, 88, npart)]

    def periodicBC(particle, fieldSet, time):
        if particle.lon > 180:
            particle.lon -= 360

    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile(outfile, pset, outputdt=delta(days=1))
    kernels = pset.Kernel(AdvectionRK4) + periodicBC
    pset.execute(kernels, runtime=delta(days=1)*160, dt=delta(hours=6),
                 output_file=pfile)
    assert np.allclose([pset[i].lat - latp[i] for i in range(len(pset))], 0, atol=2e-2)


def make_plot(trajfile):
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    import cartopy

    class ParticleData(object):
        def __init__(self):
            self.id = []

    def load_particles_file(fname, varnames):
        T = ParticleData()
        pfile = Dataset(fname, 'r')
        T.id = pfile.variables['trajectory'][:]
        for v in varnames:
            setattr(T, v, pfile.variables[v][:])
        return T

    T = load_particles_file(trajfile, ['lon', 'lat', 'time'])
    plt.axes(projection=cartopy.crs.PlateCarree())
    plt.scatter(T.lon, T.lat, c=T.time, s=10)
    plt.show()


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
def test_nemo_curvilinear(mode, tmpdir):
    outfile = tmpdir.join('nemo_particles')
    run_nemo_curvilinear(mode, outfile)


def test_nemo_3D_samegrid():
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

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

    assert fieldset.U.dataFiles is not fieldset.W.dataFiles


def fieldset_nemo_setup():
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

    return filenames, variables, dimensions


def compute_particle_advection(field_set, mode, lonp, latp):

    def periodicBC(particle, fieldSet, time):
        if particle.lon > 15.0:
            particle.lon -= 15.0
        if particle.lon < 0:
            particle.lon += 15.0
        if particle.lat > 60.0:
            particle.lat -= 11.0
        if particle.lat < 49.0:
            particle.lat += 11.0

    def OutOfBounds_reinitialisation(particle, fieldset, time):
        particle.lat = 2.5
        particle.lon = 52.0 + (-1e-3 + np.random.rand() * 2.0 * 1e-3)

    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile("nemo_particles", pset, outputdt=delta(days=1))
    kernels = pset.Kernel(AdvectionRK4) + periodicBC
    pset.execute(kernels, runtime=delta(days=4), dt=delta(hours=6),
                 output_file=pfile, recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds_reinitialisation})
    return pset


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
def test_nemo_curvilinear_auto_chunking(mode):
    dask.config.set({'array.chunk-size': '2MiB'})
    filenames, variables, dimensions = fieldset_nemo_setup()
    field_set = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize='auto')
    assert field_set.U.dataFiles is not field_set.W.dataFiles
    npart = 20
    lonp = 2.5 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_particle_advection(field_set, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) != 1)


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
def test_nemo_curvilinear_no_chunking(mode):
    dask.config.set({'array.chunk-size': '128MiB'})
    filenames, variables, dimensions = fieldset_nemo_setup()
    field_set = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=False)
    assert field_set.U.dataFiles is not field_set.W.dataFiles
    npart = 20
    lonp = 2.5 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_particle_advection(field_set, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == 1)


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
def test_nemo_curvilinear_specific_chunking(mode):
    dask.config.set({'array.chunk-size': '128MiB'})
    filenames, variables, dimensions = fieldset_nemo_setup()
    chs = {'U': {'depthu': 75, 'y': 16, 'x': 16},
           'V': {'depthv': 75, 'y': 16, 'x': 16},
           'W': {'depthw': 75, 'y': 16, 'x': 16}}

    field_set = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
    assert field_set.U.dataFiles is not field_set.W.dataFiles
    npart = 20
    lonp = 2.5 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_particle_advection(field_set, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(201.0/16.0)) * int(math.ceil(151.0/16.0))))


if __name__ == "__main__":
    p = ArgumentParser(description="""Chose the mode using mode option""")
    p.add_argument('--mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
    args = p.parse_args()

    outfile = "nemo_particles"

    run_nemo_curvilinear(args.mode, outfile)
    make_plot(outfile+'.nc')
