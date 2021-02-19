import math
from datetime import timedelta as delta
from glob import glob
from os import path

import numpy as np
import pytest
import dask

from parcels import AdvectionRK4
from parcels import Field
from parcels import FieldSet
from parcels import JITParticle
from parcels import ParticleFile
from parcels import ParticleSet
from parcels import ScipyParticle
from parcels import Variable
from parcels.tools.statuscodes import DaskChunkingError

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
        chs = {'U': {'depth': ('depthu', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
               'V': {'depth': ('depthv', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
               'W': {'depth': ('depthw', 75), 'lat': ('y', 16), 'lon': ('x', 16)}}
    elif chunk_mode == 'failsafe':  # chunking time and but not depth
        filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
                     'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles}}
        variables = {'U': 'uo',
                     'V': 'vo'}
        dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                      'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}
        chs = {'U': {'time': ('time_counter', 1), 'depth': ('depthu', 25), 'lat': ('y', -1), 'lon': ('x', -1)},
               'V': {'time': ('time_counter', 1), 'depth': ('time_counter', 25), 'lat': ('y', 8), 'lon': ('x', 8)}}

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=chs)
    return fieldset


def fieldset_from_globcurrent(chunk_mode):
    filenames = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                          '200201*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = {'U': {'lat': ('lat', 8), 'lon': ('lon', 8)},
               'V': {'lat': ('lat', 8), 'lon': ('lon', 8)}}
    elif chunk_mode == 'failsafe':  # chunking time but not lat
        chs = {'U': {'time': ('time', 1), 'lat': ('lat', 10), 'lon': ('lon', -1)},
               'V': {'time': ('time', 1), 'lat': ('lat', -1), 'lon': ('lon', -1)}}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, chunksize=chs)
    return fieldset


def fieldset_from_pop_1arcs(chunk_mode):
    filenames = path.join(path.join(path.dirname(__file__), 'POPSouthernOcean_data'), 't.x1_SAMOC_flux.1690*.nc')
    variables = {'U': 'UVEL', 'V': 'VVEL', 'W': 'WVEL'}
    timestamps = np.expand_dims(np.array([np.datetime64('2000-%.2d-01' % m) for m in range(1, 7)]), axis=1)
    dimensions = {'lon': 'ULON', 'lat': 'ULAT', 'depth': 'w_dep'}
    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = {'lon': ('i', 8), 'lat': ('j', 8), 'depth': ('k', 3)}
    elif chunk_mode == 'failsafe':  # here: bad depth entry
        chs = {'depth': ('wz', 3), 'lat': ('j', 8), 'lon': ('i', 8)}

    fieldset = FieldSet.from_pop(filenames, variables, dimensions, chunksize=chs, timestamps=timestamps)
    return fieldset


def fieldset_from_swash(chunk_mode):
    filenames = path.join(path.join(path.dirname(__file__), 'SWASH_data'), 'field_*.nc')
    variables = {'U': 'cross-shore velocity',
                 'V': 'along-shore velocity',
                 'W': 'vertical velocity',
                 'depth_w': 'time varying depth',
                 'depth_u': 'time varying depth_u'}
    dimensions = {'U': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'V': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'W': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'depth_w': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'depth_u': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'}}
    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = {'U': {'depth': ('z_u', 6), 'lat': ('y', 4), 'lon': ('x', 4)},
               'V': {'depth': ('z_u', 6), 'lat': ('y', 4), 'lon': ('x', 4)},
               'W': {'depth': ('z', 7), 'lat': ('y', 4), 'lon': ('x', 4)},
               'depth_w': {'depth': ('z', 7), 'lat': ('y', 4), 'lon': ('x', 4)},
               'depth_u': {'depth': ('z_u', 6), 'lat': ('y', 4), 'lon': ('x', 4)}
               }
    elif chunk_mode == 'failsafe':  # here: incorrect matching between dimension names and their attachment to the NC-variables
        chs = {'U': {'depth': ('depth', 7), 'lat': ('y', 4), 'lon': ('x', 4)},
               'V': {'depth': ('z_u', 6), 'lat': ('y', 4), 'lon': ('x', 4)},
               'W': {'depth': ('z', 7), 'lat': ('y', 4), 'lon': ('x', 4)},
               'depth_w': {'depth': ('z', 7), 'lat': ('y', 4), 'lon': ('x', 4)},
               'depth_u': {'depth': ('z_u', 6), 'lat': ('y', 4), 'lon': ('x', 4)}
               }
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='flat', allow_time_extrapolation=True, chunksize=chs)
    fieldset.U.set_depth_from_field(fieldset.depth_u)
    fieldset.V.set_depth_from_field(fieldset.depth_u)
    fieldset.W.set_depth_from_field(fieldset.depth_w)
    return fieldset


def fieldset_from_ofam(chunk_mode):
    filenames = {'U': path.join(path.dirname(__file__), 'OFAM_example_data', 'OFAM_simple_U.nc'),
                 'V': path.join(path.dirname(__file__), 'OFAM_example_data', 'OFAM_simple_V.nc')}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'depth': 'st_ocean',
                  'time': 'Time'}

    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = {'lon': ('xu_ocean', 100), 'lat': ('yu_ocean', 50), 'depth': ('st_edges_ocean', 60), 'time': ('Time', 1)}
    return FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, chunksize=chs)


def fieldset_from_mitgcm(chunk_mode, using_add_field=True):
    data_path = path.join(path.dirname(__file__), "MITgcm_example_data/")
    filenames = {"U": data_path + "mitgcm_UV_surface_zonally_reentrant.nc",
                 "V": data_path + "mitgcm_UV_surface_zonally_reentrant.nc"}
    variables = {"U": "UVEL", "V": "VVEL"}
    dimensions = {"U": {"lon": "XG", "lat": "YG", "time": "time"},
                  "V": {"lon": "XG", "lat": "YG", "time": "time"}}

    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific_same':
        chs = {'U': {'lat': ('YC', 50), 'lon': ('XG', 100)},
               'V': {'lat': ('YG', 50), 'lon': ('XC', 100)}}
    elif chunk_mode == 'specific_different':
        chs = {'U': {'lat': ('YC', 50), 'lon': ('XG', 100)},
               'V': {'lat': ('YG', 40), 'lon': ('XC', 100)}}
    if using_add_field:
        if chs in [False, 'auto']:
            chs = {'U': chs, 'V': chs}
        fieldset = FieldSet.from_mitgcm(filenames['U'], {'U': variables['U']}, dimensions['U'],
                                        mesh='flat', chunksize=chs['U'])
        fieldset2 = FieldSet.from_mitgcm(filenames['V'], {'V': variables['V']}, dimensions['V'],
                                         mesh='flat', chunksize=chs['V'])
        fieldset.add_field(fieldset2.V)
    else:
        fieldset = FieldSet.from_mitgcm(filenames, variables, dimensions, mesh='flat', chunksize=chs)
    return fieldset


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


def compute_pop_particle_advection(field_set, mode, lonp, latp):
    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile("globcurrent_particles_chunk", pset, outputdt=delta(days=15))
    pset.execute(AdvectionRK4, runtime=delta(days=90), dt=delta(days=2), output_file=pfile)
    return pset


def compute_swash_particle_advection(field_set, mode, lonp, latp, depthp):
    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp, depth=depthp)
    pfile = ParticleFile("swash_particles_chunk", pset, outputdt=delta(seconds=0.05))
    pset.execute(AdvectionRK4, runtime=delta(seconds=0.2), dt=delta(seconds=0.005), output_file=pfile)
    return pset


def compute_ofam_particle_advection(field_set, mode, lonp, latp, depthp):
    pset = ParticleSet(field_set, pclass=ptype[mode], lon=lonp, lat=latp, depth=depthp)
    pfile = ParticleFile("ofam_particles_chunk", pset, outputdt=delta(days=1))
    pset.execute(AdvectionRK4, runtime=delta(days=10), dt=delta(minutes=5), output_file=pfile)
    return pset


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific', 'failsafe'])
def test_nemo_3D(mode, chunk_mode):
    if chunk_mode in ['auto', ]:
        dask.config.set({'array.chunk-size': '256KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_nemo_3D(chunk_mode)
    npart = 20
    lonp = 2.5 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(field_set, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    if chunk_mode != 'failsafe':
        assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
        assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert field_set.gridset.size == 3  # because three different grids in 'auto' mode
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(75.0/75.0)) * int(math.ceil(201.0/16.0)) * int(math.ceil(151.0/16.0))))
    elif chunk_mode == 'failsafe':  # chunking time and depth but not lat and lon
        assert (len(field_set.U.grid.load_chunk) != 1)
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(75.0/25.0)) * int(math.ceil(201.0/201.0)) * int(math.ceil(151.0/151.0))))
        assert (len(field_set.V.grid.load_chunk) != 1)
        assert (len(field_set.V.grid.load_chunk) == (1 * int(math.ceil(75.0/75.0)) * int(math.ceil(201.0/8.0)) * int(math.ceil(151.0/8.0))))


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific', 'failsafe'])
def test_globcurrent_2D(mode, chunk_mode):
    if chunk_mode in ['auto', ]:
        dask.config.set({'array.chunk-size': '16KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_globcurrent(chunk_mode)
    lonp = [25]
    latp = [-35]
    pset = None
    try:
        pset = compute_globcurrent_particle_advection(field_set, mode, lonp, latp)
    except DaskChunkingError:
        # we removed the failsafe, so now if all chunksize dimensions are incorrect, there is nothing left to chunk,
        # which raises an error saying so. This is the expected behaviour
        if chunk_mode == 'failsafe':
            return True
    # GlobCurrent sample file dimensions: time=UNLIMITED, lat=41, lon=81
    if chunk_mode != 'failsafe':  # chunking time but not lat
        assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(41.0/8.0)) * int(math.ceil(81.0/8.0))))
    elif chunk_mode == 'failsafe':  # chunking time but not lat
        assert (len(field_set.U.grid.load_chunk) != 1)
        assert (len(field_set.V.grid.load_chunk) != 1)
    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific', 'failsafe'])
def test_pop(mode, chunk_mode):
    if chunk_mode in ['auto', ]:
        dask.config.set({'array.chunk-size': '256KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_pop_1arcs(chunk_mode)
    npart = 20
    lonp = 70.0 * np.ones(npart)
    latp = [i for i in -45.0+(-0.25+np.random.rand(npart)*2.0*0.25)]
    compute_pop_particle_advection(field_set, mode, lonp, latp)
    # POP sample file dimensions: k=21, j=60, i=60
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    if chunk_mode is False:
        assert field_set.gridset.size == 1
        assert (len(field_set.U.grid.load_chunk) == 1)
        assert (len(field_set.V.grid.load_chunk) == 1)
        assert (len(field_set.W.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert field_set.gridset.size == 3  # because three different grids in 'auto' mode
        assert (len(field_set.U.grid.load_chunk) != 1)
        assert (len(field_set.V.grid.load_chunk) != 1)
        assert (len(field_set.W.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert field_set.gridset.size == 1
        assert (len(field_set.U.grid.load_chunk) == (int(math.ceil(21.0/3.0)) * int(math.ceil(60.0/8.0)) * int(math.ceil(60.0/8.0))))
    elif chunk_mode == 'failsafe':   # here: done a typo in the netcdf dimname field
        assert field_set.gridset.size == 1
        assert (len(field_set.U.grid.load_chunk) != 1)
        assert (len(field_set.V.grid.load_chunk) != 1)
        assert (len(field_set.W.grid.load_chunk) != 1)
        assert (len(field_set.U.grid.load_chunk) == (int(math.ceil(21.0/3.0)) * int(math.ceil(60.0/8.0)) * int(math.ceil(60.0/8.0))))


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific', 'failsafe'])
def test_swash(mode, chunk_mode):
    if chunk_mode in ['auto', ]:
        dask.config.set({'array.chunk-size': '32KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_swash(chunk_mode)
    npart = 20
    lonp = [i for i in 9.5 + (-0.2 + np.random.rand(npart) * 2.0 * 0.2)]
    latp = [i for i in np.arange(start=12.3, stop=13.1, step=0.04)[0:20]]
    depthp = [-0.1, ] * npart
    compute_swash_particle_advection(field_set, mode, lonp, latp, depthp)
    # SWASH sample file dimensions: t=1, z=7, z_u=6, y=21, x=51
    if chunk_mode not in ['failsafe', ]:
        assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk)), "U {} vs V {}".format(field_set.U.grid.chunk_info, field_set.V.grid.chunk_info)
    if chunk_mode not in ['failsafe', 'auto']:
        assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk)), "U {} vs W {}".format(field_set.U.grid.chunk_info, field_set.W.grid.chunk_info)
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    else:
        assert (len(field_set.U.grid.load_chunk) != 1)
        assert (len(field_set.V.grid.load_chunk) != 1)
        assert (len(field_set.W.grid.load_chunk) != 1)
    if chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(6.0 / 6.0)) * int(math.ceil(21.0 / 4.0)) * int(math.ceil(51.0 / 4.0))))
        assert (len(field_set.V.grid.load_chunk) == (1 * int(math.ceil(6.0 / 6.0)) * int(math.ceil(21.0 / 4.0)) * int(math.ceil(51.0 / 4.0))))
        assert (len(field_set.W.grid.load_chunk) == (1 * int(math.ceil(7.0 / 7.0)) * int(math.ceil(21.0 / 4.0)) * int(math.ceil(51.0 / 4.0))))


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific'])
def test_ofam_3D(mode, chunk_mode):
    if chunk_mode in ['auto', ]:
        dask.config.set({'array.chunk-size': '1024KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_ofam(chunk_mode)
    lonp = [180]
    latp = [10]
    depthp = [2.5]  # the depth of the first layer in OFAM
    pset = compute_ofam_particle_advection(field_set, mode, lonp, latp, depthp)
    # OFAM sample file dimensions: time=UNLIMITED, st_ocean=1, st_edges_ocean=52, lat=601, lon=2001
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        numblocks = [i for i in field_set.U.grid.chunk_info[1:3]]
        dblocks = 1
        vblocks = 0
        for bsize in field_set.U.grid.chunk_info[3:3+numblocks[0]]:
            vblocks += bsize
        ublocks = 0
        for bsize in field_set.U.grid.chunk_info[3+numblocks[0]:3+numblocks[0]+numblocks[1]]:
            ublocks += bsize
        matching_numblocks = (ublocks == 2001 and vblocks == 601 and dblocks == 1)
        matching_fields = (field_set.U.grid.chunk_info == field_set.V.grid.chunk_info)
        matching_uniformblocks = (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(1.0/60.0)) * int(math.ceil(601.0/50.0)) * int(math.ceil(2001.0/100.0))))
        assert (matching_uniformblocks or (matching_fields and matching_numblocks))
    assert(abs(pset[0].lon - 173) < 1)
    assert(abs(pset[0].lat - 11) < 1)


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific_same', 'specific_different'])
@pytest.mark.parametrize('using_add_field', [False, True])
def test_mitgcm(mode, chunk_mode, using_add_field):
    if chunk_mode in ['auto', ]:
        dask.config.set({'array.chunk-size': '256KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_mitgcm(chunk_mode, using_add_field)
    lons, lats = 5e5, 5e5

    pset = ParticleSet.from_list(fieldset=field_set, pclass=ptype[mode], lon=lons, lat=lats)
    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5))
    # MITgcm sample file dimensions: time=10, XG=400, YG=200
    if chunk_mode != 'specific_different':
        assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode in [False, ]:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode in ['auto', ]:
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif 'specific' in chunk_mode:
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(400.0/50.0)) * int(math.ceil(200.0/100.0))))
    if chunk_mode == 'specific_same':
        assert field_set.gridset.size == 1
    elif chunk_mode == 'specific_different':
        assert field_set.gridset.size == 2
    assert np.allclose(pset[0].lon, 5.27e5, atol=1e3)


@pytest.mark.parametrize('mode', ['jit'])
def test_diff_entry_dimensions_chunks(mode):
    data_path = path.join(path.dirname(__file__), 'NemoNorthSeaORCA025-N006_data/')
    ufiles = sorted(glob(data_path + 'ORCA*U.nc'))
    vfiles = sorted(glob(data_path + 'ORCA*V.nc'))
    mesh_mask = data_path + 'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'data': vfiles}}
    variables = {'U': 'uo',
                 'V': 'vo'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}
    chs = {'U': {'depth': ('depthu', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
           'V': {'depth': ('depthv', 75), 'lat': ('y', 16), 'lon': ('x', 16)}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=chs)
    npart = 20
    lonp = 5.2 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(fieldset, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    assert (len(fieldset.U.grid.load_chunk) == len(fieldset.V.grid.load_chunk))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_3d_2dfield_sampling(mode):
    data_path = path.join(path.dirname(__file__), 'NemoNorthSeaORCA025-N006_data/')
    ufiles = sorted(glob(data_path + 'ORCA*U.nc'))
    vfiles = sorted(glob(data_path + 'ORCA*V.nc'))
    mesh_mask = data_path + 'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'data': vfiles},
                 'nav_lon': {'lon': mesh_mask, 'lat': mesh_mask, 'data': [ufiles[0], ]}}
    variables = {'U': 'uo',
                 'V': 'vo',
                 'nav_lon': 'nav_lon'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'nav_lon': {'lon': 'glamf', 'lat': 'gphif'}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=False)
    fieldset.nav_lon.data = np.ones(fieldset.nav_lon.data.shape, dtype=np.float32)
    fieldset.add_field(Field('rectilinear_2D', np.ones((2, 2)),
                             lon=np.array([-10, 20]), lat=np.array([40, 80]), chunksize=False))

    class MyParticle(ptype[mode]):
        sample_var_curvilinear = Variable('sample_var_curvilinear')
        sample_var_rectilinear = Variable('sample_var_rectilinear')
    pset = ParticleSet(fieldset, pclass=MyParticle, lon=2.5, lat=52)

    def Sample2D(particle, fieldset, time):
        particle.sample_var_curvilinear += fieldset.nav_lon[time, particle.depth, particle.lat, particle.lon]
        particle.sample_var_rectilinear += fieldset.rectilinear_2D[time, particle.depth, particle.lat, particle.lon]

    runtime, dt = 86400*4, 6*3600
    pset.execute(pset.Kernel(AdvectionRK4) + Sample2D, runtime=runtime, dt=dt)

    assert pset.sample_var_rectilinear == runtime/dt
    assert pset.sample_var_curvilinear == runtime/dt


@pytest.mark.parametrize('mode', ['jit'])
def test_diff_entry_chunksize_error_nemo_simple(mode):
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
    chs = {'U': {'depth': ('depthu', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
           'V': {'depth': ('depthv', 20), 'lat': ('y', 4), 'lon': ('x', 16)},
           'W': {'depth': ('depthw', 16), 'lat': ('y', 16), 'lon': ('x', 4)}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=chs)
    npart = 20
    lonp = 5.2 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(fieldset, mode, lonp, latp)


@pytest.mark.parametrize('mode', ['jit'])
def test_diff_entry_chunksize_error_nemo_complex_conform_depth(mode):
    # ==== this test is expected to fall-back to a pre-defined minimal chunk as ==== #
    # ==== the requested chunks don't match, or throw a value error.            ==== #
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
    chs = {'U': {'depth': ('depthu', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
           'V': {'depth': ('depthv', 75), 'lat': ('y', 4), 'lon': ('x', 16)},
           'W': {'depth': ('depthw', 75), 'lat': ('y', 16), 'lon': ('x', 4)}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=chs)
    npart = 20
    lonp = 5.2 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(fieldset, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    npart_U = 1
    npart_U = [npart_U * k for k in fieldset.U.nchunks[1:]]
    npart_V = 1
    npart_V = [npart_V * k for k in fieldset.V.nchunks[1:]]
    npart_W = 1
    npart_W = [npart_W * k for k in fieldset.V.nchunks[1:]]
    chn = {'U': {'lat': int(math.ceil(201.0/chs['U']['lat'][1])),
                 'lon': int(math.ceil(151.0/chs['U']['lon'][1])),
                 'depth': int(math.ceil(75.0/chs['U']['depth'][1]))},
           'V': {'lat': int(math.ceil(201.0/chs['V']['lat'][1])),
                 'lon': int(math.ceil(151.0/chs['V']['lon'][1])),
                 'depth': int(math.ceil(75.0/chs['V']['depth'][1]))},
           'W': {'lat': int(math.ceil(201.0/chs['W']['lat'][1])),
                 'lon': int(math.ceil(151.0/chs['W']['lon'][1])),
                 'depth': int(math.ceil(75.0/chs['W']['depth'][1]))}}
    npart_U_request = 1
    npart_U_request = [npart_U_request * chn['U'][k] for k in chn['U']]
    npart_V_request = 1
    npart_V_request = [npart_V_request * chn['V'][k] for k in chn['V']]
    npart_W_request = 1
    npart_W_request = [npart_W_request * chn['W'][k] for k in chn['W']]
    assert (npart_U != npart_U_request)
    assert (npart_V != npart_V_request)
    assert (npart_W != npart_W_request)


@pytest.mark.parametrize('mode', ['jit'])
def test_diff_entry_chunksize_error_nemo_complex_nonconform_depth(mode):
    # ==== this test is expected to fall-back to a pre-defined minimal chunk as the ==== #
    # ==== requested chunks don't match, or throw a value error                     ==== #
    data_path = path.join(path.dirname(__file__), 'NemoNorthSeaORCA025-N006_data/')
    ufiles = sorted(glob(data_path + 'ORCA*U.nc'))
    vfiles = sorted(glob(data_path + 'ORCA*V.nc'))
    wfiles = sorted(glob(data_path + 'ORCA*W.nc'))
    mesh_mask = data_path + 'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles}}
    variables = {'U': 'uo',
                 'V': 'vo'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}
    chs = {'U': {'depth': ('depthu', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
           'V': {'depth': ('depthv', 15), 'lat': ('y', 4), 'lon': ('x', 16)}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=chs)
    npart = 20
    lonp = 5.2 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(fieldset, mode, lonp, latp)


@pytest.mark.parametrize('mode', ['jit'])
def test_erroneous_fieldset_init(mode):
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
    chs = {'U': {'depth': ('depthu', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
           'V': {'depth': ('depthv', 75), 'lat': ('y', 16), 'lon': ('x', 16)},
           'W': {'depth': ('depthw', 75), 'lat': ('y', 16), 'lon': ('x', 16)}}

    try:
        FieldSet.from_nemo(filenames, variables, dimensions, chunksize=chs)
    except ValueError:
        return True
    return False


@pytest.mark.parametrize('mode', ['jit'])
def test_diff_entry_chunksize_correction_globcurrent(mode):
    filenames = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                          '200201*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    chs = {'U': {'lat': ('lat', 16), 'lon': ('lon', 16)},
           'V': {'lat': ('lat', 16), 'lon': ('lon', 4)}}
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, chunksize=chs)
    lonp = [25]
    latp = [-35]
    compute_globcurrent_particle_advection(fieldset, mode, lonp, latp)
    # GlobCurrent sample file dimensions: time=UNLIMITED, lat=41, lon=81
    npart_U = 1
    npart_U = [npart_U * k for k in fieldset.U.nchunks[1:]]
    npart_V = 1
    npart_V = [npart_V * k for k in fieldset.V.nchunks[1:]]
    npart_V_request = 1
    chn = {'U': {'lat': int(math.ceil(41.0/chs['U']['lat'][1])),
                 'lon': int(math.ceil(81.0/chs['U']['lon'][1]))},
           'V': {'lat': int(math.ceil(41.0/chs['V']['lat'][1])),
                 'lon': int(math.ceil(81.0/chs['V']['lon'][1]))}}
    npart_V_request = [npart_V_request * chn['V'][k] for k in chn['V']]
    assert (npart_V_request != npart_U)
    assert (npart_V_request == npart_V)
