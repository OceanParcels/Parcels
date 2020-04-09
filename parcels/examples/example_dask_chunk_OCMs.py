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
from parcels import VectorField, NestedField, SummedField
from parcels.tools.loggers import logger

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
        chs = {'U': {'depthu': 75, 'depthv': 75, 'depthw': 75, 'y': 16, 'x': 16},
               'V': {'depthu': 75, 'depthv': 75, 'depthw': 75, 'y': 16, 'x': 16},
               'W': {'depthu': 75, 'depthv': 75, 'depthw': 75, 'y': 16, 'x': 16}}

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
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
        chs = {'U': {'lat': 16, 'lon': 16},
               'V': {'lat': 16, 'lon': 16}}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, field_chunksize=chs)
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
        chs = {'i': 8, 'j': 8, 'w_dep': 3}

    fieldset = FieldSet.from_pop(filenames, variables, dimensions, field_chunksize=chs, timestamps=timestamps)
    return fieldset


def fieldset_from_swash(chunk_mode):
    filenames = path.join(path.join(path.dirname(__file__), 'SWASH_data'), 'field_*.nc')
    variables = {'U': 'cross-shore velocity',
                 'V': 'along-shore velocity',
                 'W': 'vertical velocity',
                 'depth': 'time varying depth',
                 'depth_u': 'time varying depth_u'}
    dimensions = {'U': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'V': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'W': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'depth': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'depth_u': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'}}
    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = (1, 7, 4, 4)
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='flat', allow_time_extrapolation=True, field_chunksize=chs)
    fieldset.U.set_depth_from_field(fieldset.depth_u)
    fieldset.V.set_depth_from_field(fieldset.depth_u)
    fieldset.W.set_depth_from_field(fieldset.depth)
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


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific'])
def test_nemo_3D(mode, chunk_mode):
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '2MiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_nemo_3D(chunk_mode)
    npart = 20
    lonp = 2.5 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(field_set, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(201.0/16.0)) * int(math.ceil(151.0/16.0))))


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific'])
def test_pop(mode, chunk_mode):
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '1MiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_pop_1arcs(chunk_mode)
    npart = 20
    lonp = 70.0 * np.ones(npart)
    latp = [i for i in -45.0+(-0.25+np.random.rand(npart)*2.0*0.25)]
    compute_pop_particle_advection(field_set, mode, lonp, latp)
    # POP sample file dimensions: k=20, j=60, i=60
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (int(math.ceil(21.0/3.0)) * int(math.ceil(60.0/8.0)) * int(math.ceil(60.0/8.0))))


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific'])
def test_swash(mode, chunk_mode):
    if chunk_mode == 'auto':
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
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode != 'auto':
        assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(6.0 / 7.0)) * int(math.ceil(21.0 / 4.0)) * int(math.ceil(51.0 / 4.0))))
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(7.0 / 7.0)) * int(math.ceil(21.0 / 4.0)) * int(math.ceil(51.0 / 4.0))))


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('chunk_mode', [False, 'auto', 'specific'])
def test_globcurrent_2D(mode, chunk_mode):
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '32KiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_globcurrent(chunk_mode)
    lonp = [25]
    latp = [-35]
    pset = compute_globcurrent_particle_advection(field_set, mode, lonp, latp)
    # GlobCurrent sample file dimensions: time=UNLIMITED, lat=41, lon=81
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(41.0/16.0)) * int(math.ceil(81.0/16.0))))
    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)


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
    chs = {'U': {'depthu': 75, 'depthv': 75, 'y': 16, 'x': 16},
           'V': {'depthu': 75, 'depthv': 75, 'y': 16, 'x': 16}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
    npart = 20
    lonp = 5.2 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(fieldset, mode, lonp, latp)
    # Nemo sample file dimensions: depthu=75, y=201, x=151
    assert (len(fieldset.U.grid.load_chunk) == len(fieldset.V.grid.load_chunk))


# ==== TO BE EXTERNALIZED OR CHECKED WHEN #782 IS FIXED ==== #
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_3d_2dfield_sampling(mode):
    logger.warning("Test is to be re-enabled after #782 is fixed to test tertiary effects.")
    return True
    data_path = path.join(path.dirname(__file__), 'NemoNorthSeaORCA025-N006_data/')
    ufiles = sorted(glob(data_path + 'ORCA*U.nc'))
    vfiles = sorted(glob(data_path + 'ORCA*V.nc'))
    mesh_mask = data_path + 'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'data': vfiles},
                 # 'nav_lon': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ufiles[0]}}
                 'nav_lon': {'lon': mesh_mask, 'lat': mesh_mask, 'data': [ufiles[0], ]}}
    variables = {'U': 'uo',
                 'V': 'vo',
                 'nav_lon': 'nav_lon'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'nav_lon': {'lon': 'glamf', 'lat': 'gphif'}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=False)
    fieldset.nav_lon.data = np.ones(fieldset.nav_lon.data.shape, dtype=np.float32)
    fieldset.add_field(Field('rectilinear_2D', np.ones((2, 2)),
                             lon=np.array([-10, 20]), lat=np.array([40, 80]), field_chunksize=False))

    class MyParticle(ptype[mode]):
        sample_var_curvilinear = Variable('sample_var_curvilinear')
        sample_var_rectilinear = Variable('sample_var_rectilinear')
    pset = ParticleSet(fieldset, pclass=MyParticle, lon=2.5, lat=52)

    def Sample2D(particle, fieldset, time):
        particle.sample_var_curvilinear += fieldset.nav_lon[time, particle.depth, particle.lat, particle.lon]
        particle.sample_var_rectilinear += fieldset.rectilinear_2D[time, particle.depth, particle.lat, particle.lon]

    runtime, dt = 86400*4, 6*3600
    pset.execute(pset.Kernel(AdvectionRK4) + Sample2D, runtime=runtime, dt=dt)
    print(pset.xi)

    for f in fieldset.get_fields():
        if type(f) in [VectorField, NestedField, SummedField]:  # or not f.grid.defer_load:
            continue
        g = f.grid
        npart = 1
        npart = [npart * k for k in f.nchunks[1:]]
        print("Field '{}': grid type: {}; grid chunksize: {}; grid mesh: {}; field N partitions: {}; field nchunks: {}; grid chunk_info: {}; grid load_chunk: {}; grid layout: {}".format(f.name, g.gtype, g.master_chunksize, g.mesh, npart, f.nchunks, g.chunk_info, g.load_chunk, (g.tdim, g.zdim, g.ydim, g.xdim)))
    for i in range(0, len(fieldset.gridset.grids)):
        g = fieldset.gridset.grids[i]
        print(
            "Grid {}: grid type: {}; grid chunksize: {}; grid mesh: {}; grid chunk_info: {}; grid load_chunk: {}; grid layout: {}".format(
                i, g.gtype, g.master_chunksize, g.mesh, g.chunk_info, g.load_chunk, (g.tdim, g.zdim, g.ydim, g.xdim)))

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
    chs = {'U': {'depthu': 75, 'y': 16, 'x': 16},
           'V': {'depthv': 20, 'y': 4, 'x': 16},
           'W': {'depthw': 15, 'y': 16, 'x': 4}}
    try:
        fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
    except ValueError:
        return True
    npart = 20
    lonp = 5.2 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    compute_nemo_particle_advection(fieldset, mode, lonp, latp)
    return False


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
    chs = {'U': {'depthu': 75, 'depthv': 75, 'depthw': 75, 'y': 16, 'x': 16},
           'V': {'depthu': 75, 'depthv': 75, 'depthw': 75, 'y': 4, 'x': 16},
           'W': {'depthu': 75, 'depthv': 75, 'depthw': 75, 'y': 16, 'x': 4}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
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
    chn = {'U': {'lat': int(math.ceil(201.0/chs['U']['y'])),
                 'lon': int(math.ceil(151.0/chs['U']['x'])),
                 'depth': int(math.ceil(75.0/chs['U']['depthu']))},
           'V': {'lat': int(math.ceil(201.0/chs['V']['y'])),
                 'lon': int(math.ceil(151.0/chs['V']['x'])),
                 'depth': int(math.ceil(75.0/chs['V']['depthv']))},
           'W': {'lat': int(math.ceil(201.0/chs['W']['y'])),
                 'lon': int(math.ceil(151.0/chs['W']['x'])),
                 'depth': int(math.ceil(75.0/chs['W']['depthw']))}}
    npart_U_request = 1
    npart_U_request = [npart_U_request * chn['U'][k] for k in chn['U']]
    npart_V_request = 1
    npart_V_request = [npart_V_request * chn['V'][k] for k in chn['V']]
    npart_W_request = 1
    npart_W_request = [npart_W_request * chn['W'][k] for k in chn['W']]
    assert (len(fieldset.U.grid.load_chunk) == len(fieldset.V.grid.load_chunk))
    assert (len(fieldset.U.grid.load_chunk) == len(fieldset.W.grid.load_chunk))
    assert (npart_U == npart_V)
    assert (npart_U == npart_W)
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
    chs = {'U': {'depthu': 75, 'depthv': 15, 'y': 16, 'x': 16},
           'V': {'depthu': 75, 'depthv': 15, 'y': 4, 'x': 16}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
    npart = 20
    lonp = 5.2 * np.ones(npart)
    latp = [i for i in 52.0+(-1e-3+np.random.rand(npart)*2.0*1e-3)]
    try:
        compute_nemo_particle_advection(fieldset, mode, lonp, latp)
    except IndexError:  # incorrect data access, in case grids were created
        return True
    except AssertionError:  # U-V grids are not equal to one another, throwing assertion errors
        return True
    return False


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
    chs = {'U': {'depthu': 75, 'y': 16, 'x': 16},
           'V': {'depthv': 75, 'y': 16, 'x': 16},
           'W': {'depthw': 75, 'y': 16, 'x': 16}}

    try:
        FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs)
    except ValueError:
        return True
    return False


@pytest.mark.parametrize('mode', ['jit'])
def test_diff_entry_chunksize_correction_globcurrent(mode):
    filenames = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                          '200201*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    chs = {'U': {'lat': 16, 'lon': 16},
           'V': {'lat': 16, 'lon': 4}}
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, field_chunksize=chs)
    lonp = [25]
    latp = [-35]
    compute_globcurrent_particle_advection(fieldset, mode, lonp, latp)
    # GlobCurrent sample file dimensions: time=UNLIMITED, lat=41, lon=81
    npart_U = 1
    npart_U = [npart_U * k for k in fieldset.U.nchunks[1:]]
    npart_V = 1
    npart_V = [npart_V * k for k in fieldset.V.nchunks[1:]]
    npart_V_request = 1
    chn = {'U': {'lat': int(math.ceil(41.0/chs['U']['lat'])),
                 'lon': int(math.ceil(81.0/chs['U']['lon']))},
           'V': {'lat': int(math.ceil(41.0/chs['V']['lat'])),
                 'lon': int(math.ceil(81.0/chs['V']['lon']))}}
    npart_V_request = [npart_V_request * chn['V'][k] for k in chn['V']]
    assert (npart_U == npart_V)
    assert (npart_V != npart_V_request)
    assert (len(fieldset.U.grid.load_chunk) == len(fieldset.V.grid.load_chunk))
