from parcels import (FieldSet, ScipyParticle, JITParticle, Variable, ErrorCode)
from parcels.particlefile import _set_calendar
from parcels.tools.converters import _get_cftime_calendars, _get_cftime_datetimes
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
import numpy as np
import pytest
import os
from netCDF4 import Dataset
import cftime
import random as py_random

pset_modes = ['soa', 'aos']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


def fieldset(xdim=40, ydim=100):
    U = np.zeros((ydim, xdim), dtype=np.float32)
    V = np.zeros((ydim, xdim), dtype=np.float32)
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(-60, 60, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth}
    return FieldSet.from_data(data, dimensions)


@pytest.fixture(name="fieldset")
def fieldset_ficture(xdim=40, ydim=100):
    return fieldset(xdim=xdim, ydim=ydim)


def close_and_compare_netcdffiles(filepath, ofile, assystemcall=False):
    if assystemcall:
        os.system('parcels_convert_npydir_to_netcdf %s' % ofile.tempwritedir_base)
    else:
        import parcels.scripts.convert_npydir_to_netcdf as convert
        convert.convert_npydir_to_netcdf(ofile.tempwritedir_base, pfile_class=ofile.__class__)

    ncfile1 = Dataset(filepath, 'r', 'NETCDF4')

    ofile.name = filepath + 'b.nc'
    ofile.export()
    ncfile2 = Dataset(filepath + 'b.nc', 'r', 'NETCDF4')

    for v in ncfile2.variables.keys():
        assert np.allclose(ncfile1.variables[v][:], ncfile2.variables[v][:])

    for a in ncfile2.ncattrs():
        if a != 'parcels_version':
            assert getattr(ncfile1, a) == getattr(ncfile2, a)

    ncfile2.close()
    return ncfile1


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_remove_particles(fieldset, pset_mode, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_array_remove_particles.nc")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(filepath)
    pfile.write(pset, 0)
    pset.remove_indices(3)
    for p in pset:
        p.time = 1
    pfile.write(pset, 1)
    ncfile = close_and_compare_netcdffiles(filepath, pfile)
    timearr = ncfile.variables['time'][:]
    assert type(timearr[3, 1]) is not type(timearr[3, 0])  # noqa
    ncfile.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_set_towrite_False(fieldset, pset_mode, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_set_towrite_False.nc")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart))
    pset.set_variable_write_status('depth', False)
    pset.set_variable_write_status('lat', False)
    pfile = pset.ParticleFile(filepath, outputdt=1)

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset.execute(Update_lon, runtime=10, output_file=pfile)
    ncfile = close_and_compare_netcdffiles(filepath, pfile)
    assert 'time' in ncfile.variables
    assert 'depth' not in ncfile.variables
    assert 'lat' not in ncfile.variables
    ncfile.close()

    # For pytest purposes, we need to reset to original status
    pset.set_variable_write_status('depth', True)
    pset.set_variable_write_status('lat', True)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_remove_all_particles(fieldset, pset_mode, mode, tmpdir, npart=10):

    filepath = tmpdir.join("pfile_array_remove_particles.nc")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(filepath)
    pfile.write(pset, 0)
    for _ in range(npart):
        pset.remove_indices(-1)
    pfile.write(pset, 1)
    pfile.write(pset, 2)
    ncfile = close_and_compare_netcdffiles(filepath, pfile)
    assert ncfile.variables['time'][:].shape == (npart, 1)
    ncfile.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('assystemcall', [True, False])
def test_variable_written_ondelete(fieldset, pset_mode, mode, tmpdir, assystemcall, npart=3):
    filepath = tmpdir.join("pfile_on_delete_written_variables.nc")

    def move_west(particle, fieldset, time):
        tmp = fieldset.U[time, particle.depth, particle.lat, particle.lon]  # to trigger out-of-bounds error
        particle.lon -= 0.1 + tmp

    def DeleteP(particle, fieldset, time):
        particle.delete()

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(0.95, 0.05, npart)

    (dt, runtime) = (0.1, 0.8)
    lon_end = lon - runtime/dt*0.1
    noutside = len(lon_end[lon_end < 0])

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=lon, lat=lat)

    outfile = pset.ParticleFile(name=filepath, write_ondelete=True)
    outfile.add_metadata('runtime', runtime)
    pset.execute(move_west, runtime=runtime, dt=dt, output_file=outfile,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteP})

    ncfile = close_and_compare_netcdffiles(filepath, outfile, assystemcall=assystemcall)
    assert ncfile.runtime == runtime
    lon = ncfile.variables['lon'][:]
    assert (lon.size == noutside)
    ncfile.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_write_double(fieldset, pset_mode, mode, tmpdir):
    filepath = tmpdir.join("pfile_variable_write_double.nc")

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.00001)
    pset.execute(pset.Kernel(Update_lon), endtime=0.001, dt=0.00001, output_file=ofile)

    ncfile = close_and_compare_netcdffiles(filepath, ofile)
    lons = ncfile.variables['lon'][:]
    assert (isinstance(lons[0, 0], np.float64))
    ncfile.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_dtypes_pfile(fieldset, mode, pset_mode, tmpdir):
    filepath = tmpdir.join("pfile_dtypes.nc")

    dtypes = ['float32', 'float64', 'int32', 'int64']
    if mode == 'scipy':
        dtypes.append('bool_')  # bool only implemented in scipy

    class MyParticle(ptype[mode]):
        for d in dtypes:
            # need an exec() here because we need to dynamically set the variable name
            exec(f'v_{d} = Variable("v_{d}", dtype=np.{d}, initial=0.)')

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=MyParticle, lon=0, lat=0)
    pfile = pset.ParticleFile(name=filepath, outputdt=1)
    pfile.write(pset, 0)
    ncfile = close_and_compare_netcdffiles(filepath, pfile)
    for d in dtypes:
        nc_fmt = d if d != 'bool_' else 'i1'
        assert ncfile.variables[f'v_{d}'].dtype == nc_fmt


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('npart', [1, 2, 5])
def test_variable_written_once(fieldset, pset_mode, mode, tmpdir, npart):
    filepath = tmpdir.join("pfile_once_written_variables.nc")

    def Update_v(particle, fieldset, time):
        particle.v_once += 1.
        particle.age += particle.dt

    class MyParticle(ptype[mode]):
        v_once = Variable('v_once', dtype=np.float64, initial=0., to_write='once')
        age = Variable('age', dtype=np.float32, initial=0.)
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    time = np.arange(0, npart/10., 0.1, dtype=np.float64)
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=MyParticle, lon=lon, lat=lat, time=time, v_once=time)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.1)
    pset.execute(pset.Kernel(Update_v), endtime=1, dt=0.1, output_file=ofile)

    assert np.allclose(pset.v_once - time - pset.age*10, 0, atol=1e-5)
    ncfile = close_and_compare_netcdffiles(filepath, ofile)
    vfile = np.ma.filled(ncfile.variables['v_once'][:], np.nan)
    assert (vfile.shape == (npart, ))
    assert np.allclose(vfile, time)
    ncfile.close()


@pytest.mark.parametrize('type', ['repeatdt', 'timearr'])
@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('repeatdt', range(1, 3))
@pytest.mark.parametrize('dt', [-1, 1])
@pytest.mark.parametrize('maxvar', [2, 4, 10])
def test_pset_repeated_release_delayed_adding_deleting(type, fieldset, pset_mode, mode, repeatdt, tmpdir, dt, maxvar, runtime=10):
    fieldset.maxvar = maxvar
    pset = None

    class MyParticle(ptype[mode]):
        sample_var = Variable('sample_var', initial=0.)
        v_once = Variable('v_once', dtype=np.float64, initial=0., to_write='once')

    if type == 'repeatdt':
        pset = pset_type[pset_mode]['pset'](fieldset, lon=[0], lat=[0], pclass=MyParticle, repeatdt=repeatdt)
    elif type == 'timearr':
        pset = pset_type[pset_mode]['pset'](fieldset, lon=np.zeros(runtime), lat=np.zeros(runtime), pclass=MyParticle, time=list(range(runtime)))
    outfilepath = tmpdir.join("pfile_repeated_release.nc")
    pfile = pset.ParticleFile(outfilepath, outputdt=abs(dt))

    def IncrLon(particle, fieldset, time):
        particle.sample_var += 1.
        if particle.sample_var > fieldset.maxvar:
            particle.delete()
    for i in range(runtime):
        pset.execute(IncrLon, dt=dt, runtime=1., output_file=pfile)

    ncfile = close_and_compare_netcdffiles(outfilepath, pfile)
    samplevar = ncfile.variables['sample_var'][:]
    if type == 'repeatdt':
        assert samplevar.shape == (runtime // repeatdt+1, min(maxvar+1, runtime)+1)
        assert np.allclose(pset.sample_var, np.arange(maxvar, -1, -repeatdt))
    elif type == 'timearr':
        assert samplevar.shape == (runtime, min(maxvar + 1, runtime) + 1)
    # test whether samplevar[:, k] = k
    for k in range(samplevar.shape[1]):
        assert np.allclose([p for p in samplevar[:, k] if np.isfinite(p)], k)
    filesize = os.path.getsize(str(outfilepath))
    assert filesize < 1024 * 65  # test that chunking leads to filesize less than 65KB
    ncfile.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_timebackward(fieldset, pset_mode, mode, tmpdir):
    outfilepath = tmpdir.join("pfile_write_timebackward.nc")

    def Update_lon(particle, fieldset, time):
        particle.lon -= 0.1 * particle.dt

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lat=np.linspace(0, 1, 3), lon=[0, 0, 0], time=[1, 2, 3])
    pfile = pset.ParticleFile(name=outfilepath, outputdt=1.)
    pset.execute(pset.Kernel(Update_lon), runtime=4, dt=-1.,
                 output_file=pfile)
    ncfile = close_and_compare_netcdffiles(outfilepath, pfile)
    trajs = ncfile.variables['trajectory'][:, 0]
    assert np.all(np.diff(trajs) > 0)  # all particles written in order of traj ID


def test_set_calendar():
    for calendar_name, cf_datetime in zip(_get_cftime_calendars(), _get_cftime_datetimes()):
        date = getattr(cftime, cf_datetime)(1990, 1, 1)
        assert _set_calendar(date.calendar) == date.calendar
    assert _set_calendar('np_datetime64') == 'standard'


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_error_duplicate_outputdir(fieldset, tmpdir, pset_mode):
    outfilepath = tmpdir.join("error_duplicate_outputdir.nc")
    pset1 = pset_type[pset_mode]['pset'](fieldset, pclass=JITParticle, lat=0, lon=0)
    pset2 = pset_type[pset_mode]['pset'](fieldset, pclass=JITParticle, lat=0, lon=0)

    py_random.seed(1234)
    pfile1 = pset1.ParticleFile(name=outfilepath, outputdt=1., convert_at_end=False)

    py_random.seed(1234)
    error_thrown = False
    try:
        pset2.ParticleFile(name=outfilepath, outputdt=1., convert_at_end=False)
    except IOError:
        error_thrown = True
    assert error_thrown

    pfile1.delete_tempwritedir()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_reset_dt(fieldset, pset_mode, mode, tmpdir):
    # Assert that p.dt gets reset when a write_time is not a multiple of dt
    # for p.dt=0.02 to reach outputdt=0.05 and endtime=0.1, the steps should be [0.2, 0.2, 0.1, 0.2, 0.2, 0.1], resulting in 6 kernel executions
    filepath = tmpdir.join("pfile_reset_dt.nc")

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.05)
    pset.execute(pset.Kernel(Update_lon), endtime=0.1, dt=0.02, output_file=ofile)

    assert np.allclose(pset.lon, .6)
