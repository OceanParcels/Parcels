from parcels import (FieldSet, ScipyParticle, JITParticle, Variable, ErrorCode)
from parcels.particlefile import _set_calendar
from parcels.tools.converters import _get_cftime_calendars, _get_cftime_datetimes
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
import numpy as np
import pytest
import os
import cftime
import xarray as xr

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


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_remove_particles(fieldset, pset_mode, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_array_remove_particles.zarr")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(filepath)
    pfile.write(pset, 0)
    pset.remove_indices(3)
    for p in pset:
        p.time = 1
    pfile.write(pset, 1)

    ds = xr.open_zarr(filepath)
    timearr = ds['time'][:]
    assert (np.isnat(timearr[3, 1])) and (np.isfinite(timearr[3, 0]))
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_set_towrite_False(fieldset, pset_mode, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_set_towrite_False.zarr")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart))
    pset.set_variable_write_status('depth', False)
    pset.set_variable_write_status('lat', False)
    pfile = pset.ParticleFile(filepath, outputdt=1)

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset.execute(Update_lon, runtime=10, output_file=pfile)

    ds = xr.open_zarr(filepath)
    assert 'time' in ds
    assert 'depth' not in ds
    assert 'lat' not in ds
    ds.close()

    # For pytest purposes, we need to reset to original status
    pset.set_variable_write_status('depth', True)
    pset.set_variable_write_status('lat', True)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('chunks_obs', [1, None])
def test_pfile_array_remove_all_particles(fieldset, pset_mode, mode, chunks_obs, tmpdir, npart=10):

    filepath = tmpdir.join("pfile_array_remove_particles.zarr")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart), time=0)
    chunks = (npart, chunks_obs) if chunks_obs else None
    pfile = pset.ParticleFile(filepath, chunks=chunks)
    pfile.write(pset, 0)
    for _ in range(npart):
        pset.remove_indices(-1)
    pfile.write(pset, 1)
    pfile.write(pset, 2)

    ds = xr.open_zarr(filepath)
    assert np.allclose(ds['time'][:, 0], np.timedelta64(0, 's'), atol=np.timedelta64(1, 'ms'))
    if chunks_obs is not None:
        assert ds['time'][:].shape == chunks
    else:
        assert ds['time'][:].shape[0] == npart
        assert np.all(np.isnan(ds['time'][:, 1:]))
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_written_ondelete(fieldset, pset_mode, mode, tmpdir, npart=3):
    filepath = tmpdir.join("pfile_on_delete_written_variables.zarr")

    def move_west(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon]  # to trigger out-of-bounds error
        particle.lon -= 0.1 + tmp1

    def DeleteP(particle, fieldset, time):
        particle.delete()

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(0.95, 0.05, npart)

    (dt, runtime) = (0.1, 0.8)
    lon_end = lon - runtime/dt*0.1
    noutside = len(lon_end[lon_end < 0])

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=lon, lat=lat)

    outfile = pset.ParticleFile(name=filepath, write_ondelete=True, chunks=(len(pset), 1))
    outfile.add_metadata('runtime', runtime)
    pset.execute(move_west, runtime=runtime, dt=dt, output_file=outfile,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteP})

    ds = xr.open_zarr(filepath)
    assert ds.runtime == runtime
    lon = ds['lon'][:]
    assert (sum(np.isfinite(lon)) == noutside)
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_write_double(fieldset, pset_mode, mode, tmpdir):
    filepath = tmpdir.join("pfile_variable_write_double.zarr")

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.00001)
    pset.execute(pset.Kernel(Update_lon), endtime=0.001, dt=0.00001, output_file=ofile)

    ds = xr.open_zarr(filepath)
    lons = ds['lon'][:]
    assert (isinstance(lons.values[0, 0], np.float64))
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_dtypes_pfile(fieldset, pset_mode, mode, tmpdir):
    filepath = tmpdir.join("pfile_dtypes.zarr")

    dtypes = ['float32', 'float64', 'int32', 'uint32', 'int64', 'uint64']
    if mode == 'scipy':
        dtypes.extend(['bool_', 'int8', 'uint8', 'int16', 'uint16'])  # Not implemented in AoS JIT

    class MyParticle(ptype[mode]):
        for d in dtypes:
            # need an exec() here because we need to dynamically set the variable name
            exec(f'v_{d} = Variable("v_{d}", dtype=np.{d}, initial=0.)')

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=MyParticle, lon=0, lat=0, time=0)
    pfile = pset.ParticleFile(name=filepath, outputdt=1)
    pfile.write(pset, 0)

    ds = xr.open_zarr(filepath, mask_and_scale=False)  # Note masking issue at https://stackoverflow.com/questions/68460507/xarray-loading-int-data-as-float
    for d in dtypes:
        assert ds[f'v_{d}'].dtype == d


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('npart', [1, 2, 5])
def test_variable_written_once(fieldset, pset_mode, mode, tmpdir, npart):
    filepath = tmpdir.join("pfile_once_written_variables.zarr")

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
    ds = xr.open_zarr(filepath)
    vfile = np.ma.filled(ds['v_once'][:], np.nan)
    assert (vfile.shape == (npart, ))
    assert np.allclose(vfile, time)
    ds.close()


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
    outfilepath = tmpdir.join("pfile_repeated_release.zarr")
    pfile = pset.ParticleFile(outfilepath, outputdt=abs(dt), chunks=(1, 1))

    def IncrLon(particle, fieldset, time):
        particle.sample_var += 1.
        if particle.sample_var > fieldset.maxvar:
            particle.delete()
    for i in range(runtime):
        pset.execute(IncrLon, dt=dt, runtime=1., output_file=pfile)

    ds = xr.open_zarr(outfilepath)
    samplevar = ds['sample_var'][:]
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
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_timebackward(fieldset, pset_mode, mode, tmpdir):
    outfilepath = tmpdir.join("pfile_write_timebackward.zarr")

    def Update_lon(particle, fieldset, time):
        particle.lon -= 0.1 * particle.dt

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lat=np.linspace(0, 1, 3), lon=[0, 0, 0], time=[1, 2, 3])
    pfile = pset.ParticleFile(name=outfilepath, outputdt=1.)
    pset.execute(pset.Kernel(Update_lon), runtime=4, dt=-1.,
                 output_file=pfile)
    ds = xr.open_zarr(outfilepath)
    trajs = ds['trajectory'][:]
    assert trajs.values.dtype == 'int64'
    assert np.all(np.diff(trajs.values) < 0)  # all particles written in order of release
    ds.close()


def test_set_calendar():
    for calendar_name, cf_datetime in zip(_get_cftime_calendars(), _get_cftime_datetimes()):
        date = getattr(cftime, cf_datetime)(1990, 1, 1)
        assert _set_calendar(date.calendar) == date.calendar
    assert _set_calendar('np_datetime64') == 'standard'


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_reset_dt(fieldset, pset_mode, mode, tmpdir):
    # Assert that p.dt gets reset when a write_time is not a multiple of dt
    # for p.dt=0.02 to reach outputdt=0.05 and endtime=0.1, the steps should be [0.2, 0.2, 0.1, 0.2, 0.2, 0.1], resulting in 6 kernel executions
    filepath = tmpdir.join("pfile_reset_dt.zarr")

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.05)
    pset.execute(pset.Kernel(Update_lon), endtime=0.1, dt=0.02, output_file=ofile)

    assert np.allclose(pset.lon, .6)
