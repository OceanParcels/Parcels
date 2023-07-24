import contextlib
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from parcels import (
    FieldSet,
    JITParticle,
    KernelAOS,
    KernelSOA,
    ParticleFileAOS,
    ParticleFileSOA,
    ParticleSetAOS,
    ParticleSetSOA,
    ScipyParticle,
    Variable,
)

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


def DoNothing(particle, fieldset, time):
    pass


def convert_sqlite_file(fname):
    with contextlib.closing(sqlite3.connect(fname)) as con:
        df = pd.read_sql_query("SELECT * from particles", con, index_col=['trajectory', 'time'])
        metadata = pd.read_sql_query("SELECT * from metadata", con).to_dict('records')[0]
    ds = xr.Dataset.from_dataframe(df)
    ds.attrs['metadata'] = metadata
    if 'timedelta64' in ds.metadata['calendar']:
        ds['time'] = ds['time'].astype(ds.metadata['calendar'])
    else:
        ds['time'] = pd.to_datetime(ds['time'], origin=pd.Timestamp(ds.metadata['time_origin']))
    return ds


@pytest.mark.skip("Parquet store writing not yet implemented")  # TODO fix test for writing to parquet store (if that even exists)
@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_write_parquet_memorystore(fieldset, pset_mode, mode, npart=10):
    """Check that writing to a parquet MemoryStore works."""
    parquet_store = None  # MemoryStore()
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(parquet_store)
    pfile.write(pset, 0)

    ds = xr.Dataset.from_dataframe(pd.read_parquet(parquet_store))
    assert ds.dims["trajectory"] == npart
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_remove_particles(fieldset, pset_mode, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_array_remove_particles.sqlite")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(filepath, outputdt=1)
    pset.execute(DoNothing, runtime=0, dt=0, output_file=pfile)
    pset.remove_indices(3)
    for p in pset:
        p.time = 1
    pset.execute(DoNothing, runtime=0, dt=0, output_file=pfile)

    ds = convert_sqlite_file(filepath)
    latarr = ds['lat'][:]
    assert (np.isnan(latarr[3, 1])) and (np.isfinite(latarr[3, 0]))
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_set_towrite_False(fieldset, pset_mode, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_set_towrite_False.sqlite")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart))
    pset.set_variable_write_status('depth', False)
    pset.set_variable_write_status('lat', False)
    pfile = pset.ParticleFile(filepath, outputdt=1)

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset.execute(Update_lon, runtime=10, output_file=pfile)

    ds = convert_sqlite_file(filepath)
    assert 'time' in ds
    assert 'depth' not in ds
    assert 'lat' not in ds
    ds.close()

    # For pytest purposes, we need to reset to original status  # TODO: is this really necessary?
    pset.set_variable_write_status('depth', True)
    pset.set_variable_write_status('lat', True)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_remove_all_particles(fieldset, pset_mode, mode, tmpdir, npart=10):

    filepath = tmpdir.join("pfile_array_remove_particles.sqlite")
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(filepath, outputdt=1)

    pset.execute(DoNothing, runtime=1, dt=1, output_file=pfile)
    for _ in range(npart):
        pset.remove_indices(-1)

    ds = convert_sqlite_file(filepath)
    assert np.allclose(ds['time'][0], np.timedelta64(0, 's'), atol=np.timedelta64(1, 'ms'))
    assert ds['lat'][:].shape[0] == npart
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_metadata(fieldset, pset_mode, mode, tmpdir):
    filepath = tmpdir.join("pfile_metadata.sqlite")

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=0, lat=0)

    runtime = 2
    outfile = pset.ParticleFile(name=filepath, outputdt=1)
    outfile.add_metadata('runtime', runtime)
    pset.execute(DoNothing, runtime=runtime, dt=1, output_file=outfile)

    ds = convert_sqlite_file(filepath)
    assert np.isclose(ds.metadata['runtime'], runtime)
    assert np.isclose(ds.metadata['time_origin'], fieldset.time_origin.time_origin)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_calendar(pset_mode, mode, tmpdir):

    time = np.datetime64('2000-01-01')
    fieldset = FieldSet.from_data({'U': 0., 'V': 0.}, {'lat': 0., 'lon': 0., 'time': [time]})
    filepath = tmpdir.join("pfile_calendar.sqlite")

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=0, lat=0)
    outfile = pset.ParticleFile(name=filepath, outputdt=1)
    pset.execute(DoNothing, runtime=1, dt=1, output_file=outfile)
    ds = convert_sqlite_file(filepath)

    assert ds['time'][0] == time


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['jit', 'scipy'])  # TODO pytest.param('scipy', marks=pytest.mark.xfail(reason="pandas throws a mysterious error: pyarrow.lib.ArrowInvalid: Float value XXX was truncated converting to int64"))])
def test_variable_write_double(fieldset, pset_mode, mode, tmpdir):
    filepath = tmpdir.join("pfile_variable_write_double.sqlite")

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.00001)
    pset.execute(pset.Kernel(Update_lon), endtime=0.001, dt=0.00001, output_file=ofile)

    ds = convert_sqlite_file(filepath)
    lons = ds['lon'][:]
    assert (isinstance(lons.values[0, 0], np.float64))
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_dtypes_pfile(fieldset, pset_mode, mode, tmpdir):
    filepath = tmpdir.join("pfile_dtypes.sqlite")

    dtypes = ['float64', 'int64']  # only these dtypes are supported by sqlite3

    class MyParticle(ptype[mode]):
        for d in dtypes:
            # need an exec() here because we need to dynamically set the variable name
            exec(f'v_{d} = Variable("v_{d}", dtype=np.{d}, initial=0.)')

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=MyParticle, lon=0, lat=0, time=0)
    pfile = pset.ParticleFile(name=filepath, outputdt=1)
    pset.execute(DoNothing, runtime=1, dt=1, output_file=pfile)

    ds = convert_sqlite_file(filepath)
    for d in dtypes:
        assert ds[f'v_{d}'].dtype == d


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('npart', [1, 2, 5])
def test_variable_write_age(fieldset, pset_mode, mode, tmpdir, npart):
    filepath = tmpdir.join("pfile_once_written_variables.sqlite")

    def Update_v(particle, fieldset, time):
        particle.v += 1.
        particle.age += particle.dt

    class MyParticle(ptype[mode]):
        v = Variable('v', dtype=np.float64, initial=0)
        age = Variable('age', dtype=np.float32, initial=0.)
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    time = np.arange(0, npart/10., 0.1, dtype=np.float64)
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=MyParticle, lon=lon, lat=lat, time=time, v=time)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.1)
    pset.execute(pset.Kernel(Update_v), endtime=1, dt=0.1, output_file=ofile)

    assert np.allclose(pset.v - time - pset.age*10, 0, atol=1e-5)
    ds = convert_sqlite_file(filepath)
    vfile = np.ma.filled(ds['v'][:], np.nan)
    for p in range(npart):
        v = vfile[p, :]
        assert np.allclose(v[~np.isnan(v)][0], time[p]+1)
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

    if type == 'repeatdt':
        pset = pset_type[pset_mode]['pset'](fieldset, lon=[0], lat=[0], pclass=MyParticle, repeatdt=repeatdt)
    elif type == 'timearr':
        pset = pset_type[pset_mode]['pset'](fieldset, lon=np.zeros(runtime), lat=np.zeros(runtime), pclass=MyParticle, time=list(range(runtime)))
    filepath = tmpdir.join("pfile_repeated_release.sqlite")
    pfile = pset.ParticleFile(filepath, outputdt=abs(dt))

    def IncrLon(particle, fieldset, time):
        particle.sample_var += 1.
        if particle.sample_var > fieldset.maxvar:
            particle.delete()
    for i in range(runtime):
        pset.execute(pset.Kernel(IncrLon, delete_cfiles=False), dt=dt, runtime=1., output_file=pfile)

    ds = convert_sqlite_file(filepath)
    samplevar = ds['sample_var'][:]
    if type == 'repeatdt':
        assert samplevar.shape == (runtime // repeatdt, runtime)
        assert np.allclose(pset.sample_var, np.arange(maxvar, -1, -repeatdt))
    elif type == 'timearr':
        assert samplevar.shape == (runtime, runtime)
    filesize = os.path.getsize(str(filepath))
    assert filesize < 1024 * 65  # test that chunking leads to filesize less than 65KB
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_timebackward(fieldset, pset_mode, mode, tmpdir):
    filepath = tmpdir.join("pfile_write_timebackward.sqlite")

    def Update_lon(particle, fieldset, time):
        particle.lon -= 0.1 * particle.dt

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lat=np.linspace(0, 1, 3), lon=[0, 0, 0], time=[1, 2, 3])
    pfile = pset.ParticleFile(name=filepath, outputdt=1.)
    pset.execute(pset.Kernel(Update_lon), runtime=4, dt=-1.,
                 output_file=pfile)
    ds = convert_sqlite_file(filepath)
    trajs = ds['trajectory'][:]
    dt = np.diff(ds['time'][:])
    assert trajs.values.dtype == 'int64'
    assert np.all(np.diff(trajs.values) > 0)
    assert np.allclose(dt[np.isfinite(dt)], np.timedelta64(1, 's'), atol=np.timedelta64(1, 'us'))
    ds.close()


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_reset_dt(fieldset, pset_mode, mode, tmpdir):
    # Assert that p.dt gets reset when a write_time is not a multiple of dt
    # for p.dt=0.02 to reach outputdt=0.05 and endtime=0.1, the steps should be [0.2, 0.2, 0.1, 0.2, 0.2, 0.1], resulting in 6 kernel executions
    filepath = tmpdir.join("pfile_reset_dt.sqlite")

    def Update_lon(particle, fieldset, time):
        particle.lon += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.05)
    pset.execute(pset.Kernel(Update_lon), endtime=0.11, dt=0.02, output_file=ofile)

    assert np.allclose(pset.lon, .6)
