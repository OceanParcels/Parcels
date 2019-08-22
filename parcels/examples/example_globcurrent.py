from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4, Variable
from datetime import timedelta as delta
from os import path
from glob import glob
import numpy as np
import pytest
import xarray as xr


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def set_globcurrent_fieldset(filename=None, indices=None, deferred_load=True, use_xarray=False, time_periodic=False, timestamps=None):
    if filename is None:
        filename = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                             '20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    if timestamps is None:
        dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    else:
        dimensions = {'lat': 'lat', 'lon': 'lon'}
    if use_xarray:
        ds = xr.open_mfdataset(filename)
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, indices, deferred_load=deferred_load, time_periodic=time_periodic)
    else:
        return FieldSet.from_netcdf(filename, variables, dimensions, indices, deferred_load=deferred_load, time_periodic=time_periodic, timestamps=timestamps)


@pytest.mark.parametrize('use_xarray', [True, False])
def test_globcurrent_fieldset(use_xarray):
    fieldset = set_globcurrent_fieldset(use_xarray=use_xarray)
    assert(fieldset.U.lon.size == 81)
    assert(fieldset.U.lat.size == 41)
    assert(fieldset.V.lon.size == 81)
    assert(fieldset.V.lat.size == 41)

    indices = {'lon': [5], 'lat': range(20, 30)}
    fieldsetsub = set_globcurrent_fieldset(indices=indices, use_xarray=use_xarray)
    assert np.allclose(fieldsetsub.U.lon, fieldset.U.lon[indices['lon']])
    assert np.allclose(fieldsetsub.U.lat, fieldset.U.lat[indices['lat']])
    assert np.allclose(fieldsetsub.V.lon, fieldset.V.lon[indices['lon']])
    assert np.allclose(fieldsetsub.V.lat, fieldset.V.lat[indices['lat']])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('dt, lonstart, latstart', [(3600., 25, -35), (-3600., 20, -39)])
@pytest.mark.parametrize('use_xarray', [True, False])
def test_globcurrent_fieldset_advancetime(mode, dt, lonstart, latstart, use_xarray):
    basepath = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                         '20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    files = sorted(glob(str(basepath)))

    fieldsetsub = set_globcurrent_fieldset(files[0:10], use_xarray=use_xarray)
    psetsub = ParticleSet.from_list(fieldset=fieldsetsub, pclass=ptype[mode], lon=[lonstart], lat=[latstart])

    fieldsetall = set_globcurrent_fieldset(files[0:10], deferred_load=False, use_xarray=use_xarray)
    psetall = ParticleSet.from_list(fieldset=fieldsetall, pclass=ptype[mode], lon=[lonstart], lat=[latstart])
    if dt < 0:
        psetsub[0].time = fieldsetsub.U.grid.time[-1]
        psetall[0].time = fieldsetall.U.grid.time[-1]

    psetsub.execute(AdvectionRK4, runtime=delta(days=7), dt=dt)
    psetall.execute(AdvectionRK4, runtime=delta(days=7), dt=dt)

    assert abs(psetsub[0].lon - psetall[0].lon) < 1e-4


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('use_xarray', [True, False])
def test_globcurrent_particles(mode, use_xarray):
    fieldset = set_globcurrent_fieldset(use_xarray=use_xarray)

    lonstart = [25]
    latstart = [-35]

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lonstart, lat=latstart)

    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5))

    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('rundays', [300, 900])
def test_globcurrent_time_periodic(mode, rundays):
    sample_var = []
    for deferred_load in [True, False]:
        fieldset = set_globcurrent_fieldset(time_periodic=True, deferred_load=deferred_load)

        class MyParticle(ptype[mode]):
            sample_var = Variable('sample_var', initial=fieldset.U)

        pset = ParticleSet(fieldset, pclass=MyParticle, lon=25, lat=-35, time=fieldset.U.grid.time[0])

        def SampleU(particle, fieldset, time):
            particle.sample_var += fieldset.U[time, particle.depth, particle.lat, particle.lon]

        pset.execute(SampleU, runtime=delta(days=rundays), dt=delta(days=1))
        sample_var.append(pset[0].sample_var)

    assert np.allclose(sample_var[0], sample_var[1])


@pytest.mark.parametrize('dt', [-300, 300])
def test_globcurrent_xarray_vs_netcdf(dt):
    fieldsetNetcdf = set_globcurrent_fieldset(use_xarray=False)
    fieldsetxarray = set_globcurrent_fieldset(use_xarray=True)
    lonstart, latstart, runtime = (25, -35, delta(days=7))

    psetN = ParticleSet(fieldsetNetcdf, pclass=JITParticle, lon=lonstart, lat=latstart)
    psetN.execute(AdvectionRK4, runtime=runtime, dt=dt)

    psetX = ParticleSet(fieldsetxarray, pclass=JITParticle, lon=lonstart, lat=latstart)
    psetX.execute(AdvectionRK4, runtime=runtime, dt=dt)

    assert np.allclose(psetN[0].lon, psetX[0].lon)
    assert np.allclose(psetN[0].lat, psetX[0].lat)


@pytest.mark.parametrize('dt', [-300, 300])
def test_globcurrent_netcdf_timestamps(dt):
    fieldsetNetcdf = set_globcurrent_fieldset()
    timestamps = fieldsetNetcdf.U.grid.timeslices
    fieldsetTimestamps = set_globcurrent_fieldset(timestamps=timestamps)
    lonstart, latstart, runtime = (25, -35, delta(days=7))

    psetN = ParticleSet(fieldsetNetcdf, pclass=JITParticle, lon=lonstart, lat=latstart)
    psetN.execute(AdvectionRK4, runtime=runtime, dt=dt)

    psetT = ParticleSet(fieldsetTimestamps, pclass=JITParticle, lon=lonstart, lat=latstart)
    psetT.execute(AdvectionRK4, runtime=runtime, dt=dt)

    assert np.allclose(psetN[0].lon, psetT[0].lon)
    assert np.allclose(psetN[0].lat, psetT[0].lat)


def test__particles_init_time():
    fieldset = set_globcurrent_fieldset()

    lonstart = [25]
    latstart = [-35]

    # tests the different ways of initialising the time of a particle
    pset = ParticleSet(fieldset, pclass=JITParticle, lon=lonstart, lat=latstart, time=np.datetime64('2002-01-15'))
    pset2 = ParticleSet(fieldset, pclass=JITParticle, lon=lonstart, lat=latstart, time=14*86400)
    pset3 = ParticleSet(fieldset, pclass=JITParticle, lon=lonstart, lat=latstart, time=np.array([np.datetime64('2002-01-15')]))
    pset4 = ParticleSet(fieldset, pclass=JITParticle, lon=lonstart, lat=latstart, time=[np.datetime64('2002-01-15')])
    assert pset[0].time - pset2[0].time == 0
    assert pset[0].time - pset3[0].time == 0
    assert pset[0].time - pset4[0].time == 0


@pytest.mark.xfail(reason="Time extrapolation error expected to be thrown", strict=True)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('use_xarray', [True, False])
def test_globcurrent_time_extrapolation_error(mode, use_xarray):
    fieldset = set_globcurrent_fieldset(use_xarray=use_xarray)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[25], lat=[-35],
                       time=fieldset.U.time[0]-delta(days=1).total_seconds())

    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('dt', [-300, 300])
@pytest.mark.parametrize('use_xarray', [True, False])
def test_globcurrent_variable_fromfield(mode, dt, use_xarray):
    fieldset = set_globcurrent_fieldset(use_xarray=use_xarray)

    class MyParticle(ptype[mode]):
        sample_var = Variable('sample_var', initial=fieldset.U)
    time = fieldset.U.grid.time[0] if dt > 0 else fieldset.U.grid.time[-1]
    pset = ParticleSet(fieldset, pclass=MyParticle, lon=[25], lat=[-35], time=time)

    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=dt)


@pytest.mark.parametrize('deferred_load', [True, False])
@pytest.mark.parametrize('use_xarray', [True, False])
def test_globcurrent_deferred_fieldset_gradient(deferred_load, use_xarray):
    fieldset = set_globcurrent_fieldset(deferred_load=deferred_load, use_xarray=use_xarray)
    (dU_dx, dU_dy) = fieldset.U.gradient()
    fieldset.add_field(dU_dy)

    pset = ParticleSet(fieldset, pclass=JITParticle, lon=25, lat=-35)
    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(days=1))

    tdim = 3 if deferred_load else 366
    assert(dU_dx.data.shape == (tdim, 41, 81))
    assert(fieldset.dU_dy.data.shape == (tdim, 41, 81))
    assert(dU_dx is fieldset.U.gradientx)
