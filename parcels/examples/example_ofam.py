import gc
from datetime import timedelta as delta
from os import path

import numpy as np
import pytest
import xarray as xr

from parcels import AdvectionRK4
from parcels import FieldSet
from parcels import JITParticle
from parcels import ParticleSet
from parcels import ScipyParticle


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def set_ofam_fieldset(deferred_load=True, use_xarray=False):
    filenames = {'U': path.join(path.dirname(__file__), 'OFAM_example_data', 'OFAM_simple_U.nc'),
                 'V': path.join(path.dirname(__file__), 'OFAM_example_data', 'OFAM_simple_V.nc')}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'depth': 'st_ocean',
                  'time': 'Time'}
    if use_xarray:
        ds = xr.open_mfdataset([filenames['U'], filenames['V']], combine='by_coords')
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, allow_time_extrapolation=True)
    else:
        return FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, deferred_load=deferred_load, chunksize=False)


@pytest.mark.parametrize('use_xarray', [True, False])
def test_ofam_fieldset_fillvalues(use_xarray):
    fieldset = set_ofam_fieldset(deferred_load=False, use_xarray=use_xarray)
    # V.data[0, 0, 150] is a landpoint, that makes NetCDF4 generate a masked array, instead of an ndarray
    assert(fieldset.V.data[0, 0, 150] == 0)


@pytest.mark.parametrize('dt', [delta(minutes=-5), delta(minutes=5)])
def test_ofam_xarray_vs_netcdf(dt):
    fieldsetNetcdf = set_ofam_fieldset(use_xarray=False)
    fieldsetxarray = set_ofam_fieldset(use_xarray=True)
    lonstart, latstart, runtime = (180, 10, delta(days=7))

    psetN = ParticleSet(fieldsetNetcdf, pclass=JITParticle, lon=lonstart, lat=latstart)
    psetN.execute(AdvectionRK4, runtime=runtime, dt=dt)

    psetX = ParticleSet(fieldsetxarray, pclass=JITParticle, lon=lonstart, lat=latstart)
    psetX.execute(AdvectionRK4, runtime=runtime, dt=dt)

    assert np.allclose(psetN[0].lon, psetX[0].lon)
    assert np.allclose(psetN[0].lat, psetX[0].lat)


@pytest.mark.parametrize('use_xarray', [True, False])
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_ofam_particles(mode, use_xarray):
    gc.collect()
    fieldset = set_ofam_fieldset(use_xarray=use_xarray)

    lonstart = [180]
    latstart = [10]
    depstart = [2.5]  # the depth of the first layer in OFAM

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lonstart, lat=latstart, depth=depstart)

    pset.execute(AdvectionRK4, runtime=delta(days=10), dt=delta(minutes=5))

    assert(abs(pset[0].lon - 173) < 1)
    assert(abs(pset[0].lat - 11) < 1)
