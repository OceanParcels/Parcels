from datetime import timedelta as delta
from os import path

import numpy as np
import pytest
import xarray as xr

from parcels import AdvectionRK4
from parcels import FieldSet
from parcels import JITParticle
from parcels import ScipyParticle
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa


pset_modes = ['soa', 'aos']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


def set_swash_fieldset(use_xarray=False):
    filenames = path.join(path.dirname(__file__), 'SWASH_data', 'field_*.nc')
    variables = {'U': 'cross-shore velocity',
                 'V': 'along-shore velocity',
                 'depth_u': 'time varying depth_u'}
    dimensions = {'U': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'V': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'},
                  'depth_u': {'lon': 'x', 'lat': 'y', 'depth': 'not_yet_set', 'time': 't'}}
    if use_xarray:
        ds = xr.open_mfdataset(filenames, combine='by_coords')
        fieldset = FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', allow_time_extrapolation=True)
    else:
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='flat', allow_time_extrapolation=True)
    fieldset.U.set_depth_from_field(fieldset.depth_u)
    fieldset.V.set_depth_from_field(fieldset.depth_u)
    return fieldset


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_swash_xarray_vs_netcdf(pset_mode, mode):
    fieldsetNetcdf = set_swash_fieldset(use_xarray=False)
    fieldsetxarray = set_swash_fieldset(use_xarray=True)

    lonstart, latstart, depthstart, dt = (9.5, 12.5, -0.1, delta(seconds=0.005))

    psetN = pset_type[pset_mode]['pset'](fieldsetNetcdf, pclass=ptype[mode], lon=lonstart, lat=latstart, depth=depthstart)
    psetN.execute(AdvectionRK4, dt=dt)

    psetX = pset_type[pset_mode]['pset'](fieldsetxarray, pclass=ptype[mode], lon=lonstart, lat=latstart, depth=depthstart)
    psetX.execute(AdvectionRK4, dt=dt)

    assert np.allclose(psetN[0].lon, psetX[0].lon)
    assert np.allclose(psetN[0].lat, psetX[0].lat)
    assert np.allclose(psetN[0].depth, psetX[0].depth)
