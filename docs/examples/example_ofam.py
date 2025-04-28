import gc
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

import parcels


def set_ofam_fieldset(use_xarray=False):
    data_folder = parcels.download_example_dataset("OFAM_example_data")
    filenames = {
        "U": f"{data_folder}/OFAM_simple_U.nc",
        "V": f"{data_folder}/OFAM_simple_V.nc",
    }
    variables = {"U": "u", "V": "v"}
    dimensions = {
        "lat": "yu_ocean",
        "lon": "xu_ocean",
        "depth": "st_ocean",
        "time": "Time",
    }
    if use_xarray:
        ds = xr.open_mfdataset([filenames["U"], filenames["V"]], combine="by_coords")
        return parcels.FieldSet.from_xarray_dataset(ds, variables, dimensions)
    else:
        return parcels.FieldSet.from_netcdf(filenames, variables, dimensions)


@pytest.mark.parametrize("use_xarray", [True, False])
def test_ofam_fieldset_fillvalues(use_xarray):
    fieldset = set_ofam_fieldset(use_xarray=use_xarray)
    # V.data[0, 0, 150] is a landpoint, that makes NetCDF4 generate a masked array, instead of an ndarray
    assert fieldset.V.data[0, 0, 150] == 0


@pytest.mark.parametrize("dt", [timedelta(minutes=-5), timedelta(minutes=5)])
def test_ofam_xarray_vs_netcdf(dt):
    fieldsetNetcdf = set_ofam_fieldset(use_xarray=False)
    fieldsetxarray = set_ofam_fieldset(use_xarray=True)
    lonstart, latstart, runtime = (180, 10, timedelta(days=7))

    psetN = parcels.ParticleSet(
        fieldsetNetcdf, pclass=parcels.Particle, lon=lonstart, lat=latstart
    )
    psetN.execute(parcels.AdvectionRK4, runtime=runtime, dt=dt)

    psetX = parcels.ParticleSet(
        fieldsetxarray, pclass=parcels.Particle, lon=lonstart, lat=latstart
    )
    psetX.execute(parcels.AdvectionRK4, runtime=runtime, dt=dt)

    assert np.allclose(psetN[0].lon, psetX[0].lon)
    assert np.allclose(psetN[0].lat, psetX[0].lat)


@pytest.mark.parametrize("use_xarray", [True, False])
def test_ofam_particles(use_xarray):
    gc.collect()
    fieldset = set_ofam_fieldset(use_xarray=use_xarray)

    lonstart = [180]
    latstart = [10]
    depstart = [2.5]  # the depth of the first layer in OFAM

    pset = parcels.ParticleSet(
        fieldset,
        pclass=parcels.Particle,
        lon=lonstart,
        lat=latstart,
        depth=depstart,
    )

    pset.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=10), dt=timedelta(minutes=5)
    )

    assert abs(pset[0].lon - 173) < 1
    assert abs(pset[0].lat - 11) < 1
