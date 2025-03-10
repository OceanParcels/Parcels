from datetime import timedelta

import numpy as np
import pytest
import uxarray as ux

from parcels import (
    UXFieldSet,
    ParticleSet,
    Particle,
    UxAdvectionEuler
)

from tests.utils import TEST_DATA


def test_fesom_fieldset():
    # Load a FESOM dataset
    grid_path=f"{TEST_DATA}/fesom_channel.nc"
    data_path=[f"{TEST_DATA}/u.fesom_channel.nc",
               f"{TEST_DATA}/v.fesom_channel.nc",
               f"{TEST_DATA}/w.fesom_channel.nc"]
    ds = ux.open_mfdataset(grid_path,data_path)
    fieldset = UXFieldSet(ds)
    fieldset._check_complete()
    # Check that the fieldset has the expected properties
    assert fieldset.uxds == ds

def test_fesom_in_particleset():
        # Load a FESOM dataset
    grid_path=f"{TEST_DATA}/fesom_channel.nc"
    data_path=[f"{TEST_DATA}/u.fesom_channel.nc",
               f"{TEST_DATA}/v.fesom_channel.nc",
               f"{TEST_DATA}/w.fesom_channel.nc"]
    ds = ux.open_mfdataset(grid_path,data_path)
    fieldset = UXFieldSet(ds)
    pset = ParticleSet(fieldset, pclass=Particle)

def test_advection_fesom_channel():
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    grid_path=f"{TEST_DATA}/fesom_channel.nc"
    data_path=[f"{TEST_DATA}/u.fesom_channel.nc",
               f"{TEST_DATA}/v.fesom_channel.nc",
               f"{TEST_DATA}/w.fesom_channel.nc"]
    ds = ux.open_mfdataset(grid_path,data_path)
    fieldset = UXFieldSet(ds)
    print(f"Spatial hash grid shape : {fieldset._spatialhash._nx}, {fieldset._spatialhash._ny}")
    npart = 10
    pset = ParticleSet(
        fieldset, 
        pclass=Particle, 
        lon=np.zeros(npart) + 2.0, 
        lat=np.linspace(5, 15, npart))
    pset.execute(UxAdvectionEuler, runtime=timedelta(hours=24), dt=timedelta(seconds=600))


    #assert (np.diff(pset3D.lon) > 1.0e-4).all()
