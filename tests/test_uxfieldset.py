from datetime import timedelta

import numpy as np
import pytest
import uxarray as ux

from parcels import (
    UXFieldSet,
    ParticleSet,
    Particle
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
    print(type(fieldset))
    pset = ParticleSet(fieldset, pclass=Particle)