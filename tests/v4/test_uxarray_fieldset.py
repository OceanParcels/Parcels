import os
from datetime import timedelta

import uxarray as ux

from parcels import (
    FieldSet,
    Particle,
    ParticleSet,
    UXPiecewiseConstantFace,
    UXPiecewiseLinearNode,
)

# Get path of this script
V4_TEST_DATA = f"{os.path.dirname(__file__)}/test_data"


def test_fesom_fieldset():
    # Load a FESOM dataset
    grid_path = f"{V4_TEST_DATA}/fesom_channel.nc"
    data_path = [
        f"{V4_TEST_DATA}/u.fesom_channel.nc",
        f"{V4_TEST_DATA}/v.fesom_channel.nc",
        f"{V4_TEST_DATA}/w.fesom_channel.nc",
    ]
    ds = ux.open_mfdataset(grid_path, data_path)
    ds = ds.rename_vars({"u": "U", "v": "V", "w": "W"})
    fieldset = FieldSet([ds])
    fieldset._check_complete()
    # Check that the fieldset has the expected properties
    assert fieldset.datasets[0] == ds


def test_fesom_in_particleset():
    grid_path = f"{V4_TEST_DATA}/fesom_channel.nc"
    data_path = [
        f"{V4_TEST_DATA}/u.fesom_channel.nc",
        f"{V4_TEST_DATA}/v.fesom_channel.nc",
        f"{V4_TEST_DATA}/w.fesom_channel.nc",
    ]
    ds = ux.open_mfdataset(grid_path, data_path)
    ds = ds.rename_vars({"u": "U", "v": "V", "w": "W"})
    fieldset = FieldSet([ds])
    # Check that the fieldset has the expected properties
    assert fieldset.datasets[0] == ds
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.fieldset == fieldset


def test_set_interp_methods():
    grid_path = f"{V4_TEST_DATA}/fesom_channel.nc"
    data_path = [
        f"{V4_TEST_DATA}/u.fesom_channel.nc",
        f"{V4_TEST_DATA}/v.fesom_channel.nc",
        f"{V4_TEST_DATA}/w.fesom_channel.nc",
    ]
    ds = ux.open_mfdataset(grid_path, data_path)
    ds = ds.rename_vars({"u": "U", "v": "V", "w": "W"})
    fieldset = FieldSet([ds])
    # Set the interpolation method for each field
    fieldset.U.interp_method = UXPiecewiseConstantFace
    fieldset.V.interp_method = UXPiecewiseConstantFace
    fieldset.W.interp_method = UXPiecewiseLinearNode


def test_fesom_channel():
    grid_path = f"{V4_TEST_DATA}/fesom_channel.nc"
    data_path = [
        f"{V4_TEST_DATA}/u.fesom_channel.nc",
        f"{V4_TEST_DATA}/v.fesom_channel.nc",
        f"{V4_TEST_DATA}/w.fesom_channel.nc",
    ]
    ds = ux.open_mfdataset(grid_path, data_path)
    ds = ds.rename_vars({"u": "U", "v": "V", "w": "W"})
    fieldset = FieldSet([ds])
    # Set the interpolation method for each field
    fieldset.U.interp_method = UXPiecewiseConstantFace
    fieldset.V.interp_method = UXPiecewiseConstantFace
    fieldset.W.interp_method = UXPiecewiseLinearNode
    pset = ParticleSet(fieldset, pclass=Particle)
    pset.execute(endtime=timedelta(days=1), dt=timedelta(hours=1))
