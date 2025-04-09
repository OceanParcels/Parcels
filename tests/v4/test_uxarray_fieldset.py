from datetime import timedelta

import pytest
import uxarray as ux

from parcels import (
    FieldSet,
    Particle,
    ParticleSet,
    UXPiecewiseConstantFace,
    UXPiecewiseLinearNode,
    download_example_dataset,
)


@pytest.fixture
def ds_fesom_channel() -> ux.UxDataset:
    fesom_path = download_example_dataset("FESOM_periodic_channel")
    grid_path = f"{fesom_path}/fesom_channel.nc"
    data_path = [
        f"{fesom_path}/u.fesom_channel.nc",
        f"{fesom_path}/v.fesom_channel.nc",
        f"{fesom_path}/w.fesom_channel.nc",
    ]
    ds = ux.open_mfdataset(grid_path, data_path).rename_vars({"u": "U", "v": "V", "w": "W"})
    return ds


def test_fesom_fieldset(ds_fesom_channel):
    fieldset = FieldSet([ds_fesom_channel])
    fieldset._check_complete()
    # Check that the fieldset has the expected properties
    assert fieldset.datasets[0] == ds_fesom_channel


def test_fesom_in_particleset(ds_fesom_channel):
    fieldset = FieldSet([ds_fesom_channel])
    # Check that the fieldset has the expected properties
    assert fieldset.datasets[0] == ds_fesom_channel
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.fieldset == fieldset


def test_set_interp_methods(ds_fesom_channel):
    fieldset = FieldSet([ds_fesom_channel])
    # Set the interpolation method for each field
    fieldset.U.interp_method = UXPiecewiseConstantFace
    fieldset.V.interp_method = UXPiecewiseConstantFace
    fieldset.W.interp_method = UXPiecewiseLinearNode


def test_fesom_channel(ds_fesom_channel):
    fieldset = FieldSet([ds_fesom_channel])
    # Set the interpolation method for each field
    fieldset.U.interp_method = UXPiecewiseConstantFace
    fieldset.V.interp_method = UXPiecewiseConstantFace
    fieldset.W.interp_method = UXPiecewiseLinearNode
    pset = ParticleSet(fieldset, pclass=Particle)
    pset.execute(endtime=timedelta(days=1), dt=timedelta(hours=1))
