
import uxarray as ux
from datetime import timedelta
from parcels import (
    FieldSet,
    ParticleSet,
    Particle,
)
import os

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
    fieldset = FieldSet(ds)
    fieldset._check_complete()
    # Check that the fieldset has the expected properties
    assert fieldset.ds == ds
