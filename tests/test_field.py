import numpy as np
import pytest

from parcels import Field
from tests.utils import TEST_DATA


def test_field_from_netcdf_variables():
    filename = str(TEST_DATA / "perlinfieldsU.nc")
    dims = {"lon": "x", "lat": "y"}

    variable = "vozocrtx"
    f1 = Field.from_netcdf(filename, variable, dims)
    variable = ("U", "vozocrtx")
    f2 = Field.from_netcdf(filename, variable, dims)
    variable = {"U": "vozocrtx"}
    f3 = Field.from_netcdf(filename, variable, dims)

    assert np.allclose(f1.data, f2.data, atol=1e-12)
    assert np.allclose(f1.data, f3.data, atol=1e-12)

    with pytest.raises(AssertionError):
        variable = {"U": "vozocrtx", "nav_lat": "nav_lat"}  # multiple variables will fail
        f3 = Field.from_netcdf(filename, variable, dims)


@pytest.mark.parametrize("with_timestamps", [True, False])
def test_field_from_netcdf(with_timestamps):
    filenames = {
        "lon": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "lat": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
        "data": str(TEST_DATA / "Uu_eastward_nemo_cross_180lon.nc"),
    }
    variable = "U"
    dimensions = {"lon": "glamf", "lat": "gphif"}
    if with_timestamps:
        timestamp_types = [[[2]], [[np.datetime64("2000-01-01")]]]
        for timestamps in timestamp_types:
            Field.from_netcdf(filenames, variable, dimensions, interp_method="cgrid_velocity", timestamps=timestamps)
    else:
        Field.from_netcdf(filenames, variable, dimensions, interp_method="cgrid_velocity")
