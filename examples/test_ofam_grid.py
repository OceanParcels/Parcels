from parcels import Grid
import pytest
from os import path


@pytest.fixture(scope='module')
def filepath():
    return path.join(path.dirname(__file__), "OFAM_example_data")


def test_ofam_grid(filepath):
    filenames = {'U': path.join(filepath, "OFAM_simple_U.nc"),
                 'V': path.join(filepath, "OFAM_simple_V.nc")}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'depth': 'st_ocean',
                  'time': 'Time'}
    grid = Grid.from_netcdf(filenames, variables, dimensions)
    assert(grid.U.lon.size == 2001)
    assert(grid.U.lat.size == 601)
    assert(grid.U.data.shape == (4, 601, 2001))
    assert(grid.V.lon.size == 2001)
    assert(grid.V.lat.size == 601)
    assert(grid.V.data.shape == (4, 601, 2001))
