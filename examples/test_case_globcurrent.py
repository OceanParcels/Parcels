from parcels import Grid
import pytest
from os import path


@pytest.fixture(scope='module')
def filepath():
    return path.join(path.dirname(__file__), "GlobCurrent_example_data")


def test_globcurrent_grid():
    filenames = {'U': "examples/GlobCurrent_example_data/20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc",
                 'V': "examples/GlobCurrent_example_data/20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"}
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon',
                  'time': 'time'}
    grid = Grid.from_netcdf(filenames, variables, dimensions)
    assert(grid.U.lon.size == 81)
    assert(grid.U.lat.size == 41)
    assert(grid.U.data.shape == (365, 41, 81))
    assert(grid.V.lon.size == 81)
    assert(grid.V.lat.size == 41)
    assert(grid.V.data.shape == (365, 41, 81))


if __name__ == "__main__":
    test_globcurrent_grid()
