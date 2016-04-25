from parcels import Grid
import numpy as np
import pytest


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_grid_from_data(xdim, ydim):
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)
    u, v = np.meshgrid(lon, lat)

    grid = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    u_t = np.transpose(u).reshape((lat.size, lon.size))
    v_t = np.transpose(v).reshape((lat.size, lon.size))
    assert len(grid.U.data.shape) == 3  # Will be 4 once we use depth
    assert len(grid.V.data.shape) == 3
    assert np.allclose(grid.U.data[0, :], u_t, rtol=1e-12)
    assert np.allclose(grid.V.data[0, :], v_t, rtol=1e-12)
