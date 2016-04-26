from parcels import Grid
import numpy as np
import pytest


def generate_grid(xdim, ydim, zdim=1, tdim=1):
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.zeros(zdim, dtype=np.float32)
    time = np.zeros(tdim, dtype=np.float64)
    U, V = np.meshgrid(lon, lat)
    return (np.array(U, dtype=np.float32),
            np.array(V, dtype=np.float32),
            lon, lat, depth, time)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_grid_from_data(xdim, ydim):
    """ Simple test for grid initialisation from data. """
    u, v, lon, lat, depth, time = generate_grid(xdim, ydim)
    grid = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    u_t = np.transpose(u).reshape((lat.size, lon.size))
    v_t = np.transpose(v).reshape((lat.size, lon.size))
    assert len(grid.U.data.shape) == 3  # Will be 4 once we use depth
    assert len(grid.V.data.shape) == 3
    assert np.allclose(grid.U.data[0, :], u_t, rtol=1e-12)
    assert np.allclose(grid.V.data[0, :], v_t, rtol=1e-12)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_grid_from_nemo(xdim, ydim, tmpdir, filename='test_nemo'):
    """ Simple test for grid initialisation from NEMO file format. """
    filepath = tmpdir.join(filename)
    u, v, lon, lat, depth, time = generate_grid(xdim, ydim)
    grid_out = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    grid_out.write(filepath)
    grid = Grid.from_nemo(filepath)
    u_t = np.transpose(u).reshape((lat.size, lon.size))
    v_t = np.transpose(v).reshape((lat.size, lon.size))
    assert len(grid.U.data.shape) == 3  # Will be 4 once we use depth
    assert len(grid.V.data.shape) == 3
    assert np.allclose(grid.U.data[0, :], u_t, rtol=1e-12)
    assert np.allclose(grid.V.data[0, :], v_t, rtol=1e-12)
