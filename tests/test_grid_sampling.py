from parcels import Grid
import numpy as np
import pytest


@pytest.fixture
def grid(xdim=200, ydim=100):
    """ Standard grid spanning the earth's coordinates with U and V
        equivalent to longitude and latitude.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)
    U, V = np.meshgrid(lat, lon)
    return Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat,
                          depth, time)


def test_grid_sample(grid, xdim=120, ydim=80):
    """ Sample the grid using indexing notation. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V[0, x, 70.] for x in lon])
    u_s = np.array([grid.U[0, -45., y] for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-12)
    assert np.allclose(u_s, lat, rtol=1e-12)


def test_grid_sample_eval(grid, xdim=60, ydim=60):
    """ Sample the grid using the explicit eval function. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([grid.V.eval(0, x, 70.) for x in lon])
    u_s = np.array([grid.U.eval(0, -45., y) for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-12)
    assert np.allclose(u_s, lat, rtol=1e-12)
