import numpy as np

from parcels._datasets.structured.generic import datasets
from parcels._index_search import curvilinear_point_in_cell
from parcels.xgrid import XGrid


def test_spatialhash_init():
    ds = datasets["2d_left_rotated"]
    grid = XGrid.from_dataset(ds)
    spatialhash = grid.get_spatial_hash()
    assert spatialhash is not None


def test_invalid_positions():
    ds = datasets["2d_left_rotated"]
    grid = XGrid.from_dataset(ds)

    j, i = grid.get_spatial_hash().query([np.nan, np.inf], [np.nan, np.inf], curvilinear_point_in_cell)
    assert np.all(j == -1)
    assert np.all(i == -1)


def test_mixed_positions():
    ds = datasets["2d_left_rotated"]
    grid = XGrid.from_dataset(ds)
    lat = grid.lat.mean()
    lon = grid.lon.mean()
    y = [lat, np.nan]
    x = [lon, np.nan]
    j, i = grid.get_spatial_hash().query(y, x, curvilinear_point_in_cell)
    assert j[0] == 29  # Actual value for 2d_left_rotated center
    assert i[0] == 14  # Actual value for 2d_left_rotated center
    assert j[1] == -1
    assert i[1] == -1
