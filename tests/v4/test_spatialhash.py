import numpy as np

from parcels._datasets.structured.generic import datasets
from parcels.xgrid import XGrid


def test_spatialhash_init():
    ds = datasets["2d_left_rotated"]
    grid = XGrid.from_dataset(ds)
    spatialhash = grid.get_spatial_hash()
    assert spatialhash is not None


def test_invalid_positions():
    ds = datasets["2d_left_rotated"]
    grid = XGrid.from_dataset(ds)

    j, i, coords = grid.get_spatial_hash().query([np.nan, np.inf], [np.nan, np.inf])
    assert np.all(j == -3)
    assert np.all(i == -3)


def test_mixed_positions():
    ds = datasets["2d_left_rotated"]
    grid = XGrid.from_dataset(ds)
    lat = grid.lat.mean()
    lon = grid.lon.mean()
    y = [lat, np.nan]
    x = [lon, np.nan]
    j, i, coords = grid.get_spatial_hash().query(y, x)
    assert j[0] == 29  # Actual value for 2d_left_rotated center
    assert i[0] == 14  # Actual value for 2d_left_rotated center
    assert j[1] == -3
    assert i[1] == -3
