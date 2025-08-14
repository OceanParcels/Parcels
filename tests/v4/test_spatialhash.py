from parcels._datasets.structured.generic import datasets
from parcels.xgrid import XGrid


def test_spatialhash_init():
    ds = datasets["2d_left_rotated"]
    grid = XGrid.from_dataset(ds)
    spatialhash = grid.get_spatial_hash()
    assert spatialhash is not None
