import xgcm

from parcels._datasets.structured.generic import datasets


def test_left_indexed_dataset():
    """Checks that 'ds_2d_left' is right indexed on all variables."""
    ds = datasets["ds_2d_left"]
    grid = xgcm.Grid(ds)

    for _axis_name, axis in grid.axes.items():
        for pos, _dim_name in axis.coords.items():
            assert pos in ["left", "center"]


def test_right_indexed_dataset():
    """Checks that 'ds_2d_right' is right indexed on all variables."""
    ds = datasets["ds_2d_right"]
    grid = xgcm.Grid(ds)
    for _axis_name, axis in grid.axes.items():
        for pos, _dim_name in axis.coords.items():
            assert pos in ["center", "right"]
