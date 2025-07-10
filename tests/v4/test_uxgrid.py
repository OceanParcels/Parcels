import pytest

from parcels._datasets.unstructured.generic import datasets as uxdatasets
from parcels.uxgrid import UxGrid


@pytest.mark.parametrize("uxds", [pytest.param(uxds, id=key) for key, uxds in uxdatasets.items()])
def test_uxgrid_init_on_generic_datasets(uxds):
    UxGrid(uxds.uxgrid, z=uxds.coords["nz"])


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_axes(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["nz"])
    assert grid.axes == ["Z", "FACE"]


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_xgrid_get_axis_dim(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["nz"])

    assert grid.get_axis_dim("FACE") == 721
    assert grid.get_axis_dim("Z") == 2
