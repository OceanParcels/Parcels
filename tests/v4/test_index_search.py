import numpy as np
import pytest
import xarray as xr
from xgcm import Grid

from parcels._datasets.structured.generic import datasets
from parcels._index_search import _search_indices_curvilinear_2d
from parcels.field import Field
from parcels.tools.exampledata_utils import download_example_dataset
from parcels.xgrid import XGrid


@pytest.fixture
def field_cone():
    ds = datasets["2d_left_unrolled_cone"]
    grid = XGrid.from_dataset(ds)
    field = Field(
        name="test_field",
        data=ds["data_g"],
        grid=grid,
    )
    return field


def test_grid_indexing_fpoints(field_cone):
    grid = field_cone.grid

    for yi_expected in range(grid.ydim - 1):
        for xi_expected in range(grid.xdim - 1):
            x = np.array([grid.lon[yi_expected, xi_expected] + 0.00001])
            y = np.array([grid.lat[yi_expected, xi_expected] + 0.00001])

            yi, eta, xi, xsi = _search_indices_curvilinear_2d(grid, y, x)
            if eta > 0.9:
                yi_expected -= 1
            if xsi > 0.9:
                xi_expected -= 1
            assert yi == yi_expected, f"Expected yi {yi_expected} but got {yi}"
            assert xi == xi_expected, f"Expected xi {xi_expected} but got {xi}"

            cell_lon = [
                grid.lon[yi, xi],
                grid.lon[yi, xi + 1],
                grid.lon[yi + 1, xi + 1],
                grid.lon[yi + 1, xi],
            ]
            cell_lat = [
                grid.lat[yi, xi],
                grid.lat[yi, xi + 1],
                grid.lat[yi + 1, xi + 1],
                grid.lat[yi + 1, xi],
            ]
            assert x > np.min(cell_lon) and x < np.max(cell_lon)
            assert y > np.min(cell_lat) and y < np.max(cell_lat)


def test_indexing_nemo_curvilinear():
    data_folder = download_example_dataset("NemoCurvilinear_data")
    ds = xr.open_mfdataset(
        data_folder.glob("*.nc4"), combine="nested", data_vars="minimal", coords="minimal", compat="override"
    )
    ds = ds.isel({"time_counter": 0, "time": 0, "z_a": 0}, drop=True).rename(
        {"glamf": "lon", "gphif": "lat", "z": "depth"}
    )
    xgcm_grid = Grid(ds, coords={"X": {"left": "x"}, "Y": {"left": "y"}}, periodic=False)
    grid = XGrid(xgcm_grid, mesh="spherical")

    # Test points on the NEMO 1/4 degree curvilinear grid
    lats = np.array([-30, 0, 88])
    lons = np.array([30, 60, -150])

    yi, eta, xi, xsi = _search_indices_curvilinear_2d(grid, lats, lons)

    # Construct cornerpoints px
    px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])

    # Maximum 5 degree difference between px values
    for i in range(lons.shape[0]):
        np.testing.assert_allclose(px[1, i], px[:, i], atol=5)

    # Reconstruct lons values from cornerpoints
    xx = (1 - xsi) * (1 - eta) * px[0] + xsi * (1 - eta) * px[1] + xsi * eta * px[2] + (1 - xsi) * eta * px[3]
    np.testing.assert_allclose(xx, lons, atol=1e-6)
