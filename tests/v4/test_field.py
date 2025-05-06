import pytest
import uxarray as ux
import xarray as xr

from parcels import Field
from parcels._datasets.structured.generic import datasets as structured_datasets
from parcels._datasets.unstructured.generic import datasets as unstructured_datasets
from parcels.v4.grid import Grid


def test_field_init_param_types():
    data = xr.DataArray(
        attrs={
            "location": "node",
            "mesh": "flat",
        }
    )
    grid = Grid(data)
    with pytest.raises(ValueError, match="Expected `name` to be a string"):
        Field(name=123, data=data, grid=grid)

    with pytest.raises(ValueError, match="Expected `data` to be a uxarray.UxDataArray or xarray.DataArray"):
        Field(name="test", data=123, grid=grid)

    with pytest.raises(ValueError, match="Expected `grid` to be a uxarray.Grid or parcels Grid"):
        Field(name="test", data=data, grid=123)

    with pytest.raises(ValueError, match="Invalid value 'invalid'. Valid options are.*"):
        Field(name="test", data=data, grid=grid, mesh_type="invalid")


@pytest.mark.parametrize(
    "data,grid",
    [
        pytest.param(ux.UxDataArray(), Grid(xr.Dataset()), id="uxdata-grid"),
        pytest.param(
            xr.DataArray(),
            unstructured_datasets["stommel_gyre_delaunay"].uxgrid,
            id="xarray-uxgrid",
        ),
    ],
)
def test_field_incompatible_combination(data, grid):
    with pytest.raises(ValueError, match="Incompatible data-grid combination."):
        Field(
            name="test_field",
            data=data,
            grid=grid,
        )


@pytest.mark.parametrize(
    "data,grid",
    [
        pytest.param(
            structured_datasets["ds_2d_left"]["data_g"], Grid(structured_datasets["ds_2d_left"]), id="ds_2d_left"
        ),  # TODO: Perhaps this test should be expanded to cover more datasets?
    ],
)
def test_field_structured_grid_creation(data, grid):
    """Test creating a field."""
    field = Field(
        name="test_field",
        data=data,
        grid=grid,
    )
    assert field.name == "test_field"
    assert field.data.equals(data)
    assert field.grid == grid


def test_field_unstructured_grid_creation(): ...


def test_field_interpolation(): ...


def test_field_interpolation_out_of_spatial_bounds(): ...


def test_field_interpolation_out_of_time_bounds(): ...


def test_field_allow_time_extrapolation(): ...
