import numpy as np
import pytest
import uxarray as ux
import xarray as xr

from parcels import Field
from parcels._datasets.structured.generic import T as T_structured
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
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
            datasets_unstructured["stommel_gyre_delaunay"].uxgrid,
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
            datasets_structured["ds_2d_left"]["data_g"], Grid(datasets_structured["ds_2d_left"]), id="ds_2d_left"
        ),  # TODO: Perhaps this test should be expanded to cover more datasets?
    ],
)
def test_field_init_structured_grid(data, grid):
    """Test creating a field."""
    field = Field(
        name="test_field",
        data=data,
        grid=grid,
    )
    assert field.name == "test_field"
    assert field.data.equals(data)
    assert field.grid == grid


@pytest.mark.parametrize("numpy_dtype", ["timedelta64[s]", "float64"])
def test_field_init_fail_on_bad_time_type(numpy_dtype):
    """Tests that field initialisation fails when the time isn't given as datetime object (i.e., is float or timedelta)."""
    ds = datasets_structured["ds_2d_left"].copy()
    ds["time"] = np.arange(0, T_structured, dtype=numpy_dtype)

    data = ds["data_g"]
    grid = Grid(ds)
    with pytest.raises(
        ValueError,
        match="Error getting time interval.*. Are you sure that the time dimension on the xarray dataset is stored as datetime or cftime datetime objects\?",
    ):
        Field(
            name="test_field",
            data=data,
            grid=grid,
        )


@pytest.mark.parametrize(
    "data,grid",
    [
        pytest.param(
            datasets_structured["ds_2d_left"]["data_g"], Grid(datasets_structured["ds_2d_left"]), id="ds_2d_left"
        ),
    ],
)
def test_field_time_interval(data, grid):
    """Test creating a field."""
    field = Field(name="test_field", data=data, grid=grid, mesh_type="flat")
    assert field.time_interval.left == np.datetime64("2000-01-01")
    assert field.time_interval.right == np.datetime64("2001-01-01")


def test_vectorfield_init_different_time_intervals():
    # Tests that a VectorField raises a ValueError if the component fields have different time domains.
    ...


def test_field_unstructured_grid_creation(): ...


def test_field_interpolation(): ...


def test_field_interpolation_out_of_spatial_bounds(): ...


def test_field_interpolation_out_of_time_bounds(): ...
