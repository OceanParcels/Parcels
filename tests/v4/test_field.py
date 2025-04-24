import pytest

from parcels import Field
from parcels._datasets.structured.grid_datasets import datasets
from parcels.v4.grid import Grid


@pytest.mark.parametrize(
    "data,grid",
    [
        pytest.param(datasets["ds_2d_left"]["data_g"], Grid(datasets["ds_2d_left"]), id="ds_2d_left"),
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
    assert field.data == data
    assert field.grid == grid


def test_field_structured_grid_creation_spherical():
    # Field(..., mesh_type="spherical")
    ...


def test_field_unstructured_grid_creation(): ...


def test_field_interpolation(): ...


def test_field_interpolation_out_of_spatial_bounds(): ...


def test_field_interpolation_out_of_time_bounds(): ...


def test_field_allow_time_extrapolation(): ...
