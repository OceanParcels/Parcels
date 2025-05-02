import pytest

from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.v4.grid import Grid

ds = datasets_structured["ds_2d_left"]


@pytest.fixture
def fieldset() -> FieldSet:
    """Fixture to create a FieldSet object for testing."""
    grid = Grid(ds)
    U = Field("U", ds["U (A grid)"], grid, mesh_type="flat")
    V = Field("V", ds["V (A grid)"], grid, mesh_type="flat")
    UV = VectorField("UV", U, V)

    return FieldSet(
        [U, V, UV],
    )


def test_fieldset_add_constant(fieldset):
    fieldset.add_constant("test_constant", 1.0)
    assert fieldset.test_constant == 1.0


@pytest.mark.xfail(reason="Not yet implemented.")
def test_fieldset_add_constant_field(fieldset):
    fieldset.add_constant_field("test_constant_field", 1.0)

    # Get a point in the domain
    lon, lat, depth, time = ds["lon"].mean(), ds["lat"].mean(), ds["depth"].mean()

    assert fieldset.test_constant_field[time, depth, lat, lon] == 1.0


@pytest.mark.xfail(reason="need to add __getattr__.")
def test_fieldset_add_field(fieldset):
    grid = Grid(ds)
    field = Field("test_field", ds["U (A grid)"], grid, mesh_type="flat")
    fieldset.add_field(field)
    assert fieldset.test_field == field


def test_fieldset_add_field_wrong_type(fieldset):
    not_a_field = 1.0
    with pytest.raises(ValueError, match="Expected `field` to be a Field or VectorField object. Got .*"):
        fieldset.add_field(not_a_field, "test_field")


def test_fieldset_add_field_already_exists(fieldset):
    grid = Grid(ds)
    field = Field("test_field", ds["U (A grid)"], grid, mesh_type="flat")
    fieldset.add_field(field, "test_field")
    with pytest.raises(ValueError, match="FieldSet already has a Field with name 'test_field'"):
        fieldset.add_field(field, "test_field")


@pytest.mark.xfail(reason="FieldSet doesn't yet correctly handle duplicate grids.")
def test_fieldset_gridset_size(fieldset):
    assert fieldset.gridset_size == 1
