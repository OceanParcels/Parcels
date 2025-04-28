import numpy as np
import pytest
import xarray as xr

from parcels import Field, FieldSet, VectorField


@pytest.fixture
def empty_fieldset():
    """Create an empty FieldSet for testing."""
    return FieldSet([])


@pytest.fixture
def fieldset():
    """Create a FieldSet for testing."""
    return FieldSet([])  # TODO: Add fields


def test_field_attr_access(empty_fieldset):
    """Test accessing a field's attributes."""
    empty_fieldset.add_constant_field("test_field", 42.0, mesh="flat")  # TODO: update to use `fieldset` fixture
    assert empty_fieldset.test_field.name == "test_field"


def test_add_constant_field(empty_fieldset):
    """Test adding a constant field to a FieldSet."""
    # Add a constant field
    empty_fieldset.add_constant_field("test_field", 42.0, mesh="flat")

    # Check that the field was added
    assert hasattr(empty_fieldset, "test_field")
    assert isinstance(empty_fieldset.test_field, Field)
    assert empty_fieldset.test_field.name == "test_field"

    # Check the field's data # TODO: What should the shape of the underlying data be? Is this something that we need to test?
    # assert empty_fieldset.test_field.data.shape == (1, 1, 1, 1)
    # assert empty_fieldset.test_field.data[0, 0, 0, 0] == 42.0
    # assert empty_fieldset.test_field.data.dims == ("time", "depth", "lat", "lon")


def test_add_vector_field(empty_fieldset):
    """Test adding a vector field to a FieldSet."""
    # Create two fields for the vector field
    u_data = xr.DataArray(
        data=np.ones((1, 1, 1, 1)),
        name="U",
        dims="null",
        coords=[0.0, [0], [0], [0]],
        attrs=dict(description="null", units="null", location="node", mesh="constant"),
    )
    v_data = xr.DataArray(
        data=np.ones((1, 1, 1, 1)),
        name="V",
        dims="null",
        coords=[0.0, [0], [0], [0]],
        attrs=dict(description="null", units="null", location="node", mesh="constant"),
    )

    u_field = Field("U", u_data)
    v_field = Field("V", v_data)

    # Create and add the vector field
    uv_field = VectorField("UV", u_field, v_field)
    empty_fieldset.add_vector_field(uv_field)

    # Check that the vector field was added
    assert isinstance(empty_fieldset.UV, VectorField)
    assert empty_fieldset.UV.name == "UV"


def test_get_fields(empty_fieldset):
    """Test getting all fields from a FieldSet."""
    # Add some fields
    empty_fieldset.add_constant_field("field1", 1.0)
    empty_fieldset.add_constant_field("field2", 2.0)

    # Create a vector field
    u_data = xr.DataArray(
        data=np.ones((1, 1, 1, 1)),
        name="U",
        dims="null",
        coords=[0.0, [0], [0], [0]],
        attrs=dict(description="null", units="null", location="node", mesh="constant"),
    )
    v_data = xr.DataArray(
        data=np.ones((1, 1, 1, 1)),
        name="V",
        dims="null",
        coords=[0.0, [0], [0], [0]],
        attrs=dict(description="null", units="null", location="node", mesh="constant"),
    )

    u_field = Field("U", u_data)
    v_field = Field("V", v_data)
    empty_fieldset.add_field(u_field)
    empty_fieldset.add_field(v_field)

    uv_field = VectorField("UV", u_field, v_field)
    empty_fieldset.add_vector_field(uv_field)

    # Get all fields
    fields = empty_fieldset.get_fields()

    # Check that we got all fields # TODO: Update these to `in` statements
    assert len(fields) == 5  # field1, field2, U, V, UV
    assert any(isinstance(f, Field) and f.name == "field1" for f in fields)
    assert any(isinstance(f, Field) and f.name == "field2" for f in fields)
    assert any(isinstance(f, Field) and f.name == "U" for f in fields)
    assert any(isinstance(f, Field) and f.name == "V" for f in fields)
    assert any(isinstance(f, VectorField) and f.name == "UV" for f in fields)
