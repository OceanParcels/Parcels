import pytest

from tests.utils import create_fieldset_unit_mesh

fieldset = create_fieldset_unit_mesh()
field = fieldset.U

private_field_attrs = [
    "_dataFiles",
    "_loaded_time_indices",
    "_creation_log",
    "_data_chunks",
    "_c_data_chunks",
    "_chunk_set",
]

private_fieldset_attrs = [
    "_completed",
]


@pytest.mark.parametrize("private_attribute", private_field_attrs)
def test_private_attribute_field(private_attribute):
    assert private_attribute.startswith("_")
    attribute = private_attribute.lstrip("_")

    with pytest.raises(DeprecationWarning):
        assert hasattr(field, attribute)
        assert hasattr(field, private_attribute)
        assert getattr(field, attribute) is getattr(field, private_attribute)


@pytest.mark.parametrize("private_attribute", private_fieldset_attrs)
def test_private_attribute_fieldset(private_attribute):
    assert private_attribute.startswith("_")
    attribute = private_attribute.lstrip("_")

    with pytest.raises(DeprecationWarning):
        assert hasattr(fieldset, attribute)
        assert hasattr(fieldset, private_attribute)
        assert getattr(fieldset, attribute) is getattr(fieldset, private_attribute)
