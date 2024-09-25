import inspect

import pytest

from parcels import Field, FieldSet
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


class FieldPrivate:
    attributes = [
        "_dataFiles",
        "_loaded_time_indices",
        "_creation_log",
        "_data_chunks",
        "_c_data_chunks",
        "_chunk_set",
    ]
    methods = [
        "_get_dim_filenames",
        "_collect_timeslices",
        "_reshape",
        "_calc_cell_edge_sizes",
        "_search_indices_vertical_z",
        "_search_indices_vertical_s",
        "_reconnect_bnd_indices",
        "_search_indices_rectilinear",
        "_search_indices_curvilinear",
        "_search_indices",
        "_interpolator2D",
        "_interpolator3D",
        "_ccode_eval",
        "_ccode_convert",
        "_get_block_id",
        "_get_block",
        "_chunk_setup",
        "_chunk_data",
        "_rescale_and_set_minmax",
        "_data_concatenate",
        "_spatial_interpolation",
        "_time_index",
    ]


class FieldSetPrivate:
    attributes = [
        "_completed",
    ]
    methods = [
        "_add_UVfield",
        "_parse_wildcards",
        "_check_complete",
    ]


def assert_private_public_attribute_equiv(obj, private_attribute: str):
    assert private_attribute.startswith("_")
    attribute = private_attribute.lstrip("_")

    with pytest.raises(DeprecationWarning):
        assert hasattr(obj, attribute)
        assert hasattr(obj, private_attribute)
        assert getattr(obj, attribute) is getattr(obj, private_attribute)


def assert_public_method_calls_private(type_, private_method):
    """Looks at the source code to ensure that `public_method` calls `private_method`.

    Looks for the string `.{method_name}(` in the source code of `public_method`.
    """
    assert private_method.startswith("_")
    public_method_str = private_method.lstrip("_")
    private_method_str = private_method

    public_method = getattr(type_, public_method_str)
    private_method = getattr(type_, private_method_str)

    assert callable(public_method)
    assert callable(private_method)

    assert f".{private_method_str}(" in inspect.getsource(public_method)


@pytest.mark.parametrize("private_attribute", FieldPrivate.attributes)
def test_private_attribute_field(private_attribute):
    assert_private_public_attribute_equiv(field, private_attribute)


@pytest.mark.parametrize("private_attribute", FieldSetPrivate.attributes)
def test_private_attribute_fieldset(private_attribute):
    assert_private_public_attribute_equiv(fieldset, private_attribute)


@pytest.mark.parametrize("private_method", FieldPrivate.methods)
def test_private_method_field(private_method):
    assert_public_method_calls_private(Field, private_method)


@pytest.mark.parametrize("private_method", FieldSetPrivate.methods)
def test_private_method_fieldset(private_method):
    assert_public_method_calls_private(FieldSet, private_method)
