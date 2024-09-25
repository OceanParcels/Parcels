import inspect
from typing import Literal

import pytest

from parcels import Field, FieldSet
from tests.utils import create_fieldset_unit_mesh


class Action:
    """Utility class to help manage, document, and test deprecations."""

    def __init__(self, class_: Literal["Field", "FieldSet"], name: str, type_: Literal["read_only", "make_private"]):
        if name.startswith("_"):
            raise ValueError("name should not start with an underscore")

        self.class_ = class_
        self._raw_name = name
        self.type_ = type_

        if type_ == "read_only" and self.is_method:
            raise ValueError("read_only attributes should not be methods")

    @property
    def public_name(self):
        return self._raw_name.strip("()")

    @property
    def private_name(self):
        if self.type_ == "make_private":
            return f"_{self.public_name}"
        return None

    @property
    def is_method(self):
        if self._raw_name.endswith("()"):
            return True
        return False

    def __str__(self):
        return f"{self.class_}.{self.public_name}"

    def __repr__(self):
        return f"Action(class_={self.class_!r}, name={self._raw_name!r}, type_={self.type_!r})"


def test_testing_action_class():
    """Testing the Action class used for testing."""
    action = Action("MyClass", "my_attribute", "make_private")
    assert not action.is_method
    assert action.public_name == "my_attribute"
    assert action.private_name == "_my_attribute"
    assert action.class_ == "MyClass"
    assert action.type_ == "make_private"

    action = Action("Field", "my_attribute", "read_only")
    assert not action.is_method
    assert action.public_name == "my_attribute"
    assert action.private_name is None
    assert not action.is_method

    action = Action("Field", "my_method()", "make_private")
    assert action.is_method
    assert action.public_name == "my_method"
    assert action.private_name == "_my_method"

    with pytest.raises(ValueError):  # Can't have underscore in name
        Action("Field", "_my_attribute", "make_private")

    with pytest.raises(ValueError):  # Can't have read-only method
        Action("Field", "my_method()", "read_only")


# fmt: off
actions = [
    Action("Field",      "dataFiles",                       "make_private"  ),
    Action("Field",      "netcdf_engine",                   "read_only"     ),
    Action("Field",      "loaded_time_indices",             "make_private"  ),
    Action("Field",      "creation_log",                    "make_private"  ),
    Action("Field",      "data_chunks",                     "make_private"  ),
    Action("Field",      "c_data_chunks",                   "make_private"  ),
    Action("Field",      "chunk_set",                       "make_private"  ),
    Action("Field",      "cell_edge_sizes",                 "read_only"     ),
    Action("Field",      "get_dim_filenames()",             "make_private"  ),
    Action("Field",      "collect_timeslices()",            "make_private"  ),
    Action("Field",      "reshape()",                       "make_private"  ),
    Action("Field",      "calc_cell_edge_sizes()",          "make_private"  ),
    Action("Field",      "search_indices_vertical_z()",     "make_private"  ),
    Action("Field",      "search_indices_vertical_s()",     "make_private"  ),
    Action("Field",      "reconnect_bnd_indices()",         "make_private"  ),
    Action("Field",      "search_indices_rectilinear()",    "make_private"  ),
    Action("Field",      "search_indices_curvilinear()",    "make_private"  ),
    Action("Field",      "search_indices()",                "make_private"  ),
    Action("Field",      "interpolator2D()",                "make_private"  ),
    Action("Field",      "interpolator3D()",                "make_private"  ),
    Action("Field",      "spatial_interpolation()",         "make_private"  ),
    Action("Field",      "time_index()",                    "make_private"  ),
    Action("Field",      "ccode_eval()",                    "make_private"  ),
    Action("Field",      "ccode_convert()",                 "make_private"  ),
    Action("Field",      "get_block_id()",                  "make_private"  ),
    Action("Field",      "get_block()",                     "make_private"  ),
    Action("Field",      "chunk_setup()",                   "make_private"  ),
    Action("Field",      "chunk_data()",                    "make_private"  ),
    Action("Field",      "rescale_and_set_minmax()",        "make_private"  ),
    Action("Field",      "data_concatenate()",              "make_private"  ),
    Action("FieldSet",   "completed",                       "make_private"  ),
    Action("FieldSet",   "particlefile",                    "read_only"     ),
    Action("FieldSet",   "add_UVfield()",                   "make_private"  ),
    Action("FieldSet",   "check_complete()",                "make_private"  ),
    Action("FieldSet",   "parse_wildcards()",               "make_private"  ),
]
# fmt: on

# Create test data dictionary
fieldset = create_fieldset_unit_mesh()
field = fieldset.U

test_data = {
    "Field": {
        "class": Field,
        "object": field,
    },
    "FieldSet": {
        "class": FieldSet,
        "object": fieldset,
    },
}


@pytest.mark.parametrize(
    "private_attribute_action",
    filter(lambda action: not action.is_method and action.type_ == "make_private", actions),
    ids=str,
)
def test_private_attrib(private_attribute_action: Action):
    """Checks that the public attribute is equivalent to the private attribute."""
    action = private_attribute_action

    obj = test_data[action.class_]["object"]

    with pytest.raises(DeprecationWarning):
        assert hasattr(obj, action.public_name)
        assert hasattr(obj, action.private_name)
        assert getattr(obj, action.public_name) is getattr(obj, action.private_name)


@pytest.mark.parametrize(
    "private_method_action",
    filter(lambda action: action.is_method and action.type_ == "make_private", actions),
    ids=str,
)
def test_private_method(private_method_action: Action):
    """Looks at the source code to ensure that `public_method` calls `private_method`.

    Looks for the string `.{method_name}(` in the source code of `public_method`.
    """
    action = private_method_action

    class_ = test_data[action.class_]["class"]

    public_method = getattr(class_, action.public_name)
    private_method = getattr(class_, action.private_name)

    assert callable(public_method)
    assert callable(private_method)
    assert f".{action.private_name}(" in inspect.getsource(public_method)


@pytest.mark.parametrize(
    "read_only_attribute_action",
    filter(lambda action: not action.is_method and action.type_ == "read_only", actions),
    ids=str,
)
def test_read_only_attr(read_only_attribute_action: Action):
    """Tries to store a variable in the read-only attribute."""
    action = read_only_attribute_action
    obj = test_data[action.class_]["object"]

    assert hasattr(obj, action.public_name)
    with pytest.raises(AttributeError):
        setattr(obj, action.public_name, None)
