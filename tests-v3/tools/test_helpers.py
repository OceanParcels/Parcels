from datetime import timedelta

import numpy as np
import pytest

import parcels.utils._helpers as helpers
from parcels.utils._decorators import deprecated, deprecated_made_private
from parcels.utils._helpers import timedelta_to_float


def test_format_list_items_multiline():
    expected = """[
    item1,
    item2,
    item3
]"""
    assert helpers._format_list_items_multiline(["item1", "item2", "item3"], 1) == expected
    assert helpers._format_list_items_multiline([], 1) == "[]"


def test_deprecated():
    class SomeClass:
        @deprecated()
        def some_method(self, x, y):
            return x + y

        @staticmethod
        @deprecated()
        def some_static_method(x, y):
            return x + y

        @property
        @deprecated()
        def some_property(self):
            return 2

    @deprecated()
    def some_function(x, y):
        return x + y

    with pytest.warns(DeprecationWarning) as record:
        SomeClass().some_method(1, 2)
    assert "SomeClass.some_method" in record[0].message.args[0]

    with pytest.warns(DeprecationWarning) as record:
        SomeClass.some_static_method(1, 2)
    assert "SomeClass.some_static_method" in record[0].message.args[0]

    with pytest.warns(DeprecationWarning) as record:
        _ = SomeClass().some_property
    assert "SomeClass.some_property" in record[0].message.args[0]

    with pytest.warns(DeprecationWarning) as record:
        some_function(1, 2)
    assert "some_function" in record[0].message.args[0]

    with pytest.warns(DeprecationWarning) as record:
        some_function(1, 2)
    assert "some_function" in record[0].message.args[0]


def test_deprecated_made_private():
    @deprecated_made_private
    def some_function(x, y):
        return x + y

    with pytest.warns(DeprecationWarning):
        some_function(1, 2)

    assert "deprecated::" in some_function.__doc__


@pytest.mark.parametrize(
    "input, expected",
    [
        (timedelta(days=1), 24 * 60 * 60),
        (np.timedelta64(1, "D"), 24 * 60 * 60),
        (3600.0, 3600.0),
    ],
)
def test_timedelta_to_float(input, expected):
    assert timedelta_to_float(input) == expected


def test_timedelta_to_float_exceptions():
    with pytest.raises((ValueError, TypeError)):
        timedelta_to_float("invalid_type")
