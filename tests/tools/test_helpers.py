import pytest

from parcels.tools._helpers import deprecated, deprecated_made_private


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
