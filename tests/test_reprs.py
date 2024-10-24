from typing import Any

import numpy as np

from parcels import Grid, TimeConverter, Variable
from parcels.grid import RectilinearGrid


def validate_simple_repr(class_: type, kwargs: dict[str, Any]):
    """Test that the repr of an object contains all the arguments. This only works for objects where the repr matches the calling signature."""
    obj = class_(**kwargs)
    obj_repr = repr(obj)

    for param in kwargs.keys():
        assert param in obj_repr
        # skip `assert repr(value) in obj_repr` as this is not always true if init does processing on the value
    assert class_.__name__ in obj_repr


def test_grid_repr():
    """Test arguments are in the repr of a Grid object"""
    kwargs = dict(
        lon=np.array([1, 2, 3]), lat=np.array([4, 5, 6]), time=None, time_origin=TimeConverter(), mesh="spherical"
    )
    validate_simple_repr(Grid, kwargs)


def test_variable_repr():
    """Test arguments are in the repr of the Variable object."""
    kwargs = dict(name="test", dtype=np.float32, initial=0, to_write=False)
    validate_simple_repr(Variable, kwargs)


def test_rectilineargrid_repr():
    """
    Test arguments are in the repr of a RectilinearGrid object.

    Mainly to test inherited repr is correct.
    """
    kwargs = dict(
        lon=np.array([1, 2, 3]), lat=np.array([4, 5, 6]), time=None, time_origin=TimeConverter(), mesh="spherical"
    )
    validate_simple_repr(RectilinearGrid, kwargs)
