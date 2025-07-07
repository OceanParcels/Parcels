import re
from datetime import timedelta
from typing import Any

import numpy as np

import parcels
from parcels import Grid, ParticleFile, Variable
from parcels.grid import RectilinearGrid
from tests.utils import create_fieldset_unit_mesh, create_simple_pset


def assert_simple_repr(class_: type, kwargs: dict[str, Any]):
    """Test that the repr of an object contains all the arguments. This only works for objects where the repr matches the calling signature."""
    obj = class_(**kwargs)
    obj_repr = repr(obj)

    for param in kwargs.keys():
        assert param in obj_repr
        # skip `assert repr(value) in obj_repr` as this is not always true if init does processing on the value
    assert class_.__name__ in obj_repr


def valid_indentation(s: str) -> bool:
    """Make sure that all lines in string is indented with a multiple of 4 spaces."""
    if s.startswith(" "):
        return False

    lines = s.split("\n")
    for line in lines:
        line = re.sub("^( {4})+", "", line)
        if line.startswith(" "):
            return False
    return True


def test_check_indentation():
    valid = """
test
    test
test
    test
        test
    test"""
    assert valid_indentation(valid)
    invalid = """
test
    test
 invalid!
"""
    assert not valid_indentation(invalid)


def test_particletype_repr():
    kwargs = dict(pclass=parcels.Particle)
    assert_simple_repr(parcels.particle.ParticleType, kwargs)


def test_grid_repr():
    """Test arguments are in the repr of a Grid object"""
    kwargs = dict(lon=np.array([1, 2, 3]), lat=np.array([4, 5, 6]), time=None, mesh="spherical")
    assert_simple_repr(Grid, kwargs)


def test_variable_repr():
    """Test arguments are in the repr of the Variable object."""
    kwargs = dict(name="test", dtype=np.float32, initial=0, to_write=False)
    assert_simple_repr(Variable, kwargs)


def test_rectilineargrid_repr():
    """
    Test arguments are in the repr of a RectilinearGrid object.

    Mainly to test inherited repr is correct.
    """
    kwargs = dict(lon=np.array([1, 2, 3]), lat=np.array([4, 5, 6]), time=None, mesh="spherical")
    assert_simple_repr(RectilinearGrid, kwargs)


def test_particlefile_repr():
    pset = create_simple_pset()
    kwargs = dict(
        name="file.zarr", particleset=pset, outputdt=timedelta(hours=1), chunks=None, create_new_zarrfile=False
    )
    assert_simple_repr(ParticleFile, kwargs)


def test_field_repr():
    field = create_fieldset_unit_mesh().U
    assert valid_indentation(repr(field))


def test_vectorfield_repr():
    field = create_fieldset_unit_mesh().UV
    assert isinstance(field, parcels.VectorField)
    assert valid_indentation(repr(field))


def test_fieldset_repr():
    fieldset = create_fieldset_unit_mesh()
    assert valid_indentation(repr(fieldset))


def test_particleset_repr():
    pset = create_simple_pset()
    valid_indentation(repr(pset))

    pset = create_simple_pset(n=15)
    valid_indentation(repr(pset))


def capture(s):
    with open("file.txt", "a") as f:
        f.write(s)
