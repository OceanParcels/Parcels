import pytest

from parcels._typing import (
    assert_valid_gridindexingtype,
    assert_valid_interp_method,
    assert_valid_mesh,
)

validators = (
    assert_valid_interp_method,
    assert_valid_mesh,
    assert_valid_gridindexingtype,
)


@pytest.mark.parametrize("validator", validators)
def test_invalid_option(validator):
    with pytest.raises(ValueError):
        validator("invalid option")


validation_mapping = [
    (assert_valid_interp_method, "nearest"),
    (assert_valid_mesh, "spherical"),
    (assert_valid_gridindexingtype, "pop"),
]


@pytest.mark.parametrize("validator, value", validation_mapping)
def test_valid_option(validator, value):
    validator(value)
