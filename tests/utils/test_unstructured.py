import pytest

from parcels._core.utils.unstructured import (
    get_vertical_dim_name_from_location,
    get_vertical_location_from_dims,
)


def test_get_vertical_location_from_dims():
    # Test with nz1 dimension
    assert get_vertical_location_from_dims(("nz1", "time")) == "center"

    # Test with nz dimension
    assert get_vertical_location_from_dims(("nz", "time")) == "face"

    # Test with both dimensions
    with pytest.raises(ValueError):
        get_vertical_location_from_dims(("nz1", "nz", "time"))

    # Test with no vertical dimension
    with pytest.raises(ValueError):
        get_vertical_location_from_dims(("time", "x", "y"))


def test_get_vertical_dim_name_from_location():
    # Test with center location
    assert get_vertical_dim_name_from_location("center") == "nz1"

    # Test with face location
    assert get_vertical_dim_name_from_location("face") == "nz"

    with pytest.raises(KeyError):
        get_vertical_dim_name_from_location("invalid_location")
