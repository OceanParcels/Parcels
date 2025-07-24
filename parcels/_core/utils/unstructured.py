DIM_TO_VERTICAL_LOCATION_MAP = {
    "nz1": "center",
    "nz": "face",
}


def get_vertical_location_from_dims(dims: tuple[str, ...]):
    """
    Determine the vertical location of the field based on the uxarray.UxDataArray object variables.

    Only used for unstructured grids.
    """
    vertical_dims_in_data = set(dims) & set(DIM_TO_VERTICAL_LOCATION_MAP.keys())

    if len(vertical_dims_in_data) != 1:
        raise ValueError(
            f"Expected exactly one vertical dimension ({set(DIM_TO_VERTICAL_LOCATION_MAP.keys())}) in the data, got {vertical_dims_in_data}"
        )

    return DIM_TO_VERTICAL_LOCATION_MAP[vertical_dims_in_data.pop()]


def get_vertical_dim_name_from_location(location: str):
    """Determine the vertical location of the field based on the uxarray.UxGrid object variables."""
    location_to_dim_map = {v: k for k, v in DIM_TO_VERTICAL_LOCATION_MAP.items()}
    return location_to_dim_map[location]
