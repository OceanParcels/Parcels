from collections import OrderedDict

# Representation of axis shifts
axis_shift_left = -0.5
axis_shift_right = 0.5
axis_shift_center = 0
# Characterizes valid shifts only
valid_axis_shifts = [axis_shift_left, axis_shift_right, axis_shift_center]


def assert_valid_comodo(ds):
    """Verify that the dataset meets comodo conventions

    Parameters
    ----------
    ds : xarray.dataset
    """
    # TODO: implement
    assert True


def get_all_axes(ds):
    axes = set()
    for d in ds.dims:
        if "axis" in ds[d].attrs:
            axes.add(ds[d].attrs["axis"])
    return axes


def get_axis_coords(ds, axis_name):
    """Find the name of the coordinates associated with a comodo axis.

    Parameters
    ----------
    ds : xarray.dataset or xarray.dataarray
    axis_name : str
        The name of the axis to find (e.g. 'X')

    Returns
    -------
    coord_name : list
        The names of the coordinate matching that axis
    """
    coord_names = []
    for d in ds.dims:
        axis = ds[d].attrs.get("axis")
        if axis == axis_name:
            coord_names.append(d)
    return coord_names


def get_axis_positions_and_coords(ds, axis_name):
    coord_names = get_axis_coords(ds, axis_name)
    ncoords = len(coord_names)
    if ncoords == 0:
        # didn't find anything for this axis
        raise ValueError(f"Couldn't find any coordinates for axis {axis_name}")

    # now figure out what type of coordinates these are:
    # center, left, right, or outer
    coords = {name: ds[name] for name in coord_names}

    # some tortured logic for dealing with malformed c_grid_axis_shift
    # attributes such as produced by old versions of xmitgcm.
    # This should be a float (either -0.5 or 0.5)
    # this function returns that, or True of the attribute is set to
    # anything at all
    def _maybe_fix_type(attr):
        if attr is not None:
            try:
                return float(attr)
            except TypeError:
                return True

    axis_shift = {name: _maybe_fix_type(coord.attrs.get("c_grid_axis_shift")) for name, coord in coords.items()}
    coord_len = {name: len(coord) for name, coord in coords.items()}

    # look for the center coord, which is required
    # this list will potentially contain "center", "inner", and "outer" points
    coords_without_axis_shift = {name: coord_len[name] for name, shift in axis_shift.items() if not shift}
    if len(coords_without_axis_shift) == 0:
        raise ValueError(f"Couldn't find a center coordinate for axis {axis_name}")
    elif len(coords_without_axis_shift) > 1:
        raise ValueError(f"Found two coordinates without `c_grid_axis_shift` attribute for axis {axis_name}")
    center_coord_name = list(coords_without_axis_shift)[0]
    # the length of the center coord is key to decoding the other coords
    axis_len = coord_len[center_coord_name]

    # now we can start filling in the information about the different coords
    axis_coords = OrderedDict()
    axis_coords["center"] = center_coord_name

    # now check the other coords
    coord_names.remove(center_coord_name)
    for name in coord_names:
        shift = axis_shift[name]
        clen = coord_len[name]
        if clen == axis_len + 1:
            axis_coords["outer"] = name
        elif clen == axis_len - 1:
            axis_coords["inner"] = name
        elif shift == axis_shift_left:
            if clen == axis_len:
                axis_coords["left"] = name
            else:
                raise ValueError(f"Left coordinate {name} has incompatible length {clen:g} (axis_len={axis_len:g})")
        elif shift == axis_shift_right:
            if clen == axis_len:
                axis_coords["right"] = name
            else:
                raise ValueError(f"Right coordinate {name} has incompatible length {clen:g} (axis_len={axis_len:g})")
        else:
            if shift not in valid_axis_shifts:
                # string representing valid axis shifts
                valids = str(valid_axis_shifts)[1:-1]

                raise ValueError(
                    f"Coordinate {name} has invalid "
                    f"`c_grid_axis_shift` attribute `{shift!r}`. "
                    f"`c_grid_axis_shift` must be one of: {valids}"
                )
            else:
                raise ValueError(f"Coordinate {name} has missing `c_grid_axis_shift` attribute `{shift!r}`")
    return axis_coords


def _assert_data_on_grid(da):
    pass
