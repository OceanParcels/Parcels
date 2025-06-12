from collections.abc import Hashable, Mapping
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from parcels import xgcm
from parcels.basegrid import BaseGrid
from parcels.tools.converters import TimeConverter

_AXIS_DIRECTION = Literal["X", "Y", "Z", "T"]
_AXIS_DIRECTION_SPATIAL = Literal["X", "Y", "Z"]
_AXIS_POSITION = Literal["center", "left", "right", "inner", "outer"]
_XGCM_AXES = Mapping[_AXIS_DIRECTION, xgcm.Axis]


def get_dimensionality(axis: xgcm.Axis | None) -> int:
    if axis is None:
        return 1
    first_coord = list(axis.coords.items())[0]
    pos, coord = first_coord

    pos_to_dim = {  # TODO: These could do with being explicitly tested
        "center": lambda x: x,
        "left": lambda x: x,
        "right": lambda x: x,
        "inner": lambda x: x + 1,
        "outer": lambda x: x - 1,
    }

    n = axis._ds[coord].size
    return pos_to_dim[pos](n)


def get_time(axis: xgcm.Axis) -> npt.NDArray:
    return axis._ds[axis.coords["center"]].values


class XGrid(BaseGrid):
    """
    Class to represent a structured grid in Parcels. Wraps a xgcm-like Grid object (we use a trimmed down version of the xgcm.Grid class that is vendored with Parcels).

    This class provides methods and properties required for indexing and interpolating on the grid.
    """

    def __init__(self, grid: xgcm.Grid, mesh="flat"):
        self.xgcm_grid = grid
        self.mesh = mesh
        ds = grid._ds
        assert_valid_lon_lat(ds["lon"], ds["lat"], grid.axes)

        # ! Not ideal... Triggers computation on a throwaway item. Keeping for now for v3 compat, will be removed in v4.
        self.lonlat_minmax = np.array(
            [
                np.nanmin(self.xgcm_grid._ds["lon"]),
                np.nanmax(self.xgcm_grid._ds["lon"]),
                np.nanmin(self.xgcm_grid._ds["lat"]),
                np.nanmax(self.xgcm_grid._ds["lat"]),
            ]
        )

    @property
    def lon(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.xgcm_grid.axes["X"]
        except KeyError:
            return np.zeros(1)
        return self.xgcm_grid._ds["lon"].values

    @property
    def lat(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.xgcm_grid.axes["Y"]
        except KeyError:
            return np.zeros(1)
        return self.xgcm_grid._ds["lat"].values

    @property
    def depth(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.xgcm_grid.axes["Z"]
        except KeyError:
            return np.zeros(1)
        return self.xgcm_grid._ds["depth"].values

    @property
    def _datetimes(self):
        try:
            axis = self.xgcm_grid.axes["T"]
        except KeyError:
            return np.zeros(1)
        return get_time(axis)

    @property
    def time(self):
        return self._datetimes.astype(np.float64) / 1e9

    @property
    def xdim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("X"))

    @property
    def ydim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("Y"))

    @property
    def zdim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("Z"))

    @property
    def tdim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("T"))

    @property
    def time_origin(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        return TimeConverter(self._datetimes[0])

    @property
    def _z4d(self) -> Literal[0, 1]:
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        return 1 if self.depth.shape == 4 else 0

    @property
    def zonal_periodic(self): ...  # ? hmmm, from v3, do we still need this?

    @property
    def _gtype(self):
        """This class is created *purely* for compatibility with v3 code and will be removed
        or changed in future.

        TODO: Remove
        """
        from parcels.grid import GridType

        if len(self.lon.shape) <= 1:
            if self.depth is None or len(self.depth.shape) <= 1:
                return GridType.RectilinearZGrid
            else:
                return GridType.RectilinearSGrid
        else:
            if self.depth is None or len(self.depth.shape) <= 1:
                return GridType.CurvilinearZGrid
            else:
                return GridType.CurvilinearSGrid

    def search(self, z, y, x, ei=None, search2D=False): ...


def get_axis_from_dim_name(axes: _XGCM_AXES, dim: str) -> _AXIS_DIRECTION | None:
    """For a given dimension name in a grid, returns the direction axis it is on."""
    for axis_name, axis in axes.items():
        if dim in axis.coords.values():
            return axis_name
    return None


def get_position_from_dim_name(axes: _XGCM_AXES, dim: str) -> _AXIS_POSITION | None:
    """For a given dimension, returns the position of the variable in the grid."""
    for axis in axes.values():
        var_to_position = {var: position for position, var in axis.coords.items()}

        if dim in var_to_position:
            return var_to_position[dim]
    return None


def assert_all_dimensions_correspond_with_axis(da: xr.DataArray, axes: _XGCM_AXES) -> None:
    dim_to_axis = {dim: get_axis_from_dim_name(axes, dim) for dim in da.dims}

    for dim, direction in dim_to_axis.items():
        if direction is None:
            raise ValueError(
                f"Dimension {dim!r} for DataArray {da.name!r} with dims {da.dims} is not associated with a direction on the provided grid."
            )


def assert_valid_field_array(da: xr.DataArray, axes: _XGCM_AXES):
    """
    Asserts that for a data array:
    - All dimensions are associated with a direction on the grid
    - These directions are T, Z, Y, X and the array is ordered as T, Z, Y, X
    """
    assert_all_dimensions_correspond_with_axis(da, axes)

    dim_to_axis = {dim: get_axis_from_dim_name(axes, dim) for dim in da.dims}
    dim_to_axis = cast(dict[Hashable, _AXIS_DIRECTION], dim_to_axis)

    # Assert all dimensions are present
    if set(dim_to_axis.values()) != {"T", "Z", "Y", "X"}:
        raise ValueError(
            f"DataArray {da.name!r} with dims {da.dims} has directions {tuple(dim_to_axis.values())}."
            "Expected directions of 'T', 'Z', 'Y', and 'X'."
        )

    # Assert order is t, z, y, x
    if list(dim_to_axis.values()) != ["T", "Z", "Y", "X"]:
        raise ValueError(
            f"Dimension order for array {da.name!r} is not valid. Got {tuple(dim_to_axis.keys())} with associated directions of {tuple(dim_to_axis.values())}.  Expected directions of ('T', 'Z', 'Y', 'X'). Transpose your array accordingly."
        )


def assert_valid_lon_lat(da_lon, da_lat, axes: _XGCM_AXES):
    """
    Asserts that the provided longitude and latitude DataArrays are defined appropriately
    on the F points to match the internal representation in Parcels.

    - Longitude and latitude must be 1D or 2D (both must have the same dimensionality)
    - Both are defined on the left points (i.e., not the centers)
    - If 1D:
      - Longitude is associated with the X axis
      - Latitude is associated with the Y axis
    - If 2D:
      - Lon and lat are defined on the same dimensions
      - Lon and lat are transposed such they're Y, X
    """
    assert_all_dimensions_correspond_with_axis(da_lon, axes)
    assert_all_dimensions_correspond_with_axis(da_lat, axes)

    dim_to_position = {dim: get_position_from_dim_name(axes, dim) for dim in da_lon.dims}
    dim_to_position.update({dim: get_position_from_dim_name(axes, dim) for dim in da_lat.dims})

    for dim in da_lon.dims:
        if get_position_from_dim_name(axes, dim) == "center":
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} is defined on the center of the grid, but must be defined on the F points."
            )
    for dim in da_lat.dims:
        if get_position_from_dim_name(axes, dim) == "center":
            raise ValueError(
                f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} is defined on the center of the grid, but must be defined on the F points."
            )

    if da_lon.ndim != da_lat.ndim:
        raise ValueError(
            f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} have different dimensionalities."
        )
    if da_lon.ndim not in (1, 2):
        raise ValueError(
            f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be 1D or 2D."
        )

    if da_lon.ndim == 1:
        if get_axis_from_dim_name(axes, da_lon.dims[0]) != "X":
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} is not associated with the X axis."
            )
        if get_axis_from_dim_name(axes, da_lat.dims[0]) != "Y":
            raise ValueError(
                f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} is not associated with the Y axis."
            )

    if da_lon.ndim == 2:
        if da_lon.dims != da_lat.dims:
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be defined on the same dimensions."
            )

        lon_axes = [get_axis_from_dim_name(axes, dim) for dim in da_lon.dims]
        if lon_axes != ["Y", "X"]:
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be defined on the X and Y axes and transposed to have dimensions in order of Y, X."
            )
