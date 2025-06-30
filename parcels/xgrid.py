from collections.abc import Hashable, Mapping
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from parcels import xgcm
from parcels._index_search import _search_indices_curvilinear_2d
from parcels.basegrid import BaseGrid
from parcels.tools.converters import TimeConverter

_XGCM_AXIS_DIRECTION = Literal["X", "Y", "Z", "T"]
_XGCM_AXIS_POSITION = Literal["center", "left", "right", "inner", "outer"]
_AXIS_DIRECTION = Literal["X", "Y", "Z"]
_XGCM_AXES = Mapping[_XGCM_AXIS_DIRECTION, xgcm.Axis]


def get_n_cell_edges_along_dim(axis: xgcm.Axis | None) -> int:
    if axis is None:
        return 1
    first_coord = list(axis.coords.items())[0]
    _, coord_var = first_coord

    return axis._ds[coord_var].size


def get_time(axis: xgcm.Axis) -> npt.NDArray:
    return axis._ds[axis.coords["center"]].values


class XGrid(BaseGrid):
    """
    Class to represent a structured grid in Parcels. Wraps a xgcm-like Grid object (we use a trimmed down version of the xgcm.Grid class that is vendored with Parcels).

    This class provides methods and properties required for indexing and interpolating on the grid.

    Assumptions:
    - If using Parcels in the context of a periodic simulation, the provided grid already has a halo

    """

    def __init__(self, grid: xgcm.Grid, mesh="flat"):
        self.xgcm_grid = grid
        self.mesh = mesh
        ds = grid._ds
        assert_valid_lat_lon(ds["lat"], ds["lon"], grid.axes)

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
        return get_n_cell_edges_along_dim(self.xgcm_grid.axes.get("X"))

    @property
    def ydim(self):
        return get_n_cell_edges_along_dim(self.xgcm_grid.axes.get("Y"))

    @property
    def zdim(self):
        return get_n_cell_edges_along_dim(self.xgcm_grid.axes.get("Z"))

    @property
    def tdim(self):
        return get_n_cell_edges_along_dim(self.xgcm_grid.axes.get("T"))

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

    def search(self, z, y, x, ei=None):
        ds = self.xgcm_grid._ds

        zi, zeta = _search_1d_array(ds.depth.values, z)

        if ds.lon.ndim == 1:
            yi, eta = _search_1d_array(ds.lat.values, y)
            xi, xsi = _search_1d_array(ds.lon.values, x)
            return {"X": (xi, xsi), "Y": (yi, eta), "Z": (zi, zeta)}

        yi, xi = None, None
        if ei is not None:
            axis_indices = self.unravel_index(ei)
            xi = axis_indices.get("X")
            yi = axis_indices.get("Y")

        if ds.lon.ndim == 2:
            eta, xsi, yi, xi = _search_indices_curvilinear_2d(self, y, x, yi, xi)

            return {"X": (xi, xsi), "Y": (yi, eta), "Z": (zi, zeta)}

        raise NotImplementedError("Searching in >2D lon/lat arrays is not implemented yet.")

    def ravel_index(self, axis_indices: dict[_AXIS_DIRECTION, int]) -> int:
        xi = axis_indices.get("X", 0)
        yi = axis_indices.get("Y", 0)
        zi = axis_indices.get("Z", 0)
        return xi + self.xdim * yi + self.xdim * self.ydim * zi

    def unravel_index(self, ei) -> dict[_AXIS_DIRECTION, int]:
        zi = ei // (self.xdim * self.ydim)
        ei = ei % (self.xdim * self.ydim)

        yi = ei // self.xdim
        xi = ei % self.xdim
        return {
            "X": xi,
            "Y": yi,
            "Z": zi,
        }


def get_axis_from_dim_name(axes: _XGCM_AXES, dim: str) -> _XGCM_AXIS_DIRECTION | None:
    """For a given dimension name in a grid, returns the direction axis it is on."""
    for axis_name, axis in axes.items():
        if dim in axis.coords.values():
            return axis_name
    return None


def get_position_from_dim_name(axes: _XGCM_AXES, dim: str) -> _XGCM_AXIS_POSITION | None:
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
    dim_to_axis = cast(dict[Hashable, _XGCM_AXIS_DIRECTION], dim_to_axis)

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


def assert_valid_lat_lon(da_lat, da_lon, axes: _XGCM_AXES):
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

        if not np.all(np.diff(da_lon.values) > 0):
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} must be strictly increasing."
            )
        if not np.all(np.diff(da_lat.values) > 0):
            raise ValueError(f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be strictly increasing.")

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


def _search_1d_array(
    arr: np.array,
    x: float,
) -> tuple[int, int]:
    """
    Searches for the particle location in a 1D array and returns barycentric coordinate along dimension.

    Assumptions:
    - particle position x is within the bounds of the array
    - array is strictly monotonically increasing.

    Parameters
    ----------
    arr : np.array
        1D array to search in.
    x : float
        Position in the 1D array to search for.

    Returns
    -------
    int
        Index of the element just before the position x in the array.
    float
        Barycentric coordinate.
    """
    i = np.argmin(arr <= x) - 1
    bcoord = (x - arr[i]) / (arr[i + 1] - arr[i])
    return i, bcoord
