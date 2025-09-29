from collections.abc import Hashable, Mapping, Sequence
from functools import cached_property
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
import xgcm

from parcels._core.basegrid import BaseGrid
from parcels._core.index_search import _search_1d_array, _search_indices_curvilinear_2d
from parcels._typing import assert_valid_mesh

_XGRID_AXES = Literal["X", "Y", "Z"]
_XGRID_AXES_ORDERING: Sequence[_XGRID_AXES] = "ZYX"

_XGCM_AXIS_DIRECTION = Literal["X", "Y", "Z", "T"]
_XGCM_AXIS_POSITION = Literal["center", "left", "right", "inner", "outer"]
_XGCM_AXES = Mapping[_XGCM_AXIS_DIRECTION, xgcm.Axis]

_FIELD_DATA_ORDERING: Sequence[_XGCM_AXIS_DIRECTION] = "TZYX"

_DEFAULT_XGCM_KWARGS = {"periodic": False}


def get_cell_count_along_dim(ds: xr.Dataset, axis: xgcm.Axis) -> int:
    first_coord = list(axis.coords.items())[0]
    _, coord_var = first_coord

    return ds[coord_var].size - 1


def get_time(ds: xr.Dataset, axis: xgcm.Axis) -> npt.NDArray:
    return ds[axis.coords["center"]].values


def _get_xgrid_axes(grid: xgcm.Grid) -> list[_XGRID_AXES]:
    spatial_axes = [a for a in grid.axes.keys() if a in ["X", "Y", "Z"]]
    return sorted(spatial_axes, key=_XGRID_AXES_ORDERING.index)


def _drop_field_data(ds: xr.Dataset) -> xr.Dataset:
    """
    Removes DataArrays from the dataset that are associated with field data so that
    when passed to the XGCM grid, the object only functions as an in memory representation
    of the grid.
    """
    return ds.drop_vars(ds.data_vars)


def _transpose_xfield_data_to_tzyx(da: xr.DataArray, xgcm_grid: xgcm.Grid) -> xr.DataArray:
    """
    Transpose a DataArray of any shape into a 4D array of order TZYX. Uses xgcm to determine
    the axes, and inserts mock dimensions of size 1 for any axes not present in the DataArray.
    """
    ax_dims = [(get_axis_from_dim_name(xgcm_grid.axes, dim), dim) for dim in da.dims]

    if all(ax_dim[0] is None for ax_dim in ax_dims):
        # Assuming its a 1D constant field (hence has no axes)
        assert da.shape == (1, 1, 1, 1)
        return da.rename({old_dim: f"mock{axis}" for old_dim, axis in zip(da.dims, _FIELD_DATA_ORDERING, strict=True)})

    # All dimensions must be associated with an axis in the grid
    if any(ax_dim[0] is None for ax_dim in ax_dims):
        raise ValueError(
            f"DataArray {da.name!r} with dims {da.dims} has dimensions that are not associated with a direction on the provided grid."
        )

    axes_not_in_field = set(_FIELD_DATA_ORDERING) - set(ax_dim[0] for ax_dim in ax_dims)

    mock_dims_to_create = {}
    for ax in axes_not_in_field:
        mock_dims_to_create[f"mock{ax}"] = 1
        ax_dims.append((ax, f"mock{ax}"))

    if mock_dims_to_create:
        da = da.expand_dims(mock_dims_to_create, create_index_for_new_dim=False)

    ax_dims = sorted(ax_dims, key=lambda x: _FIELD_DATA_ORDERING.index(x[0]))

    return da.transpose(*[ax_dim[1] for ax_dim in ax_dims])


class XGrid(BaseGrid):
    """
    Class to represent a structured grid in Parcels. Wraps a xgcm-like Grid object (we use a trimmed down version of the xgcm.Grid class that is vendored with Parcels).

    This class provides methods and properties required for indexing and interpolating on the grid.

    Assumptions:
    - If using Parcels in the context of a spatially periodic simulation, the provided grid already has a halo

    """

    def __init__(self, grid: xgcm.Grid, mesh="flat"):
        self.xgcm_grid = grid
        self._mesh = mesh
        self._spatialhash = None
        ds = grid._ds

        # Set the coordinates for the dataset (needed to be done explicitly for curvilinear grids)
        if "lon" in ds:
            ds.set_coords("lon")
        if "lat" in ds:
            ds.set_coords("lat")

        if len(set(grid.axes) & {"X", "Y", "Z"}) > 0:  # Only if spatial grid is >0D (see #2054 for further development)
            assert_valid_lat_lon(ds["lat"], ds["lon"], grid.axes)

        assert_valid_mesh(mesh)
        self._ds = ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, mesh="flat", xgcm_kwargs=None):
        """WARNING: unstable API, subject to change in future versions."""  # TODO v4: make private or remove warning on v4 release
        if xgcm_kwargs is None:
            xgcm_kwargs = {}

        xgcm_kwargs = {**_DEFAULT_XGCM_KWARGS, **xgcm_kwargs}

        ds = _drop_field_data(ds)
        grid = xgcm.Grid(ds, **xgcm_kwargs)
        return cls(grid, mesh=mesh)

    @property
    def axes(self) -> list[_XGRID_AXES]:
        return _get_xgrid_axes(self.xgcm_grid)

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
        return self._ds["lon"].values

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
        return self._ds["lat"].values

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
        return self._ds["depth"].values

    @property
    def _datetimes(self):
        try:
            axis = self.xgcm_grid.axes["T"]
        except KeyError:
            return np.zeros(1)
        return get_time(self._ds, axis)

    @property
    def time(self):
        return self._datetimes.astype(np.float64) / 1e9

    @cached_property
    def xdim(self) -> int:
        return self.get_axis_dim("X")

    @cached_property
    def ydim(self) -> int:
        return self.get_axis_dim("Y")

    @cached_property
    def zdim(self) -> int:
        return self.get_axis_dim("Z")

    def get_axis_dim(self, axis: _XGRID_AXES) -> int:
        if axis not in self.axes:
            raise ValueError(f"Axis {axis!r} is not part of this grid. Available axes: {self.axes}")

        return get_cell_count_along_dim(self._ds, self.xgcm_grid.axes[axis])

    def localize(self, position: dict[_XGRID_AXES, tuple[int, float]], dims: list[str]) -> dict[str, tuple[int, float]]:
        """
        Uses the grid context (i.e., the staggering of the grid) to convert a position relative
        to the F-points in the grid to a position relative to the staggered grid the array
        of interest is defined on.

        Uses dimensions of the DataArray to determine the staggered grid.

        WARNING: This API is unstable and subject to change in future versions.

        Parameters
        ----------
        position : dict
            A mapping of the axis to a tuple of (index, barycentric coordinate) for the
            F-points in the grid.
        dims : list[str]
            A list of dimension names that the DataArray is defined on. This is used to determine
            the staggering of the grid and which axis each dimension corresponds to.

        Returns
        -------
        dict[str, tuple[int, float]]
            A mapping of the dimension names to a tuple of (index, barycentric coordinate) for
            the staggered grid the DataArray is defined on.

        Example
        -------
        >>> position = {'X': (5, 0.51), 'Y': (
            10, 0.25), 'Z': (3, 0.75)}
        >>> dims = ['time', 'depth', 'YC', 'XC']
        >>> grid.localize(position, dims)
        {'depth': (3, 0.75), 'YC': (9, 0.75), 'XC': (5, 0.01)}
        """
        axis_to_var = {get_axis_from_dim_name(self.xgcm_grid.axes, dim): dim for dim in dims}
        var_positions = {
            axis: get_xgcm_position_from_dim_name(self.xgcm_grid.axes, dim) for axis, dim in axis_to_var.items()
        }
        return {
            axis_to_var[axis]: _convert_center_pos_to_fpoint(
                index=index,
                bcoord=bcoord,
                xgcm_position=var_positions[axis],
                f_points_xgcm_position=self._fpoint_info[axis],
            )
            for axis, (index, bcoord) in position.items()
        }

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
        ds = self._ds

        if "Z" in self.axes:
            zi, zeta = _search_1d_array(ds.depth.values, z)
        else:
            zi, zeta = np.zeros(z.shape, dtype=int), np.zeros(z.shape, dtype=float)

        if ds.lon.ndim == 1:
            yi, eta = _search_1d_array(ds.lat.values, y)
            xi, xsi = _search_1d_array(ds.lon.values, x)
            return {"Z": (zi, zeta), "Y": (yi, eta), "X": (xi, xsi)}

        yi, xi = None, None
        if ei is not None:
            axis_indices = self.unravel_index(ei)
            xi = axis_indices.get("X")
            yi = axis_indices.get("Y")

        if ds.lon.ndim == 2:
            yi, eta, xi, xsi = _search_indices_curvilinear_2d(self, y, x, yi, xi)

            return {"Z": (zi, zeta), "Y": (yi, eta), "X": (xi, xsi)}

        raise NotImplementedError("Searching in >2D lon/lat arrays is not implemented yet.")

    @cached_property
    def _fpoint_info(self):
        """Returns a mapping of the spatial axes in the Grid to their XGCM positions."""
        xgcm_axes = self.xgcm_grid.axes
        f_point_positions = ["left", "right", "inner", "outer"]
        axis_position_mapping = {}
        for axis in self.axes:
            coords = xgcm_axes[axis].coords
            edge_positions = [pos for pos in coords.keys() if pos in f_point_positions]
            assert len(edge_positions) == 1, f"Axis {axis} has multiple edge positions: {edge_positions}"
            axis_position_mapping[axis] = edge_positions[0]

        return axis_position_mapping

    def get_axis_dim_mapping(self, dims: list[str]) -> dict[_XGRID_AXES, str]:
        """
        Maps xarray dimension names to their corresponding axis (X, Y, Z).

        WARNING: This API is unstable and subject to change in future versions.

        Parameters
        ----------
        dims : list[str]
            List of xarray dimension names

        Returns
        -------
        dict[_XGRID_AXES, str]
            Dictionary mapping axes (X, Y, Z) to their corresponding dimension names

        Examples
        --------
        >>> grid.get_axis_dim_mapping(['time', 'lat', 'lon'])
        {'Y': 'lat', 'X': 'lon'}

        Notes
        -----
        Only returns mappings for spatial axes (X, Y, Z) that are present in the grid.
        """
        result = {}
        for dim in dims:
            axis = get_axis_from_dim_name(self.xgcm_grid.axes, dim)
            if axis in self.axes:  # Only include spatial axes (X, Y, Z)
                result[cast(_XGRID_AXES, axis)] = dim
        return result


def get_axis_from_dim_name(axes: _XGCM_AXES, dim: str) -> _XGCM_AXIS_DIRECTION | None:
    """For a given dimension name in a grid, returns the direction axis it is on."""
    for axis_name, axis in axes.items():
        if dim in axis.coords.values():
            return axis_name
    return None


def get_xgcm_position_from_dim_name(axes: _XGCM_AXES, dim: str) -> _XGCM_AXIS_POSITION | None:
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

    dim_to_position = {dim: get_xgcm_position_from_dim_name(axes, dim) for dim in da_lon.dims}
    dim_to_position.update({dim: get_xgcm_position_from_dim_name(axes, dim) for dim in da_lat.dims})

    for dim in da_lon.dims:
        if get_xgcm_position_from_dim_name(axes, dim) == "center":
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} is defined on the center of the grid, but must be defined on the F points."
            )
    for dim in da_lat.dims:
        if get_xgcm_position_from_dim_name(axes, dim) == "center":
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


def _convert_center_pos_to_fpoint(
    *, index: int, bcoord: float, xgcm_position: _XGCM_AXIS_POSITION, f_points_xgcm_position: _XGCM_AXIS_POSITION
) -> tuple[int, float]:
    """Converts a physical position relative to the cell edges defined in the grid to be relative to the center point.

    This is used to "localize" a position to be relative to the staggered grid at which the field is defined, so that
    it can be easily interpolated.

    This also handles different model input cell edges and centers are staggered in different directions (e.g., with NEMO and MITgcm).
    """
    if xgcm_position != "center":  # Data is already defined on the F points
        return index, bcoord

    bcoord = bcoord - 0.5
    if bcoord < 0:
        bcoord += 1.0
        index -= 1

    # Correct relative to the f-point position
    if f_points_xgcm_position in ["inner", "right"]:
        index += 1

    return index, bcoord
