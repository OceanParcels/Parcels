from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from datetime import datetime
from enum import IntEnum

import numpy as np
import uxarray as ux
import xarray as xr

from parcels._core.utils.time import TimeInterval
from parcels._core.utils.unstructured import get_vertical_location_from_dims
from parcels._reprs import default_repr, field_repr
from parcels._typing import (
    Mesh,
    VectorType,
    assert_valid_mesh,
)
from parcels.tools.converters import (
    UnitConverter,
    unitconverters_map,
)
from parcels.tools.statuscodes import (
    AllParcelsErrorCodes,
    FieldOutOfBoundError,
    FieldOutOfBoundSurfaceError,
    FieldSamplingError,
    _raise_field_out_of_bound_error,
)
from parcels.uxgrid import UxGrid, ensure_uxgrid
from parcels.v4.grid import Grid
from parcels.v4.gridadapter import GridAdapter

from ._index_search import _search_time_index

__all__ = ["Field", "GridType", "VectorField"]


class GridType(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


def _isParticle(key):
    if hasattr(key, "obs_written"):
        return True
    else:
        return False


def _deal_with_errors(error, key, vector_type: VectorType):
    if _isParticle(key):
        key.state = AllParcelsErrorCodes[type(error)]
    elif _isParticle(key[-1]):
        key[-1].state = AllParcelsErrorCodes[type(error)]
    else:
        raise RuntimeError(f"{error}. Error could not be handled because particle was not part of the Field Sampling.")

    if vector_type and "3D" in vector_type:
        return (0, 0, 0)
    elif vector_type == "2D":
        return (0, 0)
    else:
        return 0


class Field:
    """The Field class that holds scalar field data.
    The `Field` object is a wrapper around a xarray.DataArray or uxarray.UxDataArray object.
    Additionally, it holds a dynamic Callable procedure that is used to interpolate the field data.
    During initialization, the user can supply a custom interpolation method that is used to interpolate the field data,
    so long as the interpolation method has the correct signature.

    Notes
    -----
    The xarray.DataArray or uxarray.UxDataArray object contains the field data and metadata.
        * dims: (time, [nz1 | nz], [face_lat | node_lat | edge_lat], [face_lon | node_lon | edge_lon])
        * attrs: (location, mesh, mesh_type)

    When using a xarray.DataArray object,
    * The xarray.DataArray object must have the "location" and "mesh" attributes set.
    * The "location" attribute must be set to one of the following to define which pairing of points a field is associated with.
       * "node"
       * "face"
       * "x_edge"
       * "y_edge"
    * For an A-Grid, the "location" attribute must be set to / is assumed to be "node" (node_lat,node_lon).
    * For a C-Grid, the "location" setting for a field has the following interpretation:
        * "node" ~> the field is associated with the vorticity points (node_lat, node_lon)
        * "face" ~> the field is associated with the tracer points (face_lat, face_lon)
        * "x_edge" ~> the field is associated with the u-velocity points (face_lat, node_lon)
        * "y_edge" ~> the field is associated with the v-velocity points (node_lat, face_lon)

    When using a uxarray.UxDataArray object,
    * The uxarray.UxDataArray.UxGrid object must have the "Conventions" attribute set to "UGRID-1.0"
      and the uxarray.UxDataArray object must comply with the UGRID conventions.
      See https://ugrid-conventions.github.io/ugrid-conventions/ for more information.

    """

    @staticmethod
    def _interp_template(
        self,
        ti: int,
        ei: int,
        bcoords: np.ndarray,
        tau: np.float32 | np.float64,
        t: np.float32 | np.float64,
        z: np.float32 | np.float64,
        y: np.float32 | np.float64,
        x: np.float32 | np.float64,
    ) -> np.float32 | np.float64:
        """Template function used for the signature check of the lateral interpolation methods."""
        return 0.0

    def _validate_interp_function(self, func: Callable) -> bool:
        """Ensures that the function has the correct signature."""
        template_sig = inspect.signature(self._interp_template)
        func_sig = inspect.signature(func)

        if len(template_sig.parameters) != len(func_sig.parameters):
            return False

        for (_name1, param1), (_name2, param2) in zip(
            template_sig.parameters.items(), func_sig.parameters.items(), strict=False
        ):
            if param1.kind != param2.kind:
                return False
            if param1.annotation != param2.annotation:
                return False

        return_annotation = func_sig.return_annotation
        template_return = template_sig.return_annotation

        if return_annotation != template_return:
            return False

        return True

    def __init__(
        self,
        name: str,
        data: xr.DataArray | ux.UxDataArray,
        grid: ux.Grid | UxGrid | Grid,
        mesh_type: Mesh = "flat",
        interp_method: Callable | None = None,
    ):
        if not isinstance(data, (ux.UxDataArray, xr.DataArray)):
            raise ValueError(
                f"Expected `data` to be a uxarray.UxDataArray or xarray.DataArray object, got {type(data)}."
            )
        if not isinstance(name, str):
            raise ValueError(f"Expected `name` to be a string, got {type(name)}.")
        if not isinstance(grid, (ux.Grid, UxGrid, Grid)):
            raise ValueError(
                f"Expected `grid` to be a uxarray.Grid, parcels UxGrid, or parcels Grid object, got {type(grid)}."
            )

        assert_valid_mesh(mesh_type)

        _assert_compatible_combination(data, grid)

        self.name = name
        self.data = data
        if isinstance(grid, ux.Grid):
            self.grid = ensure_uxgrid(grid)
        else:
            self.grid = grid

        try:
            self.time_interval = get_time_interval(data)
        except ValueError as e:
            e.add_note(
                f"Error getting time interval for field {name!r}. Are you sure that the time dimension on the xarray dataset is stored as datetime or cftime datetime objects?"
            )
            raise e

        # For compatibility with parts of the codebase that rely on v3 definition of Grid.
        # Should be worked to be removed in v4
        if isinstance(grid, Grid):
            self.gridadapter = GridAdapter(grid)
        else:
            self.gridadapter = None

        try:
            if isinstance(data, ux.UxDataArray):
                _assert_valid_uxdataarray(data)
                # TODO: For unstructured grids, validate that `data.uxgrid` is the same as `grid`
            else:
                pass  # TODO v4: Add validation for xr.DataArray objects
        except Exception as e:
            e.add_note(f"Error validating field {name!r}.")
            raise e

        self._mesh_type = mesh_type

        # Setting the interpolation method dynamically
        if interp_method is None:
            self._interp_method = self._interp_template  # Default to method that returns 0 always
        else:
            self._validate_interp_function(interp_method)
            self._interp_method = interp_method

        self.igrid = -1  # Default the grid index to -1

        if self._mesh_type == "flat" or (self.name not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self._mesh_type == "spherical":
            self.units = unitconverters_map[self.name]
        else:
            raise ValueError("Unsupported mesh type in data array attributes. Choose either: 'spherical' or 'flat'")

        if "time" not in self.data.dims:
            raise ValueError("Field is missing a 'time' dimension. ")

    def __repr__(self):
        return field_repr(self)

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if not isinstance(value, UnitConverter):
            raise ValueError(f"Units must be a UnitConverter object, got {type(value)}")
        self._units = value

    @property
    def lat(self):
        if type(self.data) is ux.UxDataArray:
            if self.data.attrs["location"] == "node":
                return self.grid.node_lat
            elif self.data.attrs["location"] == "face":
                return self.grid.face_lat
            elif self.data.attrs["location"] == "edge":
                return self.grid.edge_lat
        else:
            return self.gridadapter.lat

    @property
    def lon(self):
        if type(self.data) is ux.UxDataArray:
            if self.data.attrs["location"] == "node":
                return self.grid.node_lon
            elif self.data.attrs["location"] == "face":
                return self.grid.face_lon
            elif self.data.attrs["location"] == "edge":
                return self.grid.edge_lon
        else:
            return self.gridadapter.lon

    @property
    def depth(self):
        if type(self.data) is ux.UxDataArray:
            vertical_location = get_vertical_location_from_dims(self.data.dims)
            if vertical_location == "center":
                return self.grid.nz1
            elif vertical_location == "face":
                return self.grid.nz
        else:
            return self.gridadapter.depth

    @property
    def xdim(self):
        if type(self.data) is xr.DataArray:
            return self.gridadapter.xdim
        else:
            raise NotImplementedError("xdim not implemented for unstructured grids")

    @property
    def ydim(self):
        if type(self.data) is xr.DataArray:
            return self.gridadapter.ydim
        else:
            raise NotImplementedError("ydim not implemented for unstructured grids")

    @property
    def zdim(self):
        if type(self.data) is xr.DataArray:
            return self.gridadapter.zdim
        else:
            if "nz1" in self.data.dims:
                return self.data.sizes["nz1"]
            elif "nz" in self.data.dims:
                return self.data.sizes["nz"]
            else:
                return 0

    @property
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, method: Callable):
        self._validate_interp_function(method)
        self._interp_method = method

    def _check_velocitysampling(self):
        if self.name in ["U", "V", "W"]:
            warnings.warn(
                "Sampling of velocities should normally be done using fieldset.UV or fieldset.UVW object; tread carefully",
                RuntimeWarning,
                stacklevel=2,
            )

    def __getitem__(self, key):
        self._check_velocitysampling()
        try:
            if _isParticle(key):
                return self.eval(key.time, key.depth, key.lat, key.lon, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=None)

    def eval(self, time: datetime, z, y, x, particle=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        if particle is None:
            _ei = None
        else:
            _ei = particle.ei[self.igrid]

        try:
            tau, ti = _search_time_index(self, time)
            bcoords, _ei = self.grid.search(self, z, y, x, ei=_ei)
            value = self._interp_method(self, ti, _ei, bcoords, tau, time, z, y, x)

            if np.isnan(value):
                # Detect Out-of-bounds sampling and raise exception
                _raise_field_out_of_bound_error(z, y, x)
            else:
                return value

        except (FieldSamplingError, FieldOutOfBoundError, FieldOutOfBoundSurfaceError) as e:
            e.add_note(f"Error interpolating field '{self.name}'.")
            raise e

        if applyConversion:
            return self.units.to_target(value, z, y, x)
        else:
            return value

    def _rescale_and_set_minmax(self, data):
        data[np.isnan(data)] = 0
        return data

    def __getattr__(self, key: str):
        return getattr(self.data, key)

    def __contains__(self, key: str):
        return key in self.data


class VectorField:
    """VectorField class that holds vector field data needed to execute particles."""

    @staticmethod
    def _vector_interp_template(
        self,
        ti: int,
        ei: int,
        bcoords: np.ndarray,
        t: np.float32 | np.float64,
        z: np.float32 | np.float64,
        y: np.float32 | np.float64,
        x: np.float32 | np.float64,
    ) -> np.float32 | np.float64:
        """Template function used for the signature check of the lateral interpolation methods."""
        return 0.0

    def _validate_vector_interp_function(self, func: Callable):
        """Ensures that the function has the correct signature."""
        expected_params = ["ti", "ei", "bcoords", "t", "z", "y", "x"]
        expected_return_types = (np.float32, np.float64)

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check the parameter names and count
        if params != expected_params:
            raise TypeError(f"Function must have parameters {expected_params}, but got {params}")

        # Check return annotation if present
        return_annotation = sig.return_annotation
        if return_annotation not in (inspect.Signature.empty, *expected_return_types):
            raise TypeError(f"Function must return a float, but got {return_annotation}")

    def __init__(
        self, name: str, U: Field, V: Field, W: Field | None = None, vector_interp_method: Callable | None = None
    ):
        self.name = name
        self.U = U
        self.V = V
        self.W = W

        if W is None:
            assert_same_time_interval((U, V))
        else:
            assert_same_time_interval((U, V, W))

        self.time_interval = U.time_interval

        if self.W:
            self.vector_type = "3D"
        else:
            self.vector_type = "2D"

        # Setting the interpolation method dynamically
        if vector_interp_method is None:
            self._vector_interp_method = None
        else:
            self._validate_vector_interp_function(vector_interp_method)
            self._interp_method = vector_interp_method

    def __repr__(self):
        return f"""<{type(self).__name__}>
    name: {self.name!r}
    U: {default_repr(self.U)}
    V: {default_repr(self.V)}
    W: {default_repr(self.W)}"""

    @property
    def vector_interp_method(self):
        return self._vector_interp_method

    @vector_interp_method.setter
    def vector_interp_method(self, method: Callable):
        self._validate_vector_interp_function(method)
        self._vector_interp_method = method

    # @staticmethod
    # TODO : def _check_grid_dimensions(grid1, grid2):
    #     return (
    #         np.allclose(grid1.lon, grid2.lon)
    #         and np.allclose(grid1.lat, grid2.lat)
    #         and np.allclose(grid1.depth, grid2.depth)
    #         and np.allclose(grid1.time, grid2.time)
    #     )
    def _interpolate(self, time, z, y, x, ei):
        bcoords, _ei, ti = self._search_indices(time, z, y, x, ei=ei)

        if self._vector_interp_method is None:
            u = self.U.eval(time, z, y, x, _ei, applyConversion=False)
            v = self.V.eval(time, z, y, x, _ei, applyConversion=False)
            if "3D" in self.vector_type:
                w = self.W.eval(time, z, y, x, _ei, applyConversion=False)
                return (u, v, w)
            else:
                return (u, v, 0)
        else:
            (u, v, w) = self._vector_interp_method(ti, _ei, bcoords, time, z, y, x)
            return (u, v, w)

    def eval(self, time, z, y, x, ei=None, applyConversion=True):
        if ei is None:
            _ei = 0
        else:
            _ei = ei[self.igrid]

        (u, v, w) = self._interpolate(time, z, y, x, _ei)

        if applyConversion:
            u = self.U.units.to_target(u, z, y, x)
            v = self.V.units.to_target(v, z, y, x)
            if "3D" in self.vector_type:
                w = self.W.units.to_target(w, z, y, x)

        return (u, v, w)

    def __getitem__(self, key):
        try:
            if _isParticle(key):
                return self.eval(key.time, key.depth, key.lat, key.lon, key.ei)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=self.vector_type)


def _assert_valid_uxdataarray(data: ux.UxDataArray):
    """Verifies that all the required attributes are present in the xarray.DataArray or
    uxarray.UxDataArray object.
    """
    # Validate dimensions
    if not ("nz1" in data.dims or "nz" in data.dims):
        raise ValueError(
            "Field is missing a 'nz1' or 'nz' dimension in the field's metadata. "
            "This attribute is required for xarray.DataArray objects."
        )

    if "time" not in data.dims:
        raise ValueError(
            "Field is missing a 'time' dimension in the field's metadata. "
            "This attribute is required for xarray.DataArray objects."
        )

    # Validate attributes
    required_keys = ["location", "mesh"]
    for key in required_keys:
        if key not in data.attrs.keys():
            raise ValueError(
                f"Field is missing a '{key}' attribute in the field's metadata. "
                "This attribute is required for xarray.DataArray objects."
            )

    _assert_valid_uxgrid(data.uxgrid)


def _assert_valid_uxgrid(grid):
    """Verifies that all the required attributes are present in the uxarray.UxDataArray.UxGrid object."""
    if "Conventions" not in grid.attrs.keys():
        raise ValueError(
            "Field is missing a 'Conventions' attribute in the field's metadata. "
            "This attribute is required for uxarray.UxDataArray objects."
        )
    if grid.attrs["Conventions"] != "UGRID-1.0":
        raise ValueError(
            "Field has a 'Conventions' attribute that is not 'UGRID-1.0'. "
            "This attribute is required for uxarray.UxDataArray objects."
            "See https://ugrid-conventions.github.io/ugrid-conventions/ for more information."
        )


def _assert_compatible_combination(data: xr.DataArray | ux.UxDataArray, grid: ux.Grid | Grid):
    if isinstance(data, ux.UxDataArray):
        if not isinstance(grid, ux.Grid):
            raise ValueError(
                f"Incompatible data-grid combination. Data is a uxarray.UxDataArray, expected `grid` to be a uxarray.Grid object, got {type(grid)}."
            )
    elif isinstance(data, xr.DataArray):
        if not isinstance(grid, Grid):
            raise ValueError(
                f"Incompatible data-grid combination. Data is a xarray.DataArray, expected `grid` to be a parcels Grid object, got {type(grid)}."
            )


def get_time_interval(data: xr.DataArray | ux.UxDataArray) -> TimeInterval | None:
    if len(data.time) == 1:
        return None

    return TimeInterval(data.time.values[0], data.time.values[-1])


def assert_same_time_interval(fields: list[Field]) -> None:
    if len(fields) == 0:
        return

    reference_time_interval = fields[0].time_interval

    for field in fields[1:]:
        if field.time_interval != reference_time_interval:
            raise ValueError(
                f"Fields must have the same time domain. {fields[0].name}: {reference_time_interval}, {field.name}: {field.time_interval}"
            )
