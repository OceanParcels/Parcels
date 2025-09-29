from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime

import numpy as np
import uxarray as ux
import xarray as xr

from parcels._core.converters import (
    UnitConverter,
    _unitconverters_map,
)
from parcels._core.index_search import GRID_SEARCH_ERROR, LEFT_OUT_OF_BOUNDS, RIGHT_OUT_OF_BOUNDS, _search_time_index
from parcels._core.particle import KernelParticle
from parcels._core.statuscodes import (
    AllParcelsErrorCodes,
    StatusCode,
)
from parcels._core.utils.time import TimeInterval
from parcels._core.uxgrid import UxGrid
from parcels._core.xgrid import XGrid, _transpose_xfield_data_to_tzyx
from parcels._reprs import default_repr
from parcels._typing import VectorType
from parcels.interpolators import (
    UXPiecewiseLinearNode,
    XLinear,
    ZeroInterpolator,
    ZeroInterpolator_Vector,
)
from parcels.utils._helpers import _assert_same_function_signature

__all__ = ["Field", "VectorField"]


def _deal_with_errors(error, key, vector_type: VectorType):
    if isinstance(key, KernelParticle):
        key.state = AllParcelsErrorCodes[type(error)]
    elif isinstance(key[-1], KernelParticle):
        key[-1].state = AllParcelsErrorCodes[type(error)]
    else:
        raise RuntimeError(f"{error}. Error could not be handled because particles was not part of the Field Sampling.")

    if vector_type and "3D" in vector_type:
        return (0, 0, 0)
    elif vector_type == "2D":
        return (0, 0)
    else:
        return 0


_DEFAULT_INTERPOLATOR_MAPPING = {
    XGrid: XLinear,
    UxGrid: UXPiecewiseLinearNode,
}


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
        * attrs: (location, mesh, mesh)

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

    def __init__(
        self,
        name: str,
        data: xr.DataArray | ux.UxDataArray,
        grid: UxGrid | XGrid,
        interp_method: Callable | None = None,
    ):
        if not isinstance(data, (ux.UxDataArray, xr.DataArray)):
            raise ValueError(
                f"Expected `data` to be a uxarray.UxDataArray or xarray.DataArray object, got {type(data)}."
            )
        if not isinstance(name, str):
            raise ValueError(f"Expected `name` to be a string, got {type(name)}.")
        if not isinstance(grid, (UxGrid, XGrid)):
            raise ValueError(f"Expected `grid` to be a parcels UxGrid, or parcels XGrid object, got {type(grid)}.")

        _assert_compatible_combination(data, grid)

        if isinstance(grid, XGrid):
            data = _transpose_xfield_data_to_tzyx(data, grid.xgcm_grid)

        self.name = name
        self.data = data
        self.grid = grid

        try:
            self.time_interval = _get_time_interval(data)
        except ValueError as e:
            e.add_note(
                f"Error getting time interval for field {name!r}. Are you sure that the time dimension on the xarray dataset is stored as timedelta, datetime or cftime datetime objects?"
            )
            raise e

        try:
            if isinstance(data, ux.UxDataArray):
                _assert_valid_uxdataarray(data)
                # TODO: For unstructured grids, validate that `data.uxgrid` is the same as `grid`
            else:
                pass  # TODO v4: Add validation for xr.DataArray objects
        except Exception as e:
            e.add_note(f"Error validating field {name!r}.")
            raise e

        # Setting the interpolation method dynamically
        if interp_method is None:
            self._interp_method = _DEFAULT_INTERPOLATOR_MAPPING[type(self.grid)]
        else:
            _assert_same_function_signature(interp_method, ref=ZeroInterpolator, context="Interpolation")
            self._interp_method = interp_method

        self.igrid = -1  # Default the grid index to -1

        if self.grid._mesh == "flat" or (self.name not in _unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self.grid._mesh == "spherical":
            self.units = _unitconverters_map[self.name]

        if self.data.shape[0] > 1:
            if "time" not in self.data.coords:
                raise ValueError("Field data is missing a 'time' coordinate.")

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if not isinstance(value, UnitConverter):
            raise ValueError(f"Units must be a UnitConverter object, got {type(value)}")
        self._units = value

    @property
    def xdim(self):
        if type(self.data) is xr.DataArray:
            return self.grid.xdim
        else:
            raise NotImplementedError("xdim not implemented for unstructured grids")

    @property
    def ydim(self):
        if type(self.data) is xr.DataArray:
            return self.grid.ydim
        else:
            raise NotImplementedError("ydim not implemented for unstructured grids")

    @property
    def zdim(self):
        if type(self.data) is xr.DataArray:
            return self.grid.zdim
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
        _assert_same_function_signature(method, ref=ZeroInterpolator, context="Interpolation")
        self._interp_method = method

    def _check_velocitysampling(self):
        if self.name in ["U", "V", "W"]:
            warnings.warn(
                "Sampling of velocities should normally be done using fieldset.UV or fieldset.UVW object; tread carefully",
                RuntimeWarning,
                stacklevel=2,
            )

    def eval(self, time: datetime, z, y, x, particles=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        if particles is None:
            _ei = None
        else:
            _ei = particles.ei[:, self.igrid]

        tau, ti = _search_time_index(self, time)
        position = self.grid.search(z, y, x, ei=_ei)
        _update_particles_ei(particles, position, self)
        _update_particle_states_position(particles, position)

        value = self._interp_method(self, ti, position, tau, time, z, y, x)

        _update_particle_states_interp_value(particles, value)

        if applyConversion:
            value = self.units.to_target(value, z, y, x)
        return value

    def __getitem__(self, key):
        self._check_velocitysampling()
        try:
            if isinstance(key, KernelParticle):
                return self.eval(key.time, key.depth, key.lat, key.lon, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=None)


class VectorField:
    """VectorField class that holds vector field data needed to execute particles."""

    def __init__(
        self, name: str, U: Field, V: Field, W: Field | None = None, vector_interp_method: Callable | None = None
    ):
        self.name = name
        self.U = U
        self.V = V
        self.W = W
        self.grid = U.grid
        self.igrid = U.igrid

        if W is None:
            _assert_same_time_interval((U, V))
        else:
            _assert_same_time_interval((U, V, W))

        self.time_interval = U.time_interval

        if self.W:
            self.vector_type = "3D"
        else:
            self.vector_type = "2D"

        # Setting the interpolation method dynamically
        if vector_interp_method is None:
            self._vector_interp_method = None
        else:
            _assert_same_function_signature(vector_interp_method, ref=ZeroInterpolator_Vector, context="Interpolation")
            self._vector_interp_method = vector_interp_method

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
        _assert_same_function_signature(method, ref=ZeroInterpolator_Vector, context="Interpolation")
        self._vector_interp_method = method

    def eval(self, time: datetime, z, y, x, particles=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        if particles is None:
            _ei = None
        else:
            _ei = particles.ei[:, self.igrid]

        tau, ti = _search_time_index(self.U, time)
        position = self.grid.search(z, y, x, ei=_ei)
        _update_particles_ei(particles, position, self)
        _update_particle_states_position(particles, position)

        if self._vector_interp_method is None:
            u = self.U._interp_method(self.U, ti, position, tau, time, z, y, x)
            v = self.V._interp_method(self.V, ti, position, tau, time, z, y, x)
            if "3D" in self.vector_type:
                w = self.W._interp_method(self.W, ti, position, tau, time, z, y, x)
            else:
                w = 0.0

            if applyConversion:
                u = self.U.units.to_target(u, z, y, x)
                v = self.V.units.to_target(v, z, y, x)

        else:
            (u, v, w) = self._vector_interp_method(self, ti, position, tau, time, z, y, x, applyConversion)

        for vel in (u, v, w):
            _update_particle_states_interp_value(particles, vel)

        if applyConversion and ("3D" in self.vector_type):
            w = self.W.units.to_target(w, z, y, x) if self.W else 0.0

        if "3D" in self.vector_type:
            return (u, v, w)
        else:
            return (u, v)

    def __getitem__(self, key):
        try:
            if isinstance(key, KernelParticle):
                return self.eval(key.time, key.depth, key.lat, key.lon, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=self.vector_type)


def _update_particles_ei(particles, position, field):
    """Update the element index (ei) of the particles"""
    if particles is not None:
        if isinstance(field.grid, XGrid):
            particles.ei[:, field.igrid] = field.grid.ravel_index(
                {
                    "X": position["X"][0],
                    "Y": position["Y"][0],
                    "Z": position["Z"][0],
                }
            )
        elif isinstance(field.grid, UxGrid):
            particles.ei[:, field.igrid] = field.grid.ravel_index(
                {
                    "Z": position["Z"][0],
                    "FACE": position["FACE"][0],
                }
            )


def _update_particle_states_position(particles, position):
    """Update the particle states based on the position dictionary."""
    if particles:  # TODO also support uxgrid search
        for dim in ["X", "Y"]:
            if dim in position:
                particles.state = np.maximum(
                    np.where(position[dim][0] == -1, StatusCode.ErrorOutOfBounds, particles.state), particles.state
                )
                particles.state = np.maximum(
                    np.where(position[dim][0] == GRID_SEARCH_ERROR, StatusCode.ErrorGridSearching, particles.state),
                    particles.state,
                )
        if "Z" in position:
            particles.state = np.maximum(
                np.where(position["Z"][0] == RIGHT_OUT_OF_BOUNDS, StatusCode.ErrorOutOfBounds, particles.state),
                particles.state,
            )
            particles.state = np.maximum(
                np.where(position["Z"][0] == LEFT_OUT_OF_BOUNDS, StatusCode.ErrorThroughSurface, particles.state),
                particles.state,
            )


def _update_particle_states_interp_value(particles, value):
    """Update the particle states based on the interpolated value, but only if state is not an Error already."""
    if particles:
        particles.state = np.maximum(
            np.where(np.isnan(value), StatusCode.ErrorInterpolation, particles.state), particles.state
        )


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


def _assert_compatible_combination(data: xr.DataArray | ux.UxDataArray, grid: ux.Grid | XGrid):
    if isinstance(data, ux.UxDataArray):
        if not isinstance(grid, UxGrid):
            raise ValueError(
                f"Incompatible data-grid combination. Data is a uxarray.UxDataArray, expected `grid` to be a UxGrid object, got {type(grid)}."
            )
    elif isinstance(data, xr.DataArray):
        if not isinstance(grid, XGrid):
            raise ValueError(
                f"Incompatible data-grid combination. Data is a xarray.DataArray, expected `grid` to be a parcels Grid object, got {type(grid)}."
            )


def _get_time_interval(data: xr.DataArray | ux.UxDataArray) -> TimeInterval | None:
    if data.shape[0] == 1:
        return None

    return TimeInterval(data.time.values[0], data.time.values[-1])


def _assert_same_time_interval(fields: list[Field]) -> None:
    if len(fields) == 0:
        return

    reference_time_interval = fields[0].time_interval

    for field in fields[1:]:
        if field.time_interval != reference_time_interval:
            raise ValueError(
                f"Fields must have the same time domain. {fields[0].name}: {reference_time_interval}, {field.name}: {field.time_interval}"
            )
