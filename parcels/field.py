import inspect
import warnings
from collections.abc import Callable
from datetime import datetime
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import uxarray as ux
import xarray as xr
from uxarray.grid.neighbors import _barycentric_coordinates

from parcels._typing import (
    Mesh,
    VectorType,
)
from parcels.tools._helpers import default_repr, field_repr
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

from ._index_search import _search_indices_rectilinear, _search_time_index

if TYPE_CHECKING:
    pass

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
        grid: ux.UxGrid | None = None,  # TODO Nick : Once parcels.Grid class is added, allow for it to be passed here
        mesh_type: Mesh = "flat",
        interp_method: Callable | None = None,
        allow_time_extrapolation: bool | None = None,
    ):
        self.name = name
        self.data = data
        self.grid = grid

        _validate_dataarray(data, name)

        self._parent_mesh = data.attrs["mesh"]
        self._mesh_type = mesh_type
        self._location = data.attrs["location"]
        self._vertical_location = None

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

        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if len(getattr(self.data, "time", [])) == 1 else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation

        if type(self.data) is ux.UxDataArray:
            self._spatialhash = self.grid.get_spatial_hash()
            self._gtype = None
            # Set the vertical location
            if "nz1" in data.dims:
                self._vertical_location = "center"
            elif "nz" in data.dims:
                self._vertical_location = "face"
        else:  # TODO Nick : This bit probably needs an overhaul once the parcels.Grid class is integrated.
            self._spatialhash = None
            # Set the grid type
            if "x_g" in self.data.coords:
                lon = self.data.x_g
            elif "x_c" in self.data.coords:
                lon = self.data.x_c
            else:
                lon = self.data.lon

            if "nz1" in self.data.coords:
                depth = self.data.nz1
            elif "nz" in self.data.coords:
                depth = self.data.nz
            elif "depth" in self.data.coords:
                depth = self.data.depth
            else:
                depth = None

            if len(lon.shape) <= 1:
                if depth is None or len(depth.shape) <= 1:
                    self._gtype = GridType.RectilinearZGrid
                else:
                    self._gtype = GridType.RectilinearSGrid
            else:
                if depth is None or len(depth.shape) <= 1:
                    self._gtype = GridType.CurvilinearZGrid
                else:
                    self._gtype = GridType.CurvilinearSGrid

            self._lonlat_minmax = np.array(
                [np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat)], dtype=np.float32
            )

    def __repr__(self):
        return field_repr(self)

    @property
    def lonlat_minmax(self):
        return self._lonlat_minmax

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
            if self._location == "node":
                return self.grid.node_lat
            elif self._location == "face":
                return self.grid.face_lat
            elif self._location == "edge":
                return self.grid.edge_lat
        else:
            return self.data.lat

    @property
    def lon(self):
        if type(self.data) is ux.UxDataArray:
            if self._location == "node":
                return self.grid.node_lon
            elif self._location == "face":
                return self.grid.face_lon
            elif self._location == "edge":
                return self.grid.edge_lon
        else:
            return self.data.lon

    @property
    def depth(self):
        if type(self.data) is ux.UxDataArray:
            if self._vertical_location == "center":
                return self.grid.nz1
            elif self._vertical_location == "face":
                return self.grid.nz
        else:
            return self.data.depth

    @property
    def xdim(self):
        if type(self.data) is xr.DataArray:
            if "face_lon" in self.data.dims:
                return self.data.sizes["face_lon"]
            elif "node_lon" in self.data.dims:
                return self.data.sizes["node_lon"]
            else:
                return self.data.sizes["lon"]
        else:
            return 0  # TODO : Discuss what we want to return as xdim for uxdataarray obj

    @property
    def ydim(self):
        if type(self.data) is xr.DataArray:
            if "face_lat" in self.data.dims:
                return self.data.sizes["face_lat"]
            elif "node_lat" in self.data.dims:
                return self.data.sizes["node_lat"]
            else:
                return self.data.sizes["lat"]
        else:
            return 0  # TODO : Discuss what we want to return as ydim for uxdataarray obj

    @property
    def zdim(self):
        if "nz1" in self.data.dims:
            return self.data.sizes["nz1"]
        elif "nz" in self.data.dims:
            return self.data.sizes["nz"]
        else:
            return 0

    @property
    def n_face(self):
        if type(self.data) is ux.uxDataArray:
            return self.grid.n_face
        else:
            return 0  # TODO : Discuss what we want to return as n_face for dataarray obj

    @property
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, method: Callable):
        self._validate_interp_function(method)
        self._interp_method = method

    def _get_ux_barycentric_coordinates(self, y, x, fi):
        """Checks if a point is inside a given face id. Used for unstructured grids."""
        # Check if particle is in the same face, otherwise search again.
        n_nodes = self.grid.n_nodes_per_face[fi].to_numpy()
        node_ids = self.grid.face_node_connectivity[fi, 0:n_nodes]
        nodes = np.column_stack(
            (
                np.deg2rad(self.grid.node_lon[node_ids].to_numpy()),
                np.deg2rad(self.grid.node_lat[node_ids].to_numpy()),
            )
        )

        coord = np.deg2rad([x, y])
        bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
        err = abs(np.dot(bcoord, nodes[:, 0]) - coord[0]) + abs(np.dot(bcoord, nodes[:, 1]) - coord[1])
        return bcoord, err

    def _search_indices_unstructured(self, z, y, x, ei=None, search2D=False):
        tol = 1e-10
        if ei is None:
            # Search using global search
            fi, bcoords = self._spatialhash.query([[x, y]])  # Get the face id for the particle
            if fi == -1:
                raise FieldOutOfBoundError(z, y, x)
            # TODO Joe : Do the vertical grid search
            # zi = self._vertical_search(z)
            zi = 0  # For now
            return bcoords, self.ravel_index(zi, 0, fi)
        else:
            zi, fi = self.unravel_index(ei[self.igrid])  # Get the z, and face index of the particle
            # Search using nearest neighbors
            bcoords, err = self._get_ux_barycentric_coordinates(y, x, fi)

            if ((bcoords >= 0).all()) and ((bcoords <= 1.0).all()) and err < tol:
                # TODO Joe : Do the vertical grid search
                return bcoords, ei
            else:
                # In this case we need to search the neighbors
                for neighbor in self.grid.face_face_connectivity[fi, :]:
                    bcoords, err = self._get_ux_barycentric_coordinates(y, x, neighbor)
                    if ((bcoords >= 0).all()) and ((bcoords <= 1.0).all()) and err < tol:
                        # TODO Joe: Do the vertical grid search
                        return bcoords, self.ravel_index(zi, 0, neighbor)

                # If we reach this point, we do a global search as a last ditch effort the particle is out of bounds
                fi, bcoords = self._spatialhash.query([[x, y]])  # Get the face id for the particle
                if fi == -1:
                    raise FieldOutOfBoundError(z, y, x)

    def _search_indices_structured(self, z, y, x, ei=None, search2D=False):
        if self._gtype in [GridType.RectilinearSGrid, GridType.RectilinearZGrid]:
            (zeta, eta, xsi, zi, yi, xi) = _search_indices_rectilinear(self, z, y, x, ei=ei, search2D=search2D)
        else:
            ## TODO :  Still need to implement the search_indices_curvilinear
            # (zeta, eta, xsi, zi, yi, xi) = _search_indices_curvilinear(
            #     self, z, y, x, ei=ei, search2D=search2D
            # )
            raise NotImplementedError("Curvilinear grid search not implemented yet")

        return (zeta, eta, xsi, zi, yi, xi)

    def _search_indices(self, time: datetime, z, y, x, ei=None, search2D=False):
        tau, ti = _search_time_index(self, time, self.allow_time_extrapolation)

        if ei is None:
            _ei = None
        else:
            _ei = ei[self.igrid]

        if type(self.data) is ux.UxDataArray:
            bcoords, ei = self._search_indices_unstructured(z, y, x, ei=_ei, search2D=search2D)
        else:
            bcoords, ei = self._search_indices_structured(z, y, x, ei=_ei, search2D=search2D)
        return bcoords, ei, tau, ti

    def _interpolate(self, time: datetime, z, y, x, ei):
        try:
            bcoords, _ei, tau, ti = self._search_indices(time, z, y, x, ei=ei)
            val = self._interp_method(ti, _ei, bcoords, tau, time, z, y, x)

            if np.isnan(val):
                # Detect Out-of-bounds sampling and raise exception
                _raise_field_out_of_bound_error(z, y, x)
            else:
                return val

        except (FieldSamplingError, FieldOutOfBoundError, FieldOutOfBoundSurfaceError) as e:
            e.add_note(f"Error interpolating field '{self.name}'.")
            raise e

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

        value = self._interpolate(time, z, y, x, ei=_ei)

        if applyConversion:
            return self.units.to_target(value, z, y, x)
        else:
            return value

    def _rescale_and_set_minmax(self, data):
        data[np.isnan(data)] = 0
        return data

    def ravel_index(self, zi, yi, xi):
        """Return the flat index of the given grid points.

        Parameters
        ----------
        zi : int
            z index
        yi : int
            y index
        xi : int
            x index. When using an unstructured grid, this is the face index (fi)

        Returns
        -------
        int
            flat index
        """
        if type(self.data) is xr.DataArray:
            return xi + self.xdim * (yi + self.ydim * zi)
        else:
            return xi + self.n_face * zi

    def unravel_index(self, ei):
        """Return the zi, yi, xi indices for a given flat index.
        Only used when working with fields on a structured grid.

        Parameters
        ----------
        ei : int
            The flat index to be unraveled.

        Returns
        -------
        zi : int
            The z index.
        yi : int
            The y index.
        xi : int
            The x index.
        """
        if type(self.data) is xr.DataArray:
            _ei = ei[self.igrid]
            zi = _ei // (self.xdim * self.ydim)
            _ei = _ei % (self.xdim * self.ydim)
            yi = _ei // self.xdim
            xi = _ei % self.xdim
            return zi, yi, xi
        else:
            _ei = ei[self.igrid]
            zi = _ei // self.n_face
            fi = _ei % self.n_face
            return zi, fi

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


def _validate_dataarray(data, name):
    """Verifies that all the required attributes are present in the xarray.DataArray or
    uxarray.UxDataArray object.
    """
    if isinstance(data, ux.UxDataArray):
        # Validate dimensions
        if not ("nz1" in data.dims or "nz" in data.dims):
            raise ValueError(
                f"Field {name} is missing a 'nz1' or 'nz' dimension in the field's metadata. "
                "This attribute is required for xarray.DataArray objects."
            )

        if "time" not in data.dims:
            raise ValueError(
                f"Field {name} is missing a 'time' dimension in the field's metadata. "
                "This attribute is required for xarray.DataArray objects."
            )

    # Validate attributes
    required_keys = ["location", "mesh"]
    for key in required_keys:
        if key not in data.attrs.keys():
            raise ValueError(
                f"Field {name} is missing a '{key}' attribute in the field's metadata. "
                "This attribute is required for xarray.DataArray objects."
            )

    if type(data) is ux.UxDataArray:
        _validate_uxgrid(data.uxgrid, name)


def _validate_uxgrid(grid, name):
    """Verifies that all the required attributes are present in the uxarray.UxDataArray.UxGrid object."""
    if "Conventions" not in grid.attrs.keys():
        raise ValueError(
            f"Field {name} is missing a 'Conventions' attribute in the field's metadata. "
            "This attribute is required for uxarray.UxDataArray objects."
        )
    if grid.attrs["Conventions"] != "UGRID-1.0":
        raise ValueError(
            f"Field {name} has a 'Conventions' attribute that is not 'UGRID-1.0'. "
            "This attribute is required for uxarray.UxDataArray objects."
            "See https://ugrid-conventions.github.io/ugrid-conventions/ for more information."
        )
