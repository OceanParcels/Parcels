import collections
import math
import warnings
from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import xarray as xr
import uxarray as ux

import parcels.tools.interpolation_utils as i_u
from parcels._compat import add_note
from parcels._interpolation import (
    InterpolationContext2D,
    InterpolationContext3D,
    get_2d_interpolator_registry,
    get_3d_interpolator_registry,
)
from parcels._typing import (
    GridIndexingType,
    InterpMethod,
    InterpMethodOption,
    Mesh,
    VectorType,
    assert_valid_gridindexingtype,
    assert_valid_interp_method,
)
from parcels.tools._helpers import default_repr, field_repr, should_calculate_next_ti
from parcels.tools.converters import (
    TimeConverter,
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
from parcels.tools.warnings import FieldSetWarning
import inspect
from typing import Callable, Union

#from ._index_search import _search_indices_curvilinear, _search_indices_rectilinear, _search_time_index

if TYPE_CHECKING:
    import numpy.typing as npt

    from parcels.xfieldset import XFieldSet

__all__ = ["XField", "XVectorField"]


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
    
class XField:
    """The XField class that holds scalar field data. 
    The `XField` object is a wrapper around a xarray.DataArray or uxarray.UxDataArray object. 
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
        t: Union[np.float32,np.float64],
        z: Union[np.float32,np.float64],
        y: Union[np.float32,np.float64],
        x: Union[np.float32,np.float64]
    )-> Union[np.float32,np.float64]:
        """ Template function used for the signature check of the lateral interpolation methods."""
        return 0.0
    
    def _validate_interp_function(self, func: Callable):
        """Ensures that the function has the correct signature."""
        expected_params = ["ti", "ei", "bcoords", "t", "z", "y", "x"]
        expected_return_types = (np.float32,np.float64)

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check the parameter names and count
        if params != expected_params:
            raise TypeError(
                f"Function must have parameters {expected_params}, but got {params}"
            )

        # Check return annotation if present
        return_annotation = sig.return_annotation
        if return_annotation not in (inspect.Signature.empty, *expected_return_types):
            raise TypeError(
                f"Function must return a float, but got {return_annotation}"
            )

    def __init__(
        self,
        name: str,
        data: xr.DataArray | ux.UxDataArray,
        interp_method: Callable | None = None,
        allow_time_extrapolation: bool | None = None,
    ):
        
        self.name = name
        self.data = data

        self._validate_dataarray(data)

        self._parent_mesh = data.attributes["mesh"]
        self._mesh_type = data.attributes["mesh_type"]
        self._location = data.attributes["location"]

        # Set the vertical location
        if "nz1" in data.dims:
            self._vertical_location = "center"
        elif "nz" in data.dims:
            self._vertical_location = "face"

        # Setting the interpolation method dynamically
        if interp_method is None:
            self._interp_method = self._interp_template # Default to method that returns 0 always
        else:
            self._validate_interp_function(interp_method)
            self._interp_method = interp_method

        self.igrid = -1 # Default the grid index to -1

        if self._mesh_type == "flat" or (self.name not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self._mesh_type == "spherical":
            self.units = unitconverters_map[self.name]
        else:
            raise ValueError("Unsupported mesh type in data array attributes. Choose either: 'spherical' or 'flat'")
        
        self.fieldset: XFieldSet | None = None
        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if len(self.data["time"]) == 1 else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation

    def __repr__(self):
        return field_repr(self)

    @property
    def grid(self):
        if type(self.data) is ux.UxDataArray:
            return self.data.uxgrid
        else:
            return self.data # To do : need to decide on what to return for xarray.DataArray objects
        
    @property
    def lat(self):
        if type(self.data) is ux.UxDataArray:
            if self._location == "node":
                return self.data.uxgrid.node_lat
            elif self._location == "face":
                return self.data.uxgrid.face_lat
            elif self._location == "edge":
                return self.data.uxgrid.edge_lat
        else:
            if self._location == "node":
                return self.data.node_lat
            elif self._location == "face":
                return self.data.face_lat
            elif self._location == "x_edge":
                return self.data.face_lat
            elif self._location == "y_edge":
                return self.data.node_lat

    @property
    def lon(self):
        if type(self.data) is ux.UxDataArray:
            if self._location == "node":
                return self.data.uxgrid.node_lon
            elif self._location == "face":
                return self.data.uxgrid.face_lon
            elif self._location == "edge":
                return self.data.uxgrid.edge_lon
        else:
            if self._location == "node":
                return self.data.node_lon
            elif self._location == "face":
                return self.data.face_lon
            elif self._location == "x_edge":
                return self.data.node_lon
            elif self._location == "y_edge":
                return self.data.face_lon

    @property
    def depth(self):
        if type(self.data) is ux.UxDataArray:
            if self._vertical_location == "center":
                return self.data.uxgrid.nz1
            elif self._vertical_location == "face":
                return self.data.uxgrid.nz
        else:
            if self._vertical_location == "center":
                return self.data.nz1
            elif self._vertical_location == "face":
                return self.data.nz

    @property
    def nx(self):
        if type(self.data) is xr.DataArray:
            if "face_lon" in self.data.dims:
                return self.data.sizes["face_lon"]
            elif "node_lon" in self.data.dims:
                return self.data.sizes["node_lon"]
        else:
            return 0 # To do : Discuss what we want to return for uxdataarray obj
    @property
    def ny(self):
        if type(self.data) is xr.DataArray:
            if "face_lat" in self.data.dims:
                return self.data.sizes["face_lat"]
            elif "node_lat" in self.data.dims:
                return self.data.sizes["node_lat"]
        else:
            return 0 # To do : Discuss what we want to return for uxdataarray obj
       
    @property
    def interp_method(self):
        return self._interp_method 

    @interp_method.setter
    def interp_method(self, method: Callable):
        self._validate_interp_function(method)
        self._interp_method = method

    # @property
    # def gridindexingtype(self):
    #     return self._gridindexingtype
    def _search_indices(self, time, z, y, x, ei=None, search2D=False):

        tau, ti = self._search_time_index(time) # To do : Need to implement this method

        if type(self.data) is ux.UxDataArray:
            bcoords, ei = self._search_indices_unstructured(z, y, x, ei=ei, search2D=search2D) # To do : Need to implement this method
        else:
            bcoords, ei = self._search_indices_structured(z, y, x, ei=ei, search2D=search2D) # To do : Need to implement this method
        return bcoords, ei, ti 
    
    def _interpolate(self, time, z, y, x, ei=None):

        try:
            bcoords, ei, ti = self._search_indices(time, z, y, x, ei=ei)
            val = self._interp_method(ti, ei, bcoords, time, z, y, x)

            if np.isnan(val):
                # Detect Out-of-bounds sampling and raise exception
                _raise_field_out_of_bound_error(z, y, x)
            else:
                return val
            
        except (FieldSamplingError, FieldOutOfBoundError, FieldOutOfBoundSurfaceError) as e:
            e = add_note(e, f"Error interpolating field '{self.name}'.", before=True)
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
        
    def eval(self, time, z, y, x, ei=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """

        value = self._interpolate(time, z, y, x, ei=ei)

        if applyConversion:
            return self.units.to_target(value, z, y, x)
        else:
            return value
      
    def _rescale_and_set_minmax(self, data):
        data[np.isnan(data)] = 0
        return data

    def ravel_index(self, zi, yi, xi):
        """Return the flat index of the given grid points.
        Only used when working with fields on a structured grid.

        Parameters
        ----------
        zi : int
            z index
        yi : int
            y index
        xi : int
            x index

        Returns
        -------
        int
            flat index
        """
        if type(self.data) is xr.DataArray:
            return xi + self.nx * (yi + self.ny * zi)
        else:
            return None

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
            zi = _ei // (self.nx * self.ny)
            _ei = _ei % (self.nx * self.ny)
            yi = _ei // self.nx
            xi = _ei % self.nx
            return zi, yi, xi
        else:
            return None,None,None # To do : Discuss what we want to return for uxdataarray

    def _validate_dataarray(self):
        """ Verifies that all the required attributes are present in the xarray.DataArray or
         uxarray.UxDataArray object."""
      
        # Validate dimensions
        if not( "nz1" in self.data.dims or "nz" in self.data.dims ):
            raise ValueError(
                f"Field {self.name} is missing a 'nz1' or 'nz' dimension in the field's metadata. "
                "This attribute is required for xarray.DataArray objects."
            )
        
        if not( "time" in self.data.dims ):
            raise ValueError(
                f"Field {self.name} is missing a 'time' dimension in the field's metadata. "
                "This attribute is required for xarray.DataArray objects."
            )
        
        # Validate attributes
        required_keys = ["location", "mesh", "mesh_type"]
        for key in required_keys:
            if key not in self.data.attrs.keys():
                raise ValueError(
                    f"Field {self.name} is missing a '{key}' attribute in the field's metadata. "
                    "This attribute is required for xarray.DataArray objects."
                )
            
        if type(self.data) is ux.UxDataArray:
            self._validate_uxgrid()

            
    def _validate_uxgrid(self):
        """ Verifies that all the required attributes are present in the uxarray.UxDataArray.UxGrid object."""

        if "Conventions" not in self.data.uxgrid.attrs.keys():
            raise ValueError(
                f"Field {self.name} is missing a 'Conventions' attribute in the field's metadata. "
                "This attribute is required for uxarray.UxDataArray objects."
            )
        if self.data.uxgrid.attrs["Conventions"] != "UGRID-1.0":
            raise ValueError(
                f"Field {self.name} has a 'Conventions' attribute that is not 'UGRID-1.0'. "
                "This attribute is required for uxarray.UxDataArray objects."
                "See https://ugrid-conventions.github.io/ugrid-conventions/ for more information."
            )

        
    def __getattr__(self, key: str):
        return getattr(self.data, key)

    def __contains__(self, key: str):
        return key in self.data
    

class XVectorField:
    """XVectorField class that holds vector field data needed to execute particles."""
    def __init__(self, name: str, U: XField, V: XField, W: XField | None = None):
        self.name = name
        self.U = U
        self.V = V
        self.W = W

        if self.W:
            self.vector_type = "3D"
        else:
            self.vector_type = "2D"

    def __repr__(self):
        return f"""<{type(self).__name__}>
    name: {self.name!r}
    U: {default_repr(self.U)}
    V: {default_repr(self.V)}
    W: {default_repr(self.W)}"""


    # @staticmethod
    # To do : def _check_grid_dimensions(grid1, grid2):
    #     return (
    #         np.allclose(grid1.lon, grid2.lon)
    #         and np.allclose(grid1.lat, grid2.lat)
    #         and np.allclose(grid1.depth, grid2.depth)
    #         and np.allclose(grid1.time, grid2.time)
    #     )


# Private helper routines
def _barycentric_coordinates(nodes, point):
    """
    Compute the barycentric coordinates of a point P inside a convex polygon using area-based weights.
    So that this method generalizes to n-sided polygons, we use the Waschpress points as the generalized
    barycentric coordinates, which is only valid for convex polygons.

    Parameters
    ----------
        nodes : numpy.ndarray
            Spherical coordinates (lon,lat) of each corner node of a face
        point : numpy.ndarray
            Spherical coordinates (lon,lat) of the point
    Returns
    -------
    numpy.ndarray
        Barycentric coordinates corresponding to each vertex.

    """
    n = len(nodes)
    sum_wi = 0
    w = []

    for i in range(0, n):
        vim1 = nodes[i - 1]
        vi = nodes[i]
        vi1 = nodes[(i + 1) % n]
        a0 = _triangle_area(vim1, vi, vi1)
        a1 = _triangle_area(point, vim1, vi)
        a2 = _triangle_area(point, vi, vi1)
        sum_wi += a0 / (a1 * a2)
        w.append(a0 / (a1 * a2))

    barycentric_coords = [w_i / sum_wi for w_i in w]

    return barycentric_coords

def _triangle_area(A, B, C):
    """
    Compute the area of a triangle given by three points.
    """
    return 0.5 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))