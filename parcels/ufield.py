import collections
import datetime
import math
import warnings
from collections.abc import Iterable
from ctypes import POINTER, Structure, c_float, c_int, pointer
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import xarray as xr

import parcels.tools.interpolation_utils as i_u
from parcels._typing import (
    GridIndexingType,
    InterpMethod,
    Mesh,
    TimePeriodic,
    VectorType,
    assert_valid_gridindexingtype,
    assert_valid_interp_method,
)
from parcels.tools._helpers import deprecated_made_private
from parcels.tools.converters import (
    Geographic,
    GeographicPolar,
    TimeConverter,
    UnitConverter,
    unitconverters_map,
)
from parcels.tools.statuscodes import (
    AllParcelsErrorCodes,
    FieldOutOfBoundError,
    FieldOutOfBoundSurfaceError,
    FieldSamplingError,
    TimeExtrapolationError,
)
from parcels.tools.warnings import FieldSetWarning, _deprecated_param_netcdf_decodewarning

from .fieldfilebuffer import (
    DaskFileBuffer,
    DeferredDaskFileBuffer,
    DeferredNetcdfFileBuffer,
    NetcdfFileBuffer,
)
from .grid import CGrid, Grid, GridType

if TYPE_CHECKING:
    from ctypes import _Pointer as PointerType

    from parcels.fieldset import FieldSet



__all__ = ["UField", "UVectorField", "UNestedField"]


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

    if vector_type == "3D":
        return (0, 0, 0)
    elif vector_type == "2D":
        return (0, 0)
    else:
        return 0
    
class UField(BaseField):
    """Class that encapsulates access to field data.

    Parameters
    ----------
    name : str
        Name of the field
    data : np.ndarray
        2D, 3D or 4D numpy array of field data.

        1. If data shape is [xdim, ydim], [xdim, ydim, zdim], [xdim, ydim, tdim] or [xdim, ydim, zdim, tdim],
           whichever is relevant for the dataset, use the flag transpose=True
        2. If data shape is [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
           use the flag transpose=False
        3. If data has any other shape, you first need to reorder it
    lon : np.ndarray or list
        Longitude coordinates (numpy vector or array) of the field (only if grid is None)
    lat : np.ndarray or list
        Latitude coordinates (numpy vector or array) of the field (only if grid is None)
    face_node_connectivity: np.ndarray or list dimensioned [nfaces, max_nodes_per_face]
        Connectivity array between faces and nodes (only if grid is None)
    depth : np.ndarray or list
        Depth coordinates (numpy vector or array) of the field (only if grid is None)
    time : np.ndarray
        Time coordinates (numpy vector) of the field (only if grid is None)
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation: (only if grid is None)

        1. spherical: Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat (default): No conversion, lat/lon are assumed to be in m.
    timestamps : np.ndarray
        A numpy array containing the timestamps for each of the files in filenames, for loading
        from netCDF files only. Default is None if the netCDF dimensions dictionary includes time.
    grid : parcels.grid.Grid
        :class:`parcels.grid.Grid` object containing all the lon, lat depth, time
        mesh and time_origin information. Can be constructed from any of the Grid objects
    fieldtype : str
        Type of Field to be used for UnitConverter (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
    transpose : bool
        Transpose data to required (lon, lat) layout
    vmin : float
        Minimum allowed value on the field. Data below this value are set to zero
    vmax : float
        Maximum allowed value on the field. Data above this value are set to zero
    cast_data_dtype : str
        Cast Field data to dtype. Supported dtypes are "float32" (np.float32 (default)) and "float64 (np.float64).
        Note that dtype can only be "float32" in JIT mode
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis (only if grid is None)
    interp_method : str
        Method for interpolation. Options are 'linear' (default), 'nearest',
        'linear_invdist_land_tracer', 'cgrid_velocity', 'cgrid_tracer' and 'bgrid_velocity'
    allow_time_extrapolation : bool
        boolean whether to allow for extrapolation in time
        (i.e. beyond the last available time snapshot)
    time_periodic : bool, float or datetime.timedelta
        To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object).
        The last value of the time series can be provided (which is the same as the initial one) or not (Default: False)
        This flag overrides the allow_time_extrapolation and sets it to False
    chunkdims_name_map : str, optional
        Gives a name map to the FieldFileBuffer that declared a mapping between chunksize name, NetCDF dimension and Parcels dimension;
        required only if currently incompatible OCM field is loaded and chunking is used by 'chunksize' (which is the default)
    to_write : bool
        Write the Field in NetCDF format at the same frequency as the ParticleFile outputdt,
        using a filenaming scheme based on the ParticleFile name

    Examples
    --------

    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    @property
    def face_node_connectivity(self):
        return self.grid.face_node_connectivity
    
    # To do
    #@classmethod
    #def from_netcdf()
    # Likely want to use uxarray for this

    # To do
    #@classmethod
    #def from_uxarray()

    # def _reshape(self, data, transpose=False):

    # def set_scaling_factor(self, factor):
    #     """Scales the field data by some constant factor.

    #     Parameters
    #     ----------
    #     factor :
    #         scaling factor


    #     Examples
    #     --------
    #     For usage examples see the following tutorial:

    #     * `Unit converters <../examples/tutorial_unitconverters.ipynb>`__
    #     """


    # def set_depth_from_field(self, field):
    #     """Define the depth dimensions from another (time-varying) field.

    #     Notes
    #     -----
    #     See `this tutorial <../examples/tutorial_timevaryingdepthdimensions.ipynb>`__
    #     for a detailed explanation on how to set up time-evolving depth dimensions.

    #     """
    #     self.grid.depth_field = field
    #     if self.grid != field.grid:
    #         field.grid.depth_field = field

    # def _calc_cell_edge_sizes(self):
    #     """Method to calculate cell sizes based on numpy.gradient method.

    #     Currently only works for Rectilinear Grids
    #     """

    # def cell_areas(self):
    #     """Method to calculate cell sizes based on cell_edge_sizes.

    #     Currently only works for Rectilinear Grids
    #     """

    # def _search_indices_vertical_s(
    #     self, x: float, y: float, z: float, xi: int, yi: int, xsi: float, eta: float, ti: int, time: float
    # ):

    # def _reconnect_bnd_indices(self, xi, yi, xdim, ydim, sphere_mesh):

    # def _search_indices_curvilinear(self, x, y, z, ti=-1, time=-1, particle=None, search2D=False):

    # def _search_indices(self, x, y, z, ti=-1, time=-1, particle=None, search2D=False):

        # Do hash table search based on x,y location
        # Get list of elements to check
        # Loop over elements, check if particle is in element (barycentric coordinate calc)
        #
        #  (l1,l2,l3, inside_element) = self.barycentric_coordinates(x, y)
        #
        # return barycentric coordinates (2d)
        #        vertical interpolation weight
        #        element index
        #        nearest vertical layer index *above* particle (lower k bound); particle is between layer k and k+1

    #def _interpolator2D(self, ti, z, y, x, particle=None):
    #   """Interpolation method for 2D UField. The UField.data is assumed to 
    #      be provided at the ugrid vertices. The method uses either nearest
    #      neighbor or linear (barycentric) interpolation """

        # (bc, _, ei, _) = self._search_indices(x, y, z, ti, particle=particle)

        # if self.interp_method == "nearest"
        #   idxs = self.face_node_connectivity[ei]
        #   vi = idxs[np.argmax(bc)]
        #   return self.data[ti,vi]
        #
        #  Do nearest neighbour interpolation using vertex with largest barycentric coordinate
        # elif self.interp_method == "linear"
        #  Do barycentric interpolation

    # def _interpolator3D(self, ti, z, y, x, time, particle=None):
        # (bc, zeta, ei, zi) = self._search_indices(x, y, z, ti, particle=particle)

        # if self.interp_method == "nearest"
        #   idxs = self.face_node_connectivity[ei]
        #   vi = idxs[np.argmax(bc)]
        #   zii = zi if zeta <= 0.5 else zi + 1
        #   return self.data[ti,zi,vi]
        #
        #  Do nearest neighbour interpolation using vertex with largest barycentric coordinate
        # elif self.interp_method == "linear"
        #  Do barycentric interpolation
