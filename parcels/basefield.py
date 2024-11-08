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
from .ugrid import UGrid

if TYPE_CHECKING:
    from ctypes import _Pointer as PointerType

    from parcels.fieldset import FieldSet

__all__ = ["Field", "VectorField", "NestedField"]


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


class BaseField:
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
    For usage examples see the following tutorials:

    * `Nested Fields <../examples/tutorial_NestedFields.ipynb>`__
    """

    def __init__(
        self,
        name: str | tuple[str, str],
        data,
        lon=None,
        lat=None,
        face_node_connectivity=None,
        depth=None,
        time=None,
        grid=None,
        mesh: Mesh = "flat",
        timestamps=None,
        fieldtype=None,
        transpose=False,
        vmin=None,
        vmax=None,
        cast_data_dtype="float32",
        time_origin=None,
        interp_method: InterpMethod = "linear",
        allow_time_extrapolation: bool | None = None,
        time_periodic: TimePeriodic = False,
        gridindexingtype: GridIndexingType = "nemo",
        to_write=False,
        **kwargs,
    ):
        if kwargs.get("netcdf_decodewarning") is not None:
            _deprecated_param_netcdf_decodewarning()
            kwargs.pop("netcdf_decodewarning")

        if not isinstance(name, tuple):
            self.name = name
            self.filebuffername = name
        else:
            self.name = name[0]
            self.filebuffername = name[1]
        self.data = data
        if grid:
            if grid.defer_load and isinstance(data, np.ndarray):
                raise ValueError(
                    "Cannot combine Grid from defer_loaded Field with np.ndarray data. please specify lon, lat, depth and time dimensions separately"
                )
            self._grid = grid
        else:
            if (time is not None) and isinstance(time[0], np.datetime64):
                time_origin = TimeConverter(time[0])
                time = np.array([time_origin.reltime(t) for t in time])
            else:
                time_origin = TimeConverter(0)

            # joe@fluidnumerics.com
            # This allows for the creation of a UGrid object
            if face_node_connectivity is None:
                self._grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
            else:
                self._grid = UGrid.create_grid(lon, lat, depth, time, face_node_connectivity, time_origin=time_origin, mesh=mesh)


        self.igrid = -1
        self.fieldtype = self.name if fieldtype is None else fieldtype
        self.to_write = to_write
        if self.grid.mesh == "flat" or (self.fieldtype not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh == "spherical":
            self.units = unitconverters_map[self.fieldtype]
        else:
            raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
        self.timestamps = timestamps
        if isinstance(interp_method, dict):
            if self.name in interp_method:
                self.interp_method = interp_method[self.name]
            else:
                raise RuntimeError(f"interp_method is a dictionary but {name} is not in it")
        else:
            self.interp_method = interp_method
        assert_valid_gridindexingtype(gridindexingtype)
        self._gridindexingtype = gridindexingtype
        if self.interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"] and self.grid._gtype in [
            GridType.RectilinearSGrid,
            GridType.CurvilinearSGrid,
        ]:
            warnings.warn(
                "General s-levels are not supported in B-grid. RectilinearSGrid and CurvilinearSGrid can still be used to deal with shaved cells, but the levels must be horizontal.",
                FieldSetWarning,
                stacklevel=2,
            )

        self.fieldset: FieldSet | None = None
        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if len(self.grid.time) == 1 else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation

        self.time_periodic = time_periodic
        if self.time_periodic is not False and self.allow_time_extrapolation:
            warnings.warn(
                "allow_time_extrapolation and time_periodic cannot be used together. allow_time_extrapolation is set to False",
                FieldSetWarning,
                stacklevel=2,
            )
            self.allow_time_extrapolation = False
        if self.time_periodic is True:
            raise ValueError(
                "Unsupported time_periodic=True. time_periodic must now be either False or the length of the period (either float in seconds or datetime.timedelta object."
            )
        if self.time_periodic is not False:
            if isinstance(self.time_periodic, datetime.timedelta):
                self.time_periodic = self.time_periodic.total_seconds()
            if not np.isclose(self.grid.time[-1] - self.grid.time[0], self.time_periodic):
                if self.grid.time[-1] - self.grid.time[0] > self.time_periodic:
                    raise ValueError("Time series provided is longer than the time_periodic parameter")
                self.grid._add_last_periodic_data_timestep = True
                self.grid.time = np.append(self.grid.time, self.grid.time[0] + self.time_periodic)
                self.grid.time_full = self.grid.time

        self.vmin = vmin
        self.vmax = vmax
        self._cast_data_dtype = cast_data_dtype
        if self.cast_data_dtype == "float32":
            self._cast_data_dtype = np.float32
        elif self.cast_data_dtype == "float64":
            self._cast_data_dtype = np.float64

        if not self.grid.defer_load:
            self.data = self._reshape(self.data, transpose)

            # Hack around the fact that NaN and ridiculously large values
            # propagate in SciPy's interpolators
            lib = np if isinstance(self.data, np.ndarray) else da
            self.data[lib.isnan(self.data)] = 0.0
            if self.vmin is not None:
                self.data[self.data < self.vmin] = 0.0
            if self.vmax is not None:
                self.data[self.data > self.vmax] = 0.0

            if self.grid._add_last_periodic_data_timestep:
                self.data = lib.concatenate((self.data, self.data[:1, :]), axis=0)

        self._scaling_factor = None

        # Variable names in JIT code
        self._dimensions = kwargs.pop("dimensions", None)
        self.indices = kwargs.pop("indices", None)
        self._dataFiles = kwargs.pop("dataFiles", None)
        if self.grid._add_last_periodic_data_timestep and self._dataFiles is not None:
            self._dataFiles = np.append(self._dataFiles, self._dataFiles[0])
        self._field_fb_class = kwargs.pop("FieldFileBuffer", None)
        self._netcdf_engine = kwargs.pop("netcdf_engine", "netcdf4")
        self._loaded_time_indices: Iterable[int] = []  # type: ignore
        self._creation_log = kwargs.pop("creation_log", "")
        self.chunksize = kwargs.pop("chunksize", None)
        self.netcdf_chunkdims_name_map = kwargs.pop("chunkdims_name_map", None)
        self.grid.depth_field = kwargs.pop("depth_field", None)

        if self.grid.depth_field == "not_yet_set":
            assert (
                self.grid._z4d
            ), "Providing the depth dimensions from another field data is only available for 4d S grids"

        # data_full_zdim is the vertical dimension of the complete field data, ignoring the indices.
        # (data_full_zdim = grid.zdim if no indices are used, for A- and C-grids and for some B-grids). It is used for the B-grid,
        # since some datasets do not provide the deeper level of data (which is ignored by the interpolation).
        self.data_full_zdim = kwargs.pop("data_full_zdim", None)
        self._data_chunks = []  # type: ignore # the data buffer of the FileBuffer raw loaded data - shall be a list of C-contiguous arrays
        self._c_data_chunks: list[PointerType | None] = []  # C-pointers to the data_chunks array
        self.nchunks: tuple[int, ...] = ()
        self._chunk_set: bool = False
        self.filebuffers = [None] * 2
        if len(kwargs) > 0:
            raise SyntaxError(f'Field received an unexpected keyword argument "{list(kwargs.keys())[0]}"')

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def grid(self):
        return self._grid

    @property
    def lon(self):
        """Lon defined on the Grid object"""
        return self.grid.lon

    @property
    def lat(self):
        """Lat defined on the Grid object"""
        return self.grid.lat

    @property
    def depth(self):
        """Depth defined on the Grid object"""
        return self.grid.depth

    @property
    def cell_edge_sizes(self):
        return self.grid.cell_edge_sizes

    @property
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, value):
        assert_valid_interp_method(value)
        self._interp_method = value

    @property
    def gridindexingtype(self):
        return self._gridindexingtype

    @property
    def cast_data_dtype(self):
        return self._cast_data_dtype

    @property
    def netcdf_engine(self):
        return self._netcdf_engine

    @classmethod
    def _get_dim_filenames(cls, filenames, dim):
        if isinstance(filenames, str) or not isinstance(filenames, collections.abc.Iterable):
            return [filenames]
        elif isinstance(filenames, dict):
            assert dim in filenames.keys(), "filename dimension keys must be lon, lat, depth or data"
            filename = filenames[dim]
            if isinstance(filename, str):
                return [filename]
            else:
                return filename
        else:
            return filenames

    @staticmethod
    def _collect_timeslices(
        timestamps, data_filenames, _grid_fb_class, dimensions, indices, netcdf_engine, netcdf_decodewarning=None
    ):
        if netcdf_decodewarning is not None:
            _deprecated_param_netcdf_decodewarning()
        if timestamps is not None:
            dataFiles = []
            for findex in range(len(data_filenames)):
                stamps_in_file = 1 if isinstance(timestamps[findex], (int, np.datetime64)) else len(timestamps[findex])
                for f in [data_filenames[findex]] * stamps_in_file:
                    dataFiles.append(f)
            timeslices = np.array([stamp for file in timestamps for stamp in file])
            time = timeslices
        else:
            timeslices = []
            dataFiles = []
            for fname in data_filenames:
                with _grid_fb_class(fname, dimensions, indices, netcdf_engine=netcdf_engine) as filebuffer:
                    ftime = filebuffer.time
                    timeslices.append(ftime)
                    dataFiles.append([fname] * len(ftime))
            time = np.concatenate(timeslices).ravel()
            dataFiles = np.concatenate(dataFiles).ravel()
        if time.size == 1 and time[0] is None:
            time[0] = 0
        time_origin = TimeConverter(time[0])
        time = time_origin.reltime(time)

        if not np.all((time[1:] - time[:-1]) > 0):
            id_not_ordered = np.where(time[1:] < time[:-1])[0][0]
            raise AssertionError(
                f"Please make sure your netCDF files are ordered in time. First pair of non-ordered files: {dataFiles[id_not_ordered]}, {dataFiles[id_not_ordered + 1]}"
            )
        return time, time_origin, timeslices, dataFiles