import collections
import math
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import xarray as xr

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
from parcels.tools._helpers import calculate_next_ti, default_repr, field_repr
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
    TimeExtrapolationError,
    _raise_field_out_of_bound_error,
)
from parcels.tools.warnings import FieldSetWarning

from ._index_search import _search_indices_curvilinear, _search_indices_rectilinear
from .fieldfilebuffer import (
    NetcdfFileBuffer,
)
from .grid import Grid, GridType

if TYPE_CHECKING:
    import numpy.typing as npt

    from parcels.fieldset import FieldSet

__all__ = ["Field", "NestedField", "VectorField"]


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


def _croco_from_z_to_sigma_scipy(fieldset, time, z, y, x, particle):
    """Calculate local sigma level of the particle, by linearly interpolating the
    scaling function that maps sigma to depth (using local ocean depth H,
    sea-surface Zeta and stretching parameters Cs_w and hc).
    See also https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.grid.html#vertical-grid-parameters
    """
    h = fieldset.H.eval(time, 0, y, x, particle=particle, applyConversion=False)
    zeta = fieldset.Zeta.eval(time, 0, y, x, particle=particle, applyConversion=False)
    sigma_levels = fieldset.U.grid.depth
    z0 = fieldset.hc * sigma_levels + (h - fieldset.hc) * fieldset.Cs_w.data[0, :, 0, 0]
    zvec = z0 + zeta * (1 + (z0 / h))
    zinds = zvec <= z
    if z >= zvec[-1]:
        zi = len(zvec) - 2
    else:
        zi = zinds.argmin() - 1 if z >= zvec[0] else 0

    return sigma_levels[zi] + (z - zvec[zi]) * (sigma_levels[zi + 1] - sigma_levels[zi]) / (zvec[zi + 1] - zvec[zi])


class Field:
    """Class that encapsulates access to field data.

    Parameters
    ----------
    name : str
        Name of the field
    data : np.ndarray
        2D, 3D or 4D numpy array of field data with shape [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
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
    grid : parcels.grid.Grid
        :class:`parcels.grid.Grid` object containing all the lon, lat depth, time
        mesh and time_origin information. Can be constructed from any of the Grid objects
    fieldtype : str
        Type of Field to be used for UnitConverter (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis (only if grid is None)
    interp_method : str
        Method for interpolation. Options are 'linear' (default), 'nearest',
        'linear_invdist_land_tracer', 'cgrid_velocity', 'cgrid_tracer' and 'bgrid_velocity'
    allow_time_extrapolation : bool
        boolean whether to allow for extrapolation in time
        (i.e. beyond the last available time snapshot)
    to_write : bool
        Write the Field in NetCDF format at the same frequency as the ParticleFile outputdt,
        using a filenaming scheme based on the ParticleFile name

    Examples
    --------
    For usage examples see the following tutorials:

    * `Nested Fields <../examples/tutorial_NestedFields.ipynb>`__
    """

    allow_time_extrapolation: bool

    def __init__(
        self,
        name: str | tuple[str, str],
        data,
        lon=None,
        lat=None,
        depth=None,
        time=None,
        grid=None,
        mesh: Mesh = "flat",
        fieldtype=None,
        time_origin: TimeConverter | None = None,
        interp_method: InterpMethod = "linear",
        allow_time_extrapolation: bool | None = None,
        gridindexingtype: GridIndexingType = "nemo",
        to_write: bool = False,
        **kwargs,
    ):
        if not isinstance(name, tuple):
            self.name = name
            self.filebuffername = name
        else:
            self.name = name[0]
            self.filebuffername = name[1]
        self.data = data
        if grid:
            self._grid = grid
        else:
            if (time is not None) and isinstance(time[0], np.datetime64):
                time_origin = TimeConverter(time[0])
                time = np.array([time_origin.reltime(t) for t in time])
            else:
                time_origin = TimeConverter(0)
            self._grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        self.igrid = -1
        self.fieldtype = self.name if fieldtype is None else fieldtype
        self.to_write = to_write
        if self.grid.mesh == "flat" or (self.fieldtype not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh == "spherical":
            self.units = unitconverters_map[self.fieldtype]
        else:
            raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
        self._loaded_time_indices: Iterable[int] = []  # type: ignore
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

        self.data = self._reshape(self.data)
        self._loaded_time_indices = range(self.grid.tdim)

        # Hack around the fact that NaN and ridiculously large values
        # propagate in SciPy's interpolators
        self.data[np.isnan(self.data)] = 0.0
        self._scaling_factor = None

        self._dimensions = kwargs.pop("dimensions", None)
        self._dataFiles = kwargs.pop("dataFiles", None)
        self._creation_log = kwargs.pop("creation_log", "")

        # data_full_zdim is the vertical dimension of the complete field data, ignoring the indices.
        # (data_full_zdim = grid.zdim if no indices are used, for A- and C-grids and for some B-grids). It is used for the B-grid,
        # since some datasets do not provide the deeper level of data (which is ignored by the interpolation).
        self.data_full_zdim = kwargs.pop("data_full_zdim", None)
        self.filebuffers = [None] * 2
        if len(kwargs) > 0:
            raise SyntaxError(f'Field received an unexpected keyword argument "{list(kwargs.keys())[0]}"')

    def __repr__(self) -> str:
        return field_repr(self)

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
    def interp_method(self):
        return self._interp_method

    @interp_method.setter
    def interp_method(self, value):
        assert_valid_interp_method(value)
        self._interp_method = value

    @property
    def gridindexingtype(self):
        return self._gridindexingtype

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
    def _collect_timeslices(data_filenames, dimensions, indices):
        timeslices = []
        dataFiles = []
        for fname in data_filenames:
            with NetcdfFileBuffer(fname, dimensions, indices) as filebuffer:
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

    @classmethod
    def from_netcdf(
        cls,
        filenames,
        variable,
        dimensions,
        grid=None,
        mesh: Mesh = "spherical",
        allow_time_extrapolation: bool | None = None,
        **kwargs,
    ) -> "Field":
        """Create field from netCDF file.

        Parameters
        ----------
        filenames : list of str or dict
            list of filenames to read for the field. filenames can be a list ``[files]`` or
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data)
            In the latter case, time values are in filenames[data]
        variable : dict, tuple of str or str
            Dict or tuple mapping field name to variable name in the NetCDF file.
        dimensions : dict
            Dictionary mapping variable names for the relevant dimensions in the NetCDF file
        mesh :
            String indicating the type of mesh coordinates and
            units used during velocity interpolation:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation in time
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        gridindexingtype : str
            The type of gridindexing. Either 'nemo' (default), 'mitgcm', 'mom5', 'pop', or 'croco' are supported.
            See also the Grid indexing documentation on oceanparcels.org
        grid :
             (Default value = None)
        **kwargs :
            Keyword arguments passed to the :class:`Field` constructor.
        """
        if isinstance(variable, str):  # for backward compatibility with Parcels < 2.0.0
            variable = (variable, variable)
        elif isinstance(variable, dict):
            assert (
                len(variable) == 1
            ), "Field.from_netcdf() supports only one variable at a time. Use FieldSet.from_netcdf() for multiple variables."
            variable = tuple(variable.items())[0]
        assert (
            len(variable) == 2
        ), "The variable tuple must have length 2. Use FieldSet.from_netcdf() for multiple variables"

        data_filenames = cls._get_dim_filenames(filenames, "data")
        lonlat_filename = cls._get_dim_filenames(filenames, "lon")
        if isinstance(filenames, dict):
            assert len(lonlat_filename) == 1
        if lonlat_filename != cls._get_dim_filenames(filenames, "lat"):
            raise NotImplementedError(
                "longitude and latitude dimensions are currently processed together from one single file"
            )
        lonlat_filename = lonlat_filename[0]
        if "depth" in dimensions:
            depth_filename = cls._get_dim_filenames(filenames, "depth")
            if isinstance(filenames, dict) and len(depth_filename) != 1:
                raise NotImplementedError("Vertically adaptive meshes not implemented for from_netcdf()")
            depth_filename = depth_filename[0]

        gridindexingtype = kwargs.get("gridindexingtype", "nemo")

        indices: dict[str, npt.NDArray] = {}

        interp_method: InterpMethod = kwargs.pop("interp_method", "linear")
        if type(interp_method) is dict:
            if variable[0] in interp_method:
                interp_method = interp_method[variable[0]]
            else:
                raise RuntimeError(f"interp_method is a dictionary but {variable[0]} is not in it")
        interp_method = cast(InterpMethodOption, interp_method)

        if "lon" in dimensions and "lat" in dimensions:
            with NetcdfFileBuffer(
                lonlat_filename,
                dimensions,
                indices,
                gridindexingtype=gridindexingtype,
            ) as filebuffer:
                lat, lon = filebuffer.latlon
                indices = filebuffer.indices
                # Check if parcels_mesh has been explicitly set in file
                if "parcels_mesh" in filebuffer.dataset.attrs:
                    mesh = filebuffer.dataset.attrs["parcels_mesh"]
        else:
            lon = 0
            lat = 0
            mesh = "flat"

        if "depth" in dimensions:
            with NetcdfFileBuffer(
                depth_filename,
                dimensions,
                indices,
                interp_method=interp_method,
                gridindexingtype=gridindexingtype,
            ) as filebuffer:
                filebuffer.name = variable[1]
                depth = filebuffer.depth
                data_full_zdim = filebuffer.data_full_zdim
        else:
            indices["depth"] = np.array([0])
            depth = np.zeros(1)
            data_full_zdim = 1

        kwargs["data_full_zdim"] = data_full_zdim

        if len(data_filenames) > 1 and "time" not in dimensions:
            raise RuntimeError("Multiple files given but no time dimension specified")

        if grid is None:
            # Concatenate time variable to determine overall dimension
            # across multiple files
            if "time" in dimensions:
                time, time_origin, timeslices, dataFiles = cls._collect_timeslices(data_filenames, dimensions, indices)
                grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
                kwargs["dataFiles"] = dataFiles
            else:  # e.g. for the CROCO CS_w field, see https://github.com/OceanParcels/Parcels/issues/1831
                grid = Grid.create_grid(lon, lat, depth, np.array([0.0]), time_origin=TimeConverter(0.0), mesh=mesh)
                data_filenames = [data_filenames[0]]
        elif grid is not None and ("dataFiles" not in kwargs or kwargs["dataFiles"] is None):
            # ==== means: the field has a shared grid, but may have different data files, so we need to collect the
            # ==== correct file time series again.
            _, _, _, dataFiles = cls._collect_timeslices(data_filenames, dimensions, indices)
            kwargs["dataFiles"] = dataFiles

        if "time" in indices:
            warnings.warn(
                "time dimension in indices is not necessary anymore. It is then ignored.", FieldSetWarning, stacklevel=2
            )

        with NetcdfFileBuffer(  # type: ignore[operator]
            data_filenames,
            dimensions,
            indices,
            interp_method=interp_method,
            data_full_zdim=data_full_zdim,
        ) as filebuffer:
            # If Field.from_netcdf is called directly, it may not have a 'data' dimension
            # In that case, assume that 'name' is the data dimension
            filebuffer.name = variable[1]
            buffer_data = filebuffer.data
            if len(buffer_data.shape) == 4:
                errormessage = (
                    f"Field {filebuffer.name} expecting a data shape of [tdim={grid.tdim}, zdim={grid.zdim}, "
                    f"ydim={grid.ydim}, xdim={grid.xdim }] "
                    f"but got shape {buffer_data.shape}."
                )
                assert buffer_data.shape[0] == grid.tdim, errormessage
                assert buffer_data.shape[2] == grid.ydim, errormessage
                assert buffer_data.shape[3] == grid.xdim, errormessage

        data = buffer_data

        if allow_time_extrapolation is None:
            allow_time_extrapolation = False if "time" in dimensions else True

        kwargs["dimensions"] = dimensions.copy()

        return cls(
            variable,
            data,
            grid=grid,
            allow_time_extrapolation=allow_time_extrapolation,
            interp_method=interp_method,
            **kwargs,
        )

    @classmethod
    def from_xarray(
        cls,
        da: xr.DataArray,
        name: str,
        dimensions,
        mesh: Mesh = "spherical",
        allow_time_extrapolation: bool | None = None,
        **kwargs,
    ):
        """Create field from xarray Variable.

        Parameters
        ----------
        da : xr.DataArray
            Xarray DataArray
        name : str
            Name of the Field
        dimensions : dict
            Dictionary mapping variable names for the relevant dimensions in the DataArray
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation in time
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        **kwargs :
            Keyword arguments passed to the :class:`Field` constructor.
        """
        data = da.data
        interp_method = kwargs.pop("interp_method", "linear")

        time = da[dimensions["time"]].values if "time" in dimensions else np.array([0.0])
        depth = da[dimensions["depth"]].values if "depth" in dimensions else np.array([0])
        lon = da[dimensions["lon"]].values
        lat = da[dimensions["lat"]].values

        time_origin = TimeConverter(time[0])
        time = time_origin.reltime(time)  # type: ignore[assignment]

        grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        return cls(
            name,
            data,
            grid=grid,
            allow_time_extrapolation=allow_time_extrapolation,
            interp_method=interp_method,
            **kwargs,
        )

    def _reshape(self, data):
        # Ensure that field data is the right data type
        if not isinstance(data, (np.ndarray)):
            data = np.array(data)

        if self.grid.xdim == 1 or self.grid.ydim == 1:
            data = np.squeeze(data)  # First remove all length-1 dimensions in data, so that we can add them below
        if self.grid.xdim == 1 and len(data.shape) < 4:
            data = np.expand_dims(data, axis=-1)
        if self.grid.ydim == 1 and len(data.shape) < 4:
            data = np.expand_dims(data, axis=-2)
        if self.grid.tdim == 1:
            if len(data.shape) < 4:
                data = data.reshape(sum(((1,), data.shape), ()))
        if self.grid.zdim == 1:
            if len(data.shape) == 4:
                data = data.reshape(sum(((data.shape[0],), data.shape[2:]), ()))
        if len(data.shape) == 4:
            errormessage = f"Field {self.name} expecting a data shape of [tdim, zdim, ydim, xdim]. "
            assert data.shape[0] == self.grid.tdim, errormessage
            assert data.shape[2] == self.grid.ydim, errormessage
            assert data.shape[3] == self.grid.xdim, errormessage
            if self.gridindexingtype == "pop":
                assert data.shape[1] == self.grid.zdim or data.shape[1] == self.grid.zdim - 1, errormessage
            else:
                assert data.shape[1] == self.grid.zdim, errormessage
        else:
            assert data.shape == (
                self.grid.tdim,
                self.grid.ydim,
                self.grid.xdim,
            ), f"Field {self.name} expecting a data shape of [tdim, ydim, xdim]. "

        return data

    def set_scaling_factor(self, factor):
        """Scales the field data by some constant factor.

        Parameters
        ----------
        factor :
            scaling factor


        Examples
        --------
        For usage examples see the following tutorial:

        * `Unit converters <../examples/tutorial_unitconverters.ipynb>`__
        """
        if self._scaling_factor:
            raise NotImplementedError(f"Scaling factor for field {self.name} already defined.")
        self._scaling_factor = factor
        self.data *= factor

    def _search_indices(self, time, z, y, x, particle=None, search2D=False):
        tau, ti = self._time_index(time)

        if self.grid._gtype in [GridType.RectilinearSGrid, GridType.RectilinearZGrid]:
            (zeta, eta, xsi, zi, yi, xi) = _search_indices_rectilinear(
                self, time, z, y, x, ti, particle=particle, search2D=search2D
            )
        else:
            (zeta, eta, xsi, zi, yi, xi) = _search_indices_curvilinear(
                self, time, z, y, x, ti, particle=particle, search2D=search2D
            )
        return (tau, zeta, eta, xsi, ti, zi, yi, xi)

    def _interpolator2D(self, time, z, y, x, particle=None):
        """Impelement 2D interpolation with coordinate transformations as seen in Delandmeter and Van Sebille (2019), 10.5194/gmd-12-3571-2019.."""
        try:
            f = get_2d_interpolator_registry()[self.interp_method]
        except KeyError:
            if self.interp_method == "cgrid_velocity":
                raise RuntimeError(
                    f"{self.name} is a scalar field. cgrid_velocity interpolation method should be used for vector fields (e.g. FieldSet.UV)"
                )
            else:
                raise RuntimeError(self.interp_method + " is not implemented for 2D grids")

        (tau, _, eta, xsi, ti, _, yi, xi) = self._search_indices(time, z, y, x, particle=particle)

        ctx = InterpolationContext2D(self.data, tau, eta, xsi, ti, yi, xi)
        return f(ctx)

    def _interpolator3D(self, time, z, y, x, particle=None):
        """Impelement 3D interpolation with coordinate transformations as seen in Delandmeter and Van Sebille (2019), 10.5194/gmd-12-3571-2019.."""
        try:
            f = get_3d_interpolator_registry()[self.interp_method]
        except KeyError:
            raise RuntimeError(self.interp_method + " is not implemented for 3D grids")

        (tau, zeta, eta, xsi, ti, zi, yi, xi) = self._search_indices(time, z, y, x, particle=particle)

        ctx = InterpolationContext3D(self.data, tau, zeta, eta, xsi, ti, zi, yi, xi, self.gridindexingtype)
        return f(ctx)

    def _spatial_interpolation(self, time, z, y, x, particle=None):
        """Interpolate spatial field values."""
        try:
            if self.grid.zdim == 1:
                val = self._interpolator2D(time, z, y, x, particle=particle)
            else:
                val = self._interpolator3D(time, z, y, x, particle=particle)

            if np.isnan(val):
                # Detect Out-of-bounds sampling and raise exception
                _raise_field_out_of_bound_error(z, y, x)
            else:
                return val

        except (FieldSamplingError, FieldOutOfBoundError, FieldOutOfBoundSurfaceError) as e:
            e = add_note(e, f"Error interpolating field '{self.name}'.", before=True)
            raise e

    def _time_index(self, time):
        """Find the index in the time array associated with a given time.

        Note that we normalize to either the first or the last index
        if the sampled value is outside the time value range.
        """
        if not self.allow_time_extrapolation and (time < self.grid.time[0] or time > self.grid.time[-1]):
            raise TimeExtrapolationError(time, field=self)
        time_index = self.grid.time <= time

        if time_index.all():
            # If given time > last known field time, use
            # the last field frame without interpolation
            ti = len(self.grid.time) - 1
        elif np.logical_not(time_index).all():
            # If given time < any time in the field, use
            # the first field frame without interpolation
            ti = 0
        else:
            ti = time_index.argmin() - 1 if time_index.any() else 0
        if self.grid.tdim == 1:
            tau = 0
        elif ti == len(self.grid.time) - 1:
            tau = 1
        else:
            tau = (
                (time - self.grid.time[ti]) / (self.grid.time[ti + 1] - self.grid.time[ti])
                if self.grid.time[ti] != self.grid.time[ti + 1]
                else 0
            )
        return tau, ti

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

    def eval(self, time, z, y, x, particle=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        if self.gridindexingtype == "croco" and self not in [self.fieldset.H, self.fieldset.Zeta]:
            z = _croco_from_z_to_sigma_scipy(self.fieldset, time, z, y, x, particle=particle)

        value = self._spatial_interpolation(time, z, y, x, particle=particle)

        if applyConversion:
            return self.units.to_target(value, z, y, x)
        else:
            return value

    def write(self, filename, varname=None):
        """Write a :class:`Field` to a netcdf file.

        Parameters
        ----------
        filename : str
            Basename of the file (i.e. '{filename}{Field.name}.nc')
        varname : str
            Name of the field, to be appended to the filename. (Default value = None)
        """
        filepath = str(Path(f"{filename}{self.name}.nc"))
        if varname is None:
            varname = self.name
        # Derive name of 'depth' variable for NEMO convention
        vname_depth = f"depth{self.name.lower()}"

        # Create DataArray objects for file I/O
        if self.grid._gtype == GridType.RectilinearZGrid:
            nav_lon = xr.DataArray(
                self.grid.lon + np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32),
                coords=[("y", self.grid.lat), ("x", self.grid.lon)],
            )
            nav_lat = xr.DataArray(
                self.grid.lat.reshape(self.grid.ydim, 1) + np.zeros(self.grid.xdim, dtype=np.float32),
                coords=[("y", self.grid.lat), ("x", self.grid.lon)],
            )
        elif self.grid._gtype == GridType.CurvilinearZGrid:
            nav_lon = xr.DataArray(self.grid.lon, coords=[("y", range(self.grid.ydim)), ("x", range(self.grid.xdim))])
            nav_lat = xr.DataArray(self.grid.lat, coords=[("y", range(self.grid.ydim)), ("x", range(self.grid.xdim))])
        else:
            raise NotImplementedError("Field.write only implemented for RectilinearZGrid and CurvilinearZGrid")

        attrs = {"units": "seconds since " + str(self.grid.time_origin)} if self.grid.time_origin.calendar else {}
        time_counter = xr.DataArray(self.grid.time, dims=["time_counter"], attrs=attrs)
        vardata = xr.DataArray(
            self.data.reshape((self.grid.tdim, self.grid.zdim, self.grid.ydim, self.grid.xdim)),
            dims=["time_counter", vname_depth, "y", "x"],
        )
        # Create xarray Dataset and output to netCDF format
        attrs = {"parcels_mesh": self.grid.mesh}
        dset = xr.Dataset(
            {varname: vardata},
            coords={"nav_lon": nav_lon, "nav_lat": nav_lat, "time_counter": time_counter, vname_depth: self.grid.depth},
            attrs=attrs,
        )
        dset.to_netcdf(filepath, unlimited_dims="time_counter")

    def _rescale_and_set_minmax(self, data):
        data[np.isnan(data)] = 0
        if self._scaling_factor:
            data *= self._scaling_factor
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
            x index

        Returns
        -------
        int
            flat index
        """
        return xi + self.grid.xdim * (yi + self.grid.ydim * zi)

    def unravel_index(self, ei):
        """Return the zi, yi, xi indices for a given flat index.

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
        _ei = ei[self.igrid]
        zi = _ei // (self.grid.xdim * self.grid.ydim)
        _ei = _ei % (self.grid.xdim * self.grid.ydim)
        yi = _ei // self.grid.xdim
        xi = _ei % self.grid.xdim
        return zi, yi, xi


class VectorField:
    """Class VectorField stores 2 or 3 fields which defines together a vector field.
    This enables to interpolate them as one single vector field in the kernels.

    Parameters
    ----------
    name : str
        Name of the vector field
    U : parcels.field.Field
        field defining the zonal component
    V : parcels.field.Field
        field defining the meridional component
    W : parcels.field.Field
        field defining the vertical component (default: None)
    """

    def __init__(self, name: str, U: Field, V: Field, W: Field | None = None):
        self.name = name
        self.U = U
        self.V = V
        self.W = W
        if self.U.gridindexingtype == "croco" and self.W:
            self.vector_type: VectorType = "3DSigma"
        elif self.W:
            self.vector_type = "3D"
        else:
            self.vector_type = "2D"
        self.gridindexingtype = U.gridindexingtype
        if self.U.interp_method == "cgrid_velocity":
            assert self.V.interp_method == "cgrid_velocity", "Interpolation methods of U and V are not the same."
            assert self._check_grid_dimensions(U.grid, V.grid), "Dimensions of U and V are not the same."
            if W is not None and self.U.gridindexingtype != "croco":
                assert W.interp_method == "cgrid_velocity", "Interpolation methods of U and W are not the same."
                assert self._check_grid_dimensions(U.grid, W.grid), "Dimensions of U and W are not the same."

    def __repr__(self):
        return f"""<{type(self).__name__}>
    name: {self.name!r}
    U: {default_repr(self.U)}
    V: {default_repr(self.V)}
    W: {default_repr(self.W)}"""

    @staticmethod
    def _check_grid_dimensions(grid1, grid2):
        return (
            np.allclose(grid1.lon, grid2.lon)
            and np.allclose(grid1.lat, grid2.lat)
            and np.allclose(grid1.depth, grid2.depth)
            and np.allclose(grid1.time_full, grid2.time_full)
        )

    def spatial_c_grid_interpolation2D(self, time, z, y, x, particle=None, applyConversion=True):
        grid = self.U.grid
        (tau, _, eta, xsi, ti, zi, yi, xi) = self.U._search_indices(time, z, y, x, particle=particle)

        if grid._gtype in [GridType.RectilinearSGrid, GridType.RectilinearZGrid]:
            px = np.array([grid.lon[xi], grid.lon[xi + 1], grid.lon[xi + 1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi + 1], grid.lat[yi + 1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])

        if grid.mesh == "spherical":
            px[0] = px[0] + 360 if px[0] < x - 225 else px[0]
            px[0] = px[0] - 360 if px[0] > x + 225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
        xx = (1 - xsi) * (1 - eta) * px[0] + xsi * (1 - eta) * px[1] + xsi * eta * px[2] + (1 - xsi) * eta * px[3]
        assert abs(xx - x) < 1e-4
        c1 = i_u._geodetic_distance(py[0], py[1], px[0], px[1], grid.mesh, np.dot(i_u.phi2D_lin(0.0, xsi), py))
        c2 = i_u._geodetic_distance(py[1], py[2], px[1], px[2], grid.mesh, np.dot(i_u.phi2D_lin(eta, 1.0), py))
        c3 = i_u._geodetic_distance(py[2], py[3], px[2], px[3], grid.mesh, np.dot(i_u.phi2D_lin(1.0, xsi), py))
        c4 = i_u._geodetic_distance(py[3], py[0], px[3], px[0], grid.mesh, np.dot(i_u.phi2D_lin(eta, 0.0), py))

        def _calc_UV(ti, yi, xi):
            if grid.zdim == 1:
                if self.gridindexingtype == "nemo":
                    U0 = self.U.data[ti, yi + 1, xi] * c4
                    U1 = self.U.data[ti, yi + 1, xi + 1] * c2
                    V0 = self.V.data[ti, yi, xi + 1] * c1
                    V1 = self.V.data[ti, yi + 1, xi + 1] * c3
                elif self.gridindexingtype in ["mitgcm", "croco"]:
                    U0 = self.U.data[ti, yi, xi] * c4
                    U1 = self.U.data[ti, yi, xi + 1] * c2
                    V0 = self.V.data[ti, yi, xi] * c1
                    V1 = self.V.data[ti, yi + 1, xi] * c3
            else:
                if self.gridindexingtype == "nemo":
                    U0 = self.U.data[ti, zi, yi + 1, xi] * c4
                    U1 = self.U.data[ti, zi, yi + 1, xi + 1] * c2
                    V0 = self.V.data[ti, zi, yi, xi + 1] * c1
                    V1 = self.V.data[ti, zi, yi + 1, xi + 1] * c3
                elif self.gridindexingtype in ["mitgcm", "croco"]:
                    U0 = self.U.data[ti, zi, yi, xi] * c4
                    U1 = self.U.data[ti, zi, yi, xi + 1] * c2
                    V0 = self.V.data[ti, zi, yi, xi] * c1
                    V1 = self.V.data[ti, zi, yi + 1, xi] * c3
            U = (1 - xsi) * U0 + xsi * U1
            V = (1 - eta) * V0 + eta * V1
            rad = np.pi / 180.0
            deg2m = 1852 * 60.0
            if applyConversion:
                meshJac = (deg2m * deg2m * math.cos(rad * y)) if grid.mesh == "spherical" else 1
            else:
                meshJac = deg2m if grid.mesh == "spherical" else 1

            jac = i_u._compute_jacobian_determinant(py, px, eta, xsi) * meshJac

            u = (
                (-(1 - eta) * U - (1 - xsi) * V) * px[0]
                + ((1 - eta) * U - xsi * V) * px[1]
                + (eta * U + xsi * V) * px[2]
                + (-eta * U + (1 - xsi) * V) * px[3]
            ) / jac
            v = (
                (-(1 - eta) * U - (1 - xsi) * V) * py[0]
                + ((1 - eta) * U - xsi * V) * py[1]
                + (eta * U + xsi * V) * py[2]
                + (-eta * U + (1 - xsi) * V) * py[3]
            ) / jac
            if isinstance(u, da.core.Array):
                u = u.compute()
                v = v.compute()
            return (u, v)

        u, v = _calc_UV(ti, yi, xi)
        if calculate_next_ti(ti, tau, self.U.grid.tdim):
            ut1, vt1 = _calc_UV(ti + 1, yi, xi)
            u = (1 - tau) * u + tau * ut1
            v = (1 - tau) * v + tau * vt1
        return (u, v)

    def spatial_c_grid_interpolation3D_full(self, time, z, y, x, particle=None):
        grid = self.U.grid
        (tau, zeta, eta, xsi, ti, zi, yi, xi) = self.U._search_indices(time, z, y, x, particle=particle)

        if grid._gtype in [GridType.RectilinearSGrid, GridType.RectilinearZGrid]:
            px = np.array([grid.lon[xi], grid.lon[xi + 1], grid.lon[xi + 1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi + 1], grid.lat[yi + 1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])

        if grid.mesh == "spherical":
            px[0] = px[0] + 360 if px[0] < x - 225 else px[0]
            px[0] = px[0] - 360 if px[0] > x + 225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
        xx = (1 - xsi) * (1 - eta) * px[0] + xsi * (1 - eta) * px[1] + xsi * eta * px[2] + (1 - xsi) * eta * px[3]
        assert abs(xx - x) < 1e-4

        px = np.concatenate((px, px))
        py = np.concatenate((py, py))
        if grid._z4d:
            pz = np.array(
                [
                    grid.depth[0, zi, yi, xi],
                    grid.depth[0, zi, yi, xi + 1],
                    grid.depth[0, zi, yi + 1, xi + 1],
                    grid.depth[0, zi, yi + 1, xi],
                    grid.depth[0, zi + 1, yi, xi],
                    grid.depth[0, zi + 1, yi, xi + 1],
                    grid.depth[0, zi + 1, yi + 1, xi + 1],
                    grid.depth[0, zi + 1, yi + 1, xi],
                ]
            )
        else:
            pz = np.array(
                [
                    grid.depth[zi, yi, xi],
                    grid.depth[zi, yi, xi + 1],
                    grid.depth[zi, yi + 1, xi + 1],
                    grid.depth[zi, yi + 1, xi],
                    grid.depth[zi + 1, yi, xi],
                    grid.depth[zi + 1, yi, xi + 1],
                    grid.depth[zi + 1, yi + 1, xi + 1],
                    grid.depth[zi + 1, yi + 1, xi],
                ]
            )

        u0 = self.U.data[ti, zi, yi + 1, xi]
        u1 = self.U.data[ti, zi, yi + 1, xi + 1]
        v0 = self.V.data[ti, zi, yi, xi + 1]
        v1 = self.V.data[ti, zi, yi + 1, xi + 1]
        w0 = self.W.data[ti, zi, yi + 1, xi + 1]
        w1 = self.W.data[ti, zi + 1, yi + 1, xi + 1]

        if calculate_next_ti(ti, tau, self.U.grid.tdim):
            u0 = (1 - tau) * u0 + tau * self.U.data[ti + 1, zi, yi + 1, xi]
            u1 = (1 - tau) * u1 + tau * self.U.data[ti + 1, zi, yi + 1, xi + 1]
            v0 = (1 - tau) * v0 + tau * self.V.data[ti + 1, zi, yi, xi + 1]
            v1 = (1 - tau) * v1 + tau * self.V.data[ti + 1, zi, yi + 1, xi + 1]
            w0 = (1 - tau) * w0 + tau * self.W.data[ti + 1, zi, yi + 1, xi + 1]
            w1 = (1 - tau) * w1 + tau * self.W.data[ti + 1, zi + 1, yi + 1, xi + 1]

        U0 = u0 * i_u.jacobian3D_lin_face(pz, py, px, zeta, eta, 0, "zonal", grid.mesh)
        U1 = u1 * i_u.jacobian3D_lin_face(pz, py, px, zeta, eta, 1, "zonal", grid.mesh)
        V0 = v0 * i_u.jacobian3D_lin_face(pz, py, px, zeta, 0, xsi, "meridional", grid.mesh)
        V1 = v1 * i_u.jacobian3D_lin_face(pz, py, px, zeta, 1, xsi, "meridional", grid.mesh)
        W0 = w0 * i_u.jacobian3D_lin_face(pz, py, px, 0, eta, xsi, "vertical", grid.mesh)
        W1 = w1 * i_u.jacobian3D_lin_face(pz, py, px, 1, eta, xsi, "vertical", grid.mesh)

        # Computing fluxes in half left hexahedron -> flux_u05
        xx = [
            px[0],
            (px[0] + px[1]) / 2,
            (px[2] + px[3]) / 2,
            px[3],
            px[4],
            (px[4] + px[5]) / 2,
            (px[6] + px[7]) / 2,
            px[7],
        ]
        yy = [
            py[0],
            (py[0] + py[1]) / 2,
            (py[2] + py[3]) / 2,
            py[3],
            py[4],
            (py[4] + py[5]) / 2,
            (py[6] + py[7]) / 2,
            py[7],
        ]
        zz = [
            pz[0],
            (pz[0] + pz[1]) / 2,
            (pz[2] + pz[3]) / 2,
            pz[3],
            pz[4],
            (pz[4] + pz[5]) / 2,
            (pz[6] + pz[7]) / 2,
            pz[7],
        ]
        flux_u0 = u0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0.5, 0, "zonal", grid.mesh)
        flux_v0_halfx = v0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0, 0.5, "meridional", grid.mesh)
        flux_v1_halfx = v1 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 1, 0.5, "meridional", grid.mesh)
        flux_w0_halfx = w0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0, 0.5, 0.5, "vertical", grid.mesh)
        flux_w1_halfx = w1 * i_u.jacobian3D_lin_face(zz, yy, xx, 1, 0.5, 0.5, "vertical", grid.mesh)
        flux_u05 = flux_u0 + flux_v0_halfx - flux_v1_halfx + flux_w0_halfx - flux_w1_halfx

        # Computing fluxes in half front hexahedron -> flux_v05
        xx = [
            px[0],
            px[1],
            (px[1] + px[2]) / 2,
            (px[0] + px[3]) / 2,
            px[4],
            px[5],
            (px[5] + px[6]) / 2,
            (px[4] + px[7]) / 2,
        ]
        yy = [
            py[0],
            py[1],
            (py[1] + py[2]) / 2,
            (py[0] + py[3]) / 2,
            py[4],
            py[5],
            (py[5] + py[6]) / 2,
            (py[4] + py[7]) / 2,
        ]
        zz = [
            pz[0],
            pz[1],
            (pz[1] + pz[2]) / 2,
            (pz[0] + pz[3]) / 2,
            pz[4],
            pz[5],
            (pz[5] + pz[6]) / 2,
            (pz[4] + pz[7]) / 2,
        ]
        flux_u0_halfy = u0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0.5, 0, "zonal", grid.mesh)
        flux_u1_halfy = u1 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0.5, 1, "zonal", grid.mesh)
        flux_v0 = v0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0, 0.5, "meridional", grid.mesh)
        flux_w0_halfy = w0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0, 0.5, 0.5, "vertical", grid.mesh)
        flux_w1_halfy = w1 * i_u.jacobian3D_lin_face(zz, yy, xx, 1, 0.5, 0.5, "vertical", grid.mesh)
        flux_v05 = flux_u0_halfy - flux_u1_halfy + flux_v0 + flux_w0_halfy - flux_w1_halfy

        # Computing fluxes in half lower hexahedron -> flux_w05
        xx = [
            px[0],
            px[1],
            px[2],
            px[3],
            (px[0] + px[4]) / 2,
            (px[1] + px[5]) / 2,
            (px[2] + px[6]) / 2,
            (px[3] + px[7]) / 2,
        ]
        yy = [
            py[0],
            py[1],
            py[2],
            py[3],
            (py[0] + py[4]) / 2,
            (py[1] + py[5]) / 2,
            (py[2] + py[6]) / 2,
            (py[3] + py[7]) / 2,
        ]
        zz = [
            pz[0],
            pz[1],
            pz[2],
            pz[3],
            (pz[0] + pz[4]) / 2,
            (pz[1] + pz[5]) / 2,
            (pz[2] + pz[6]) / 2,
            (pz[3] + pz[7]) / 2,
        ]
        flux_u0_halfz = u0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0.5, 0, "zonal", grid.mesh)
        flux_u1_halfz = u1 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0.5, 1, "zonal", grid.mesh)
        flux_v0_halfz = v0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 0, 0.5, "meridional", grid.mesh)
        flux_v1_halfz = v1 * i_u.jacobian3D_lin_face(zz, yy, xx, 0.5, 1, 0.5, "meridional", grid.mesh)
        flux_w0 = w0 * i_u.jacobian3D_lin_face(zz, yy, xx, 0, 0.5, 0.5, "vertical", grid.mesh)
        flux_w05 = flux_u0_halfz - flux_u1_halfz + flux_v0_halfz - flux_v1_halfz + flux_w0

        surf_u05 = i_u.jacobian3D_lin_face(pz, py, px, 0.5, 0.5, 0.5, "zonal", grid.mesh)
        jac_u05 = i_u.jacobian3D_lin_face(pz, py, px, zeta, eta, 0.5, "zonal", grid.mesh)
        U05 = flux_u05 / surf_u05 * jac_u05

        surf_v05 = i_u.jacobian3D_lin_face(pz, py, px, 0.5, 0.5, 0.5, "meridional", grid.mesh)
        jac_v05 = i_u.jacobian3D_lin_face(pz, py, px, zeta, 0.5, xsi, "meridional", grid.mesh)
        V05 = flux_v05 / surf_v05 * jac_v05

        surf_w05 = i_u.jacobian3D_lin_face(pz, py, px, 0.5, 0.5, 0.5, "vertical", grid.mesh)
        jac_w05 = i_u.jacobian3D_lin_face(pz, py, px, 0.5, eta, xsi, "vertical", grid.mesh)
        W05 = flux_w05 / surf_w05 * jac_w05

        jac = i_u.jacobian3D_lin(pz, py, px, zeta, eta, xsi, grid.mesh)
        dxsidt = i_u.interpolate(i_u.phi1D_quad, [U0, U05, U1], xsi) / jac
        detadt = i_u.interpolate(i_u.phi1D_quad, [V0, V05, V1], eta) / jac
        dzetdt = i_u.interpolate(i_u.phi1D_quad, [W0, W05, W1], zeta) / jac

        dphidxsi, dphideta, dphidzet = i_u.dphidxsi3D_lin(zeta, eta, xsi)

        u = np.dot(dphidxsi, px) * dxsidt + np.dot(dphideta, px) * detadt + np.dot(dphidzet, px) * dzetdt
        v = np.dot(dphidxsi, py) * dxsidt + np.dot(dphideta, py) * detadt + np.dot(dphidzet, py) * dzetdt
        w = np.dot(dphidxsi, pz) * dxsidt + np.dot(dphideta, pz) * detadt + np.dot(dphidzet, pz) * dzetdt

        if isinstance(u, da.core.Array):
            u = u.compute()
            v = v.compute()
            w = w.compute()
        return (u, v, w)

    def spatial_c_grid_interpolation3D(self, ti, z, y, x, time, particle=None, applyConversion=True):
        """Perform C grid interpolation in 3D. ::

            +---+---+---+
            |   |V1 |   |
            +---+---+---+
            |U0 |   |U1 |
            +---+---+---+
            |   |V0 |   |
            +---+---+---+

        The interpolation is done in the following by
        interpolating linearly U depending on the longitude coordinate and
        interpolating linearly V depending on the latitude coordinate.
        Curvilinear grids are treated properly, since the element is projected to a rectilinear parent element.
        """
        if self.U.grid._gtype in [GridType.RectilinearSGrid, GridType.CurvilinearSGrid]:
            (u, v, w) = self.spatial_c_grid_interpolation3D_full(time, z, y, x, particle=particle)
        else:
            if self.gridindexingtype == "croco":
                z = _croco_from_z_to_sigma_scipy(self.fieldset, time, z, y, x, particle=particle)
            (u, v) = self.spatial_c_grid_interpolation2D(time, z, y, x, particle=particle)
            w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
            if applyConversion:
                w = self.W.units.to_target(w, z, y, x)
        return (u, v, w)

    def _is_land2D(self, di, yi, xi):
        if self.U.data.ndim == 3:
            if di < np.shape(self.U.data)[0]:
                return np.isclose(self.U.data[di, yi, xi], 0.0) and np.isclose(self.V.data[di, yi, xi], 0.0)
            else:
                return True
        else:
            if di < self.U.grid.zdim and yi < np.shape(self.U.data)[-2] and xi < np.shape(self.U.data)[-1]:
                return np.isclose(self.U.data[0, di, yi, xi], 0.0) and np.isclose(self.V.data[0, di, yi, xi], 0.0)
            else:
                return True

    def spatial_slip_interpolation(self, time, z, y, x, particle=None, applyConversion=True):
        (_, zeta, eta, xsi, ti, zi, yi, xi) = self.U._search_indices(time, z, y, x, particle=particle)
        di = ti if self.U.grid.zdim == 1 else zi  # general third dimension

        f_u, f_v, f_w = 1, 1, 1
        if (
            self._is_land2D(di, yi, xi)
            and self._is_land2D(di, yi, xi + 1)
            and self._is_land2D(di + 1, yi, xi)
            and self._is_land2D(di + 1, yi, xi + 1)
            and eta > 0
        ):
            if self.U.interp_method == "partialslip":
                f_u = f_u * (0.5 + 0.5 * eta) / eta
                if self.vector_type == "3D":
                    f_w = f_w * (0.5 + 0.5 * eta) / eta
            elif self.U.interp_method == "freeslip":
                f_u = f_u / eta
                if self.vector_type == "3D":
                    f_w = f_w / eta
        if (
            self._is_land2D(di, yi + 1, xi)
            and self._is_land2D(di, yi + 1, xi + 1)
            and self._is_land2D(di + 1, yi + 1, xi)
            and self._is_land2D(di + 1, yi + 1, xi + 1)
            and eta < 1
        ):
            if self.U.interp_method == "partialslip":
                f_u = f_u * (1 - 0.5 * eta) / (1 - eta)
                if self.vector_type == "3D":
                    f_w = f_w * (1 - 0.5 * eta) / (1 - eta)
            elif self.U.interp_method == "freeslip":
                f_u = f_u / (1 - eta)
                if self.vector_type == "3D":
                    f_w = f_w / (1 - eta)
        if (
            self._is_land2D(di, yi, xi)
            and self._is_land2D(di, yi + 1, xi)
            and self._is_land2D(di + 1, yi, xi)
            and self._is_land2D(di + 1, yi + 1, xi)
            and xsi > 0
        ):
            if self.U.interp_method == "partialslip":
                f_v = f_v * (0.5 + 0.5 * xsi) / xsi
                if self.vector_type == "3D":
                    f_w = f_w * (0.5 + 0.5 * xsi) / xsi
            elif self.U.interp_method == "freeslip":
                f_v = f_v / xsi
                if self.vector_type == "3D":
                    f_w = f_w / xsi
        if (
            self._is_land2D(di, yi, xi + 1)
            and self._is_land2D(di, yi + 1, xi + 1)
            and self._is_land2D(di + 1, yi, xi + 1)
            and self._is_land2D(di + 1, yi + 1, xi + 1)
            and xsi < 1
        ):
            if self.U.interp_method == "partialslip":
                f_v = f_v * (1 - 0.5 * xsi) / (1 - xsi)
                if self.vector_type == "3D":
                    f_w = f_w * (1 - 0.5 * xsi) / (1 - xsi)
            elif self.U.interp_method == "freeslip":
                f_v = f_v / (1 - xsi)
                if self.vector_type == "3D":
                    f_w = f_w / (1 - xsi)
        if self.U.grid.zdim > 1:
            if (
                self._is_land2D(di, yi, xi)
                and self._is_land2D(di, yi, xi + 1)
                and self._is_land2D(di, yi + 1, xi)
                and self._is_land2D(di, yi + 1, xi + 1)
                and zeta > 0
            ):
                if self.U.interp_method == "partialslip":
                    f_u = f_u * (0.5 + 0.5 * zeta) / zeta
                    f_v = f_v * (0.5 + 0.5 * zeta) / zeta
                elif self.U.interp_method == "freeslip":
                    f_u = f_u / zeta
                    f_v = f_v / zeta
            if (
                self._is_land2D(di + 1, yi, xi)
                and self._is_land2D(di + 1, yi, xi + 1)
                and self._is_land2D(di + 1, yi + 1, xi)
                and self._is_land2D(di + 1, yi + 1, xi + 1)
                and zeta < 1
            ):
                if self.U.interp_method == "partialslip":
                    f_u = f_u * (1 - 0.5 * zeta) / (1 - zeta)
                    f_v = f_v * (1 - 0.5 * zeta) / (1 - zeta)
                elif self.U.interp_method == "freeslip":
                    f_u = f_u / (1 - zeta)
                    f_v = f_v / (1 - zeta)

        u = f_u * self.U.eval(time, z, y, x, particle, applyConversion=applyConversion)
        v = f_v * self.V.eval(time, z, y, x, particle, applyConversion=applyConversion)
        if self.vector_type == "3D":
            w = f_w * self.W.eval(time, z, y, x, particle, applyConversion=applyConversion)
            return u, v, w
        else:
            return u, v

    def eval(self, time, z, y, x, particle=None, applyConversion=True):
        if self.U.interp_method in ["partialslip", "freeslip"]:
            return self.spatial_slip_interpolation(time, z, y, x, particle=particle, applyConversion=applyConversion)

        if self.U.interp_method not in ["cgrid_velocity", "partialslip", "freeslip"]:
            u = self.U.eval(time, z, y, x, particle=particle, applyConversion=False)
            v = self.V.eval(time, z, y, x, particle=particle, applyConversion=False)
            if applyConversion:
                u = self.U.units.to_target(u, z, y, x)
                v = self.V.units.to_target(v, z, y, x)
        elif self.U.interp_method == "cgrid_velocity":
            tau, ti = self.U._time_index(time)
            (u, v) = self.spatial_c_grid_interpolation2D(
                time, z, y, x, particle=particle, applyConversion=applyConversion
            )
        if "3D" in self.vector_type:
            w = self.W.eval(time, z, y, x, particle=particle, applyConversion=applyConversion)
            return (u, v, w)
        else:
            return (u, v)

    def __getitem__(self, key):
        try:
            if _isParticle(key):
                return self.eval(key.time, key.depth, key.lat, key.lon, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=self.vector_type)


class NestedField(list):
    """NestedField is a class that allows for interpolation of fields on different grids of potentially varying resolution.

    The NestedField class is a list of Fields where the first Field that contains the particle within the domain is then used for interpolation.
    This induces that the order of the fields in the list matters.
    Each one it its turn, a field is interpolated: if the interpolation succeeds or if an error other
    than `ErrorOutOfBounds` is thrown, the function is stopped. Otherwise, next field is interpolated.
    NestedField returns an `ErrorOutOfBounds` only if last field is as well out of boundaries.
    NestedField is composed of either Fields or VectorFields.

    Parameters
    ----------
    name : str
        Name of the NestedField
    F : list of Field
        List of fields (order matters). F can be a scalar Field, a VectorField, or the zonal component (U) of the VectorField
    V : list of Field
        List of fields defining the meridional component of a VectorField, if F is the zonal component. (default: None)
    W : list of Field
        List of fields defining the vertical component of a VectorField, if F and V are the zonal and meridional components (default: None)


    Examples
    --------
    See `here <../examples/tutorial_NestedFields.ipynb>`__
    for a detailed tutorial

    """

    def __init__(self, name: str, F, V=None, W=None):
        if V is None:
            if isinstance(F[0], VectorField):
                vector_type = F[0].vector_type
            for Fi in F:
                assert isinstance(Fi, Field) or (
                    isinstance(Fi, VectorField) and Fi.vector_type == vector_type
                ), "Components of a NestedField must be Field or VectorField"
                self.append(Fi)
        elif W is None:
            for i, Fi, Vi in zip(range(len(F)), F, V, strict=True):
                assert isinstance(Fi, Field) and isinstance(
                    Vi, Field
                ), "F, and V components of a NestedField must be Field"
                self.append(VectorField(f"{name}_{i}", Fi, Vi))
        else:
            for i, Fi, Vi, Wi in zip(range(len(F)), F, V, W, strict=True):
                assert (
                    isinstance(Fi, Field) and isinstance(Vi, Field) and isinstance(Wi, Field)
                ), "F, V and W components of a NestedField must be Field"
                self.append(VectorField(f"{name}_{i}", Fi, Vi, Wi))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            for iField in range(len(self)):
                try:
                    if _isParticle(key):
                        val = list.__getitem__(self, iField).eval(key.time, key.depth, key.lat, key.lon, particle=None)
                    else:
                        val = list.__getitem__(self, iField).eval(*key)
                    break
                except tuple(AllParcelsErrorCodes.keys()) as error:
                    if iField == len(self) - 1:
                        vector_type = self[iField].vector_type if isinstance(self[iField], VectorField) else None
                        return _deal_with_errors(error, key, vector_type=vector_type)
                    else:
                        pass
            return val
