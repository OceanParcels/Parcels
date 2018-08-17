from parcels.field import Field, VectorField
from parcels.gridset import GridSet
from parcels.grid import RectilinearZGrid
from parcels.loggers import logger
import numpy as np
from os import path
from glob import glob
from copy import deepcopy


__all__ = ['FieldSet', 'FieldList']


class FieldSet(object):
    """FieldSet class that holds hydrodynamic data needed to execute particles

    :param U: :class:`parcels.field.Field` object for zonal velocity component
    :param V: :class:`parcels.field.Field` object for meridional velocity component
    :param fields: Dictionary of additional :class:`parcels.field.Field` objects
    """
    def __init__(self, U, V, fields=None):
        self.gridset = GridSet()
        if U:
            self.add_field(U, 'U')
            self.time_origin = self.U.grid.time_origin if isinstance(self.U, Field) else self.U[0].grid.time_origin
        if V:
            self.add_field(V, 'V')

        # Add additional fields as attributes
        if fields:
            for name, field in fields.items():
                self.add_field(field, name)

    @classmethod
    def from_data(cls, data, dimensions, transpose=False, mesh='spherical',
                  allow_time_extrapolation=None, time_periodic=False, **kwargs):
        """Initialise FieldSet object from raw data

        :param data: Dictionary mapping field names to numpy arrays.
               Note that at least a 'U' and 'V' numpy array need to be given

               1. If data shape is [xdim, ydim], [xdim, ydim, zdim], [xdim, ydim, tdim] or [xdim, ydim, zdim, tdim],
                  whichever is relevant for the dataset, use the flag transpose=True
               2. If data shape is [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
                  use the flag transpose=False (default value)
               3. If data has any other shape, you first need to reorder it
        :param dimensions: Dictionary mapping field dimensions (lon,
               lat, depth, time) to numpy arrays.
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable
               (e.g. dimensions['U'], dimensions['V'], etc).
        :param transpose: Boolean whether to transpose data on read-in
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        """

        fields = {}
        for name, datafld in data.items():
            # Use dimensions[name] if dimensions is a dict of dicts
            dims = dimensions[name] if name in dimensions else dimensions

            if allow_time_extrapolation is None:
                allow_time_extrapolation = False if 'time' in dims else True

            lon = dims['lon']
            lat = dims['lat']
            depth = np.zeros(1, dtype=np.float32) if 'depth' not in dims else dims['depth']
            time = np.zeros(1, dtype=np.float64) if 'time' not in dims else dims['time']
            grid = RectilinearZGrid(lon, lat, depth, time, mesh=mesh)

            fields[name] = Field(name, datafld, grid=grid, transpose=transpose,
                                 allow_time_extrapolation=allow_time_extrapolation, time_periodic=time_periodic, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    def add_field(self, field, name=None):
        """Add a :class:`parcels.field.Field` object to the FieldSet

        :param field: :class:`parcels.field.Field` object to be added
        :param name: Name of the :class:`parcels.field.Field` object to be added
        """
        name = field.name if name is None else name
        if isinstance(field, list):
            setattr(self, name, FieldList(field))
            for fld in field:
                self.gridset.add_grid(fld)
                fld.fieldset = self
        else:
            setattr(self, name, field)
            self.gridset.add_grid(field)
            field.fieldset = self

    def add_vector_field(self, vfield):
        """Add a :class:`parcels.field.VectorField` object to the FieldSet

        :param vfield: :class:`parcels.field.VectorField` object to be added
        """
        setattr(self, vfield.name, vfield)
        vfield.fieldset = self

    def check_complete(self):
        assert self.U, 'FieldSet does not have a Field named "U"'
        assert self.V, 'FieldSet does not have a Field named "V"'
        for attr, value in vars(self).items():
            if type(value) is Field:
                assert value.name == attr, 'Field %s.name (%s) is not consistent' % (value.name, attr)

        for g in self.gridset.grids:
            g.check_zonal_periodic()
            if len(g.time) == 1:
                continue
            assert isinstance(g.time_origin, type(self.time_origin)), 'time origins of different grids must be have the same type'
            if g.time_origin:
                g.time = g.time + (g.time_origin - self.time_origin) / np.timedelta64(1, 's')
                if g.defer_load:
                    g.time_full = g.time_full + (g.time_origin - self.time_origin) / np.timedelta64(1, 's')
                g.time_origin = self.time_origin
        if not hasattr(self, 'UV'):
            if isinstance(self.U, FieldList):
                self.add_vector_field(VectorFieldList('UV', self.U, self.V))
            else:
                self.add_vector_field(VectorField('UV', self.U, self.V))
        if not hasattr(self, 'UVW') and hasattr(self, 'W'):
            if isinstance(self.U, FieldList):
                self.add_vector_field(VectorFieldList('UVW', self.U, self.V, self.W))
            else:
                self.add_vector_field(VectorField('UVW', self.U, self.V, self.W))

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, indices=None,
                    mesh='spherical', allow_time_extrapolation=None, time_periodic=False, full_load=False, **kwargs):
        """Initialises FieldSet object from NetCDF files

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files,
               or be a list of file.
        :param variables: Dictionary mapping variables to variable
               names in the netCDF file(s).
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable
               (e.g. dimensions['U'], dimensions['V'], etc).
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param full_load: boolean whether to fully load the data or only pre-load them. (default: False)
               It is advised not to fully load the data, since in that case Parcels deals with
               a better memory management during particle set execution.
               full_load is however sometimes necessary for plotting the fields.
        """

        fields = {}
        for var, name in variables.items():
            # Resolve all matching paths for the current variable
            paths = filenames[var] if type(filenames) is dict else filenames
            if not isinstance(paths, list):
                paths = sorted(glob(str(paths)))
            if len(paths) == 0:
                raise IOError("FieldSet files not found: %s" % str(filenames[var]))
            for fp in paths:
                if not path.exists(fp):
                    raise IOError("FieldSet file not found: %s" % str(fp))

            # Use dimensions[var] and indices[var] if either of them is a dict of dicts
            dims = dimensions[var] if var in dimensions else dimensions
            dims['data'] = name
            inds = indices[var] if (indices and var in indices) else indices

            grid = None
            # check if grid has already been processed (i.e. if other fields have same filenames, dimensions and indices)
            for procvar, _ in fields.items():
                procdims = dimensions[procvar] if procvar in dimensions else dimensions
                procinds = indices[procvar] if (indices and procvar in indices) else indices
                if (type(filenames) is not dict or filenames[procvar] == filenames[var]) \
                        and procdims == dims and procinds == inds:
                    grid = fields[procvar].grid
                    kwargs['dataFiles'] = fields[procvar].dataFiles
                    break
            fields[var] = Field.from_netcdf(paths, var, dims, inds, grid=grid, mesh=mesh,
                                            allow_time_extrapolation=allow_time_extrapolation,
                                            time_periodic=time_periodic, full_load=full_load, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    @classmethod
    def from_nemo(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                  allow_time_extrapolation=None, time_periodic=False,
                  tracer_interp_method='linear', **kwargs):
        """Initialises FieldSet object from NetCDF files of Curvilinear NEMO fields.
        Note that this assumes there is a variable mesh_mask that is used for the dimensions

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files,
               or be a list of file. At least a 'mesh_mask' needs to be present
        :param variables: Dictionary mapping variables to variable
               names in the netCDF file(s). Must include a variable 'mesh_mask' that
               holds the dimensions
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable
               (e.g. dimensions['U'], dimensions['V'], etc).
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param tracer_interp_method: Method for interpolation of tracer fields. Either 'linear' or 'nearest'
               Note that in the case of from_nemo(), the velocity fields are default to 'cgrid_linear'

        """

        dimension_filename = filenames.pop('mesh_mask')

        interp_method = {}
        for v in variables:
            if v in ['U', 'V', 'W']:
                interp_method[v] = 'cgrid_linear'
            else:
                interp_method[v] = tracer_interp_method

        return cls.from_netcdf(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                               allow_time_extrapolation=allow_time_extrapolation, interp_method=interp_method,
                               dimension_filename=dimension_filename, **kwargs)

    @classmethod
    def from_parcels(cls, basename, uvar='vozocrtx', vvar='vomecrty', indices=None, extra_fields=None,
                     allow_time_extrapolation=None, time_periodic=False, full_load=False, **kwargs):
        """Initialises FieldSet data from NetCDF files using the Parcels FieldSet.write() conventions.

        :param basename: Base name of the file(s); may contain
               wildcards to indicate multiple files.
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
        :param extra_fields: Extra fields to read beyond U and V
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param full_load: boolean whether to fully load the data or only pre-load them. (default: False)
               It is advised not to fully load the data, since in that case Parcels deals with
               a better memory management during particle set execution.
               full_load is however sometimes necessary for plotting the fields.
        """

        if extra_fields is None:
            extra_fields = {}

        dimensions = {}
        default_dims = {'lon': 'nav_lon', 'lat': 'nav_lat',
                        'depth': 'depth', 'time': 'time_counter'}
        extra_fields.update({'U': uvar, 'V': vvar})
        for vars in extra_fields:
            dimensions[vars] = deepcopy(default_dims)
            dimensions[vars]['depth'] = 'depth%s' % vars.lower()
        filenames = dict([(v, str("%s%s.nc" % (basename, v)))
                          for v in extra_fields.keys()])
        return cls.from_netcdf(filenames, indices=indices, variables=extra_fields,
                               dimensions=dimensions, allow_time_extrapolation=allow_time_extrapolation,
                               time_periodic=time_periodic, full_load=full_load, **kwargs)

    @property
    def fields(self):
        """Returns a list of all the :class:`parcels.field.Field` objects
        associated with this FieldSet"""
        fields = []
        for v in self.__dict__.values():
            if isinstance(v, Field):
                fields.append(v)
            elif isinstance(v, FieldList):
                for v2 in v:
                    if v2 not in fields:
                        fields.append(v2)
        return fields

    def add_constant(self, name, value):
        """Add a constant to the FieldSet. Note that all constants are
        stored as 32-bit floats. While constants can be updated during
        execution in SciPy mode, they can not be updated in JIT mode.

        :param name: Name of the constant
        :param value: Value of the constant (stored as 32-bit float)
        """
        setattr(self, name, value)

    def add_periodic_halo(self, zonal=False, meridional=False, halosize=5):
        """Add a 'halo' to all :class:`parcels.field.Field` objects in a FieldSet,
        through extending the Field (and lon/lat) by copying a small portion
        of the field on one side of the domain to the other.

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """

        for grid in self.gridset.grids:
            grid.add_periodic_halo(zonal, meridional, halosize)
        for attr, value in iter(self.__dict__.items()):
            if isinstance(value, Field):
                value.add_periodic_halo(zonal, meridional, halosize)

    def eval(self, x, y):
        """Evaluate the zonal and meridional velocities (u,v) at a point (x,y)

        :param x: zonal point to evaluate
        :param y: meridional point to evaluate

        :return u, v: zonal and meridional velocities at point"""

        u = self.U.eval(x, y)
        v = self.V.eval(x, y)
        return u, v

    def write(self, filename):
        """Write FieldSet to NetCDF file using NEMO convention

        :param filename: Basename of the output fileset"""
        logger.info("Generating NEMO FieldSet output with basename: %s" % filename)

        if hasattr(self, 'U'):
            self.U.write(filename, varname='vozocrtx')
        if hasattr(self, 'V'):
            self.V.write(filename, varname='vomecrty')

        for v in self.fields:
            if (v.name is not 'U') and (v.name is not 'V'):
                v.write(filename)

    def advancetime(self, fieldset_new):
        """Replace oldest time on FieldSet with new FieldSet
        :param fieldset_new: FieldSet snapshot with which the oldest time has to be replaced"""

        logger.warning_once("Fieldset.advancetime() is deprecated.\n \
                             Parcels deals automatically with loading only 3 time steps simustaneously\
                             such that the total allocated memory remains limited.")

        advance = 0
        for gnew in fieldset_new.gridset.grids:
            gnew.advanced = False

        for fnew in fieldset_new.fields:
            if isinstance(fnew, VectorField):
                continue
            f = getattr(self, fnew.name)
            gnew = fnew.grid
            if not gnew.advanced:
                g = f.grid
                advance2 = g.advancetime(gnew)
                if advance2*advance < 0:
                    raise RuntimeError("Some Fields of the Fieldset are advanced forward and other backward")
                advance = advance2
                gnew.advanced = True
            f.advancetime(fnew, advance == 1)

    def computeTimeChunk(self, time, dt):
        signdt = np.sign(dt)
        nextTime = np.infty if dt > 0 else -np.infty

        for g in self.gridset.grids:
            g.update_status = 'not_updated'
        for f in self.fields:
            if isinstance(f, VectorField) or not f.grid.defer_load:
                continue
            if f.grid.update_status == 'not_updated':
                nextTime_loc = f.grid.computeTimeChunk(f, time, signdt)
            nextTime = min(nextTime, nextTime_loc) if signdt >= 0 else max(nextTime, nextTime_loc)

        for f in self.fields:
            if isinstance(f, VectorField) or not f.grid.defer_load or f.is_gradient:
                continue
            g = f.grid
            if g.update_status == 'first_updated':  # First load of data
                data = np.empty((g.tdim, g.zdim, g.ydim-2*g.meridional_halo, g.xdim-2*g.zonal_halo), dtype=np.float32)
                for tindex in range(3):
                    data = f.computeTimeChunk(data, tindex)
                if f._scaling_factor:
                    data *= f._scaling_factor
                f.data = f.reshape(data)
            elif g.update_status == 'updated':
                data = np.empty((g.tdim, g.zdim, g.ydim-2*g.meridional_halo, g.xdim-2*g.zonal_halo), dtype=np.float32)
                if signdt >= 0:
                    f.data[:2, :] = f.data[1:, :]
                    tindex = 2
                else:
                    f.data[1:, :] = f.data[:2, :]
                    tindex = 0
                data = f.computeTimeChunk(data, tindex)
                if f._scaling_factor:
                    data *= f._scaling_factor
                f.data[tindex, :] = f.reshape(data)[tindex, :]
            if f.gradientx is not None and g.update_status in ['first_updated', 'updated']:
                f.gradient(update=True)

        if abs(nextTime) == np.infty or np.isnan(nextTime):  # Second happens when dt=0
            return nextTime
        else:
            nSteps = int((nextTime - time) / dt)
            if nSteps == 0:
                return nextTime
            else:
                return time + nSteps * dt


class FieldList(list):
    def eval(self, time, x, y, z, applyConversion=True):
        tmp = 0
        for fld in self:
            tmp += fld.eval(time, x, y, z, applyConversion)
        return tmp

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            return self.eval(*key)


class VectorFieldList(list):
    """Class VectorFieldList stores 2 or 3 FieldLists which defines together a vector field.
    This enables to interpolate them as one single vector FieldList in the kernels.

    :param name: Name of the vector field
    :param U: FieldList defining the zonal component
    :param V: FieldList defining the meridional component
    :param W: FieldList defining the vertical component (default: None)
    """

    def __init__(self, name, U, V, W=None):
        self.name = name
        self.U = U
        self.V = V
        self.W = W

    def eval(self, time, x, y, z):
        zonal = meridional = vertical = 0
        if self.W is not None:
            for (U, V, W) in zip(self.U, self.V, self.W):
                vfld = VectorField(self.name, U, V, W)
                vfld.fieldset = self.fieldset
                (tmp1, tmp2, tmp3) = vfld.eval(time, x, y, z)
                zonal += tmp1
                meridional += tmp2
                vertical += tmp3
            return (zonal, meridional, vertical)
        else:
            for (U, V) in zip(self.U, self.V):
                vfld = VectorField(self.name, U, V)
                vfld.fieldset = self.fieldset
                (tmp1, tmp2) = vfld.eval(time, x, y, z)
                zonal += tmp1
                meridional += tmp2
            return (zonal, meridional)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            return self.eval(*key)
