from parcels.field import Field
from parcels.gridset import GridSet
from parcels.grid import RectilinearZGrid
from parcels.scripts import compute_curvilinearGrid_rotationAngles
from parcels.loggers import logger
import numpy as np
from os import path
from glob import glob
from copy import deepcopy


__all__ = ['FieldSet']


class FieldSet(object):
    """FieldSet class that holds hydrodynamic data needed to execute particles

    :param U: :class:`parcels.field.Field` object for zonal velocity component
    :param V: :class:`parcels.field.Field` object for meridional velocity component
    :param fields: Dictionary of additional :class:`parcels.field.Field` objects
    """
    def __init__(self, U, V, fields={}):
        self.gridset = GridSet()
        if U:
            self.add_field(U)
        if V:
            self.add_field(V)
        UV = Field('UV', None)
        UV.fieldset = self
        self.UV = UV

        # Add additional fields as attributes
        for name, field in fields.items():
            self.add_field(field)

    @classmethod
    def from_data(cls, data, dimensions, transpose=False, mesh='spherical',
                  allow_time_extrapolation=True, time_periodic=False, **kwargs):
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
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        """

        fields = {}
        for name, datafld in data.items():
            # Use dimensions[name] if dimensions is a dict of dicts
            dims = dimensions[name] if name in dimensions else dimensions

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

    def add_field(self, field):
        """Add a :class:`parcels.field.Field` object to the FieldSet

        :param field: :class:`parcels.field.Field` object to be added
        """
        setattr(self, field.name, field)
        self.gridset.add_grid(field)
        field.fieldset = self

    def check_complete(self):
        assert(self.U), ('U field is not defined')
        assert(self.V), ('V field is not defined')
        ugrid = self.U.grid
        for g in self.gridset.grids:
            g.check_zonal_periodic()
            if g is ugrid or len(g.time) == 1:
                continue
            assert isinstance(g.time_origin, type(ugrid.time_origin)), 'time origins of different grids must be have the same type'
            if g.time_origin:
                g.time = g.time + (g.time_origin - ugrid.time_origin) / np.timedelta64(1, 's')
                if g.defer_load:
                    g.time_full = g.time
                g.time_origin = ugrid.time_origin

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, indices={},
                    mesh='spherical', allow_time_extrapolation=False, time_periodic=False, full_load=False, **kwargs):
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
            if isinstance(filenames[var], list):
                paths = filenames[var]
            else:
                paths = sorted(glob(str(filenames[var])))
            if len(paths) == 0:
                raise IOError("FieldSet files not found: %s" % str(filenames[var]))
            for fp in paths:
                if not path.exists(fp):
                    raise IOError("FieldSet file not found: %s" % str(fp))

            # Use dimensions[var] and indices[var] if either of them is a dict of dicts
            dims = dimensions[var] if var in dimensions else dimensions
            dims['data'] = name
            inds = indices[var] if var in indices else indices

            fields[var] = Field.from_netcdf(var, dims, paths, inds, mesh=mesh,
                                            allow_time_extrapolation=allow_time_extrapolation,
                                            time_periodic=time_periodic, full_load=full_load, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    @classmethod
    def from_nemo(cls, filenames, variables, dimensions, indices={}, mesh='spherical',
                  allow_time_extrapolation=False, time_periodic=False, **kwargs):
        """Initialises FieldSet object from NetCDF files of Curvilinear NEMO fields.
        Note that this assumes the following default values for the mesh_mask:
        variables['mesh_mask'] = {'cosU': 'cosU',
                                  'sinU': 'sinU',
                                  'cosV': 'cosV',
                                  'sinV': 'sinV'}
        dimensions['mesh_mask'] = {'U': {'lon': 'glamu', 'lat': 'gphiu'},
                                   'V': {'lon': 'glamv', 'lat': 'gphiv'},
                                   'F': {'lon': 'glamf', 'lat': 'gphif'}}

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files,
               or be a list of file. At least a 'mesh_mask' needs to be present
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
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        """

        if 'mesh_mask' not in variables:
            variables['mesh_mask'] = {'cosU': 'cosU',
                                      'sinU': 'sinU',
                                      'cosV': 'cosV',
                                      'sinV': 'sinV'}
            dimensions['mesh_mask'] = {'U': {'lon': 'glamu', 'lat': 'gphiu'},
                                       'V': {'lon': 'glamv', 'lat': 'gphiv'},
                                       'F': {'lon': 'glamf', 'lat': 'gphif'}}

        rotation_angles_filename = path.join(path.dirname(filenames['mesh_mask']), 'rotation_angles.nc').replace('/', '_')
        compute_curvilinearGrid_rotationAngles(filenames['mesh_mask'], rotation_angles_filename,
                                               variables=variables['mesh_mask'], dimensions=dimensions['mesh_mask'])

        if 'cosU' not in filenames:
            filenames['cosU'] = rotation_angles_filename
            filenames['sinU'] = rotation_angles_filename
            filenames['cosV'] = rotation_angles_filename
            filenames['sinV'] = rotation_angles_filename

            variables['cosU'] = 'cosU'
            variables['sinU'] = 'sinU'
            variables['cosV'] = 'cosV'
            variables['sinV'] = 'sinV'

            dimensions['cosU'] = {'lon': 'glamu', 'lat': 'gphiu'}
            dimensions['sinU'] = {'lon': 'glamu', 'lat': 'gphiu'}
            dimensions['cosV'] = {'lon': 'glamv', 'lat': 'gphiv'}
            dimensions['sinV'] = {'lon': 'glamv', 'lat': 'gphiv'}

        variables.pop('mesh_mask')
        dimensions.pop('mesh_mask')

        return cls.from_netcdf(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                               allow_time_extrapolation=allow_time_extrapolation, **kwargs)

    @classmethod
    def from_parcels(cls, basename, uvar='vozocrtx', vvar='vomecrty', indices={}, extra_fields={},
                     allow_time_extrapolation=False, time_periodic=False, full_load=False, **kwargs):
        """Initialises FieldSet data from NetCDF files using the Parcels FieldSet.write() conventions.

        :param basename: Base name of the file(s); may contain
               wildcards to indicate multiple files.
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
        :param extra_fields: Extra fields to read beyond U and V
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param full_load: boolean whether to fully load the data or only pre-load them. (default: False)
               It is advised not to fully load the data, since in that case Parcels deals with
               a better memory management during particle set execution.
               full_load is however sometimes necessary for plotting the fields.
        """

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
        return [v for v in self.__dict__.values() if isinstance(v, Field)]

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

        # setting FieldSet constants for use in PeriodicBC kernel. Note using U-Field values
        if zonal:
            self.add_constant('halo_west', self.U.grid.lon[0])
            self.add_constant('halo_east', self.U.grid.lon[-1])
        if meridional:
            self.add_constant('halo_south', self.U.grid.lat[0])
            self.add_constant('halo_north', self.U.grid.lat[-1])

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

        self.U.write(filename, varname='vozocrtx')
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
            if fnew.name == 'UV':
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
        nextTime = np.infty * signdt

        for g in self.gridset.grids:
            g.update_status = 'not_updated'
        for f in self.fields:
            if f.name == 'UV' or not f.grid.defer_load:
                continue
            if f.grid.update_status == 'not_updated':
                nextTime_loc = f.grid.computeTimeChunk(f, time, signdt)
            nextTime = min(nextTime, nextTime_loc) if signdt >= 0 else max(nextTime, nextTime_loc)

        for f in self.fields:
            if f.name == 'UV' or not f.grid.defer_load:
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

        if abs(nextTime) == np.infty or np.isnan(nextTime):  # Second happens when dt=0
            return nextTime
        else:
            nSteps = int((nextTime - time) / dt)
            if nSteps == 0:
                return nextTime
            else:
                return time + nSteps * dt
