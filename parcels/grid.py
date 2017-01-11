from parcels.field import Field, UnitConverter, Geographic, GeographicPolar
from parcels.particleset import ParticleSet
import numpy as np
from py import path
from glob import glob
from collections import defaultdict


__all__ = ['Grid']


def unit_converters(mesh):
    """Helper function that assigns :class:`UnitConverter` objects to
    :class:`Field` objects on :class:`Grid`

    :param mesh: mesh type (either `spherical` or `flat`)"""
    if mesh == 'spherical':
        u_units = GeographicPolar()
        v_units = Geographic()
    elif mesh == 'flat':
        u_units = None
        v_units = None
    else:
        raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
    return u_units, v_units


class Grid(object):
    """Grid class that holds hydrodynamic data needed to execute particles

    :param U: :class:`parcels.field.Field` object for zonal velocity component
    :param V: :class:`parcels.field.Field` object for meridional velocity component
    :param allow_time_extrapolation: boolean whether to allow for extrapolation
    :param fields: Dictionary of additional :class:`parcels.field.Field` objects
    """
    def __init__(self, U, V, allow_time_extrapolation=False, fields={}):
        self.U = U
        self.V = V

        # Add additional fields as attributes
        for name, field in fields.items():
            setattr(self, name, field)

    @classmethod
    def from_data(cls, data_u, lon_u, lat_u, data_v, lon_v, lat_v,
                  depth=None, time=None, field_data={}, transpose=True,
                  mesh='spherical', allow_time_extrapolation=True, **kwargs):
        """Initialise Grid object from raw data

        :param data_u: Zonal (U) velocity data
        :param lon_u: Longitude coordinates of the U data
        :param lat_u: Latitude coordinates of the U data
        :param data_v: Meridional (V) velocity data
        :param lon_v: Longitude coordinates of the V data
        :param lat_v: Latitude coordinates of the V data
        :param depth: Depth coordinates of all :class:`Field` objects on the grid
        :param time: Time coordinates of all :class:`Field` objects on the grid
        :param field_data: Dictionary of extra fields (name, data)
        :param transpose: Boolean whether to transpose data on read-in
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
        """

        depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        time = np.zeros(1, dtype=np.float64) if time is None else time
        u_units, v_units = unit_converters(mesh)
        # Create velocity fields
        ufield = Field('U', data_u, lon_u, lat_u, depth=depth,
                       time=time, transpose=transpose, units=u_units,
                       allow_time_extrapolation=allow_time_extrapolation, **kwargs)
        vfield = Field('V', data_v, lon_v, lat_v, depth=depth,
                       time=time, transpose=transpose, units=v_units,
                       allow_time_extrapolation=allow_time_extrapolation, **kwargs)
        # Create additional data fields
        fields = {}
        for name, data in field_data.items():
            fields[name] = Field(name, data, lon_v, lat_u, depth=depth,
                                 time=time, transpose=transpose,
                                 allow_time_extrapolation=allow_time_extrapolation, **kwargs)
        return cls(ufield, vfield, fields=fields)

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, indices={},
                    mesh='spherical', allow_time_extrapolation=False, **kwargs):
        """Initialises grid data from files using NEMO conventions.

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files.
        :param variables: Dictionary mapping variables to variable
               names in the netCDF file(s).
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
        """

        # Determine unit converters for all fields
        u_units, v_units = unit_converters(mesh)
        units = defaultdict(UnitConverter)
        units.update({'U': u_units, 'V': v_units})
        fields = {}
        for var, name in variables.items():
            # Resolve all matching paths for the current variable
            basepath = path.local(filenames[var])
            paths = [path.local(fp) for fp in glob(str(basepath))]
            if len(paths) == 0:
                raise IOError("Grid files not found: %s" % str(basepath))
            for fp in paths:
                if not fp.exists():
                    raise IOError("Grid file not found: %s" % str(fp))
            dimensions['data'] = name
            fields[var] = Field.from_netcdf(var, dimensions, paths, indices, units=units[var],
                                            allow_time_extrapolation=allow_time_extrapolation, **kwargs)
        u = fields.pop('U')
        v = fields.pop('V')
        return cls(u, v, fields=fields)

    @classmethod
    def from_nemo(cls, basename, uvar='vozocrtx', vvar='vomecrty',
                  indices={}, extra_vars={}, allow_time_extrapolation=False, **kwargs):
        """Initialises grid data from files using NEMO conventions.

        :param basename: Base name of the file(s); may contain
               wildcards to indicate multiple files.
        :param extra_vars: Extra fields to read beyond U and V
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
        """

        dimensions = {'lon': 'nav_lon', 'lat': 'nav_lat',
                      'depth': 'depth', 'time': 'time_counter'}
        extra_vars.update({'U': uvar, 'V': vvar})
        filenames = dict([(v, str(path.local("%s%s.nc" % (basename, v))))
                          for v in extra_vars.keys()])
        return cls.from_netcdf(filenames, indices=indices, variables=extra_vars,
                               dimensions=dimensions, allow_time_extrapolation=allow_time_extrapolation,
                               **kwargs)

    @property
    def fields(self):
        """Returns a list of all the :class:`parcels.field.Field` objects
        associated with this grid"""
        return [v for v in self.__dict__.values() if isinstance(v, Field)]

    def add_field(self, field):
        """Add a :class:`parcels.field.Field` object to the grid

        :param field: :class:`parcels.field.Field` object to be added
        """
        setattr(self, field.name, field)

    def add_constant(self, name, value):
        """Add a constant to the grid. Note that all constants are
        stored as 32-bit floats. While constants can be updated during
        execution in SciPy mode, they can not be updated in JIT mode.

        :param name: Name of the constant
        :param value: Value of the constant (stored as 32-bit float)
        """
        setattr(self, name, value)

    def ParticleSet(self, **kwargs):
        """Wrapper method to define a :class:`parcels.particleset.ParticleSet` on this grid."""
        return ParticleSet(grid=self, **kwargs)

    def add_periodic_halo(self, zonal=False, meridional=False, halosize=5):
        """Add a 'halo' to all :class:`parcels.field.Field` objects on a grid,
        through extending the Field (and lon/lat) by copying a small portion
        of the field on one side of the domain to the other.

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """

        # setting grid constants for use in PeriodicBC kernel. Note using U-Field values
        if zonal:
            self.add_constant('halo_west', self.U.lon[0])
            self.add_constant('halo_east', self.U.lon[-1])
        if meridional:
            self.add_constant('halo_south', self.U.lat[0])
            self.add_constant('halo_north', self.U.lat[-1])

        for attr, value in self.__dict__.iteritems():
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
        """Write grid to NetCDF file using NEMO convention

        :param filename: Basename of the output fileset"""
        print("Generating NEMO grid output with basename: %s" % filename)

        self.U.write(filename, varname='vozocrtx')
        self.V.write(filename, varname='vomecrty')

        for v in self.fields:
            if (v.name is not 'U') and (v.name is not 'V'):
                v.write(filename)
