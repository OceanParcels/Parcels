from parcels.field import Field, UnitConverter, Geographic, GeographicPolar
from parcels.loggers import logger
import numpy as np
from os import path
from glob import glob
from copy import deepcopy
from collections import defaultdict


__all__ = ['FieldSet']


def unit_converters(mesh):
    """Helper function that assigns :class:`UnitConverter` objects to
    :class:`Field` objects on :class:`FieldSet`

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


class FieldSet(object):
    """FieldSet class that holds hydrodynamic data needed to execute particles

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
    def from_data(cls, data, dimensions, transpose=True, mesh='spherical',
                  allow_time_extrapolation=True, time_periodic=False, **kwargs):
        """Initialise FieldSet object from raw data

        :param data: Dictionary mapping field names to numpy arrays.
               Note that at least a 'U' and 'V' numpy array need to be given
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
        """

        u_units, v_units = unit_converters(mesh)
        units = defaultdict(UnitConverter)
        units.update({'U': u_units, 'V': v_units})
        fields = {}
        for name, datafld in data.items():
            # Use dimensions[name] if dimensions is a dict of dicts
            dims = dimensions[name] if name in dimensions else dimensions

            lon = dims['lon']
            lat = dims['lat']
            depth = np.zeros(1, dtype=np.float32) if 'depth' not in dims else dims['depth']
            time = np.zeros(1, dtype=np.float64) if 'time' not in dims else dims['time']

            fields[name] = Field(name, datafld, lon, lat, depth=depth,
                                 time=time, transpose=transpose, units=units[name],
                                 allow_time_extrapolation=allow_time_extrapolation, time_periodic=time_periodic, **kwargs)
        u = fields.pop('U')
        v = fields.pop('V')
        return cls(u, v, fields=fields)

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, indices={},
                    mesh='spherical', allow_time_extrapolation=False, **kwargs):
        """Initialises FieldSet data from files using NEMO conventions.

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
        """

        # Determine unit converters for all fields
        u_units, v_units = unit_converters(mesh)
        units = defaultdict(UnitConverter)
        units.update({'U': u_units, 'V': v_units})
        fields = {}
        for var, name in variables.items():
            # Resolve all matching paths for the current variable
            if isinstance(filenames[var], list):
                paths = filenames[var]
            else:
                paths = glob(str(filenames[var]))
            if len(paths) == 0:
                raise IOError("FieldSet files not found: %s" % str(filenames[var]))
            for fp in paths:
                if not path.exists(fp):
                    raise IOError("FieldSet file not found: %s" % str(fp))

            # Use dimensions[var] and indices[var] if either of them is a dict of dicts
            dims = dimensions[var] if var in dimensions else dimensions
            dims['data'] = name
            inds = indices[var] if var in indices else indices

            fields[var] = Field.from_netcdf(var, dims, paths, inds, units=units[var],
                                            allow_time_extrapolation=allow_time_extrapolation, **kwargs)
        u = fields.pop('U')
        v = fields.pop('V')
        return cls(u, v, fields=fields)

    @classmethod
    def from_nemo(cls, basename, uvar='vozocrtx', vvar='vomecrty',
                  indices={}, extra_fields={}, allow_time_extrapolation=False, **kwargs):
        """Initialises FieldSet data from files using NEMO conventions.

        :param basename: Base name of the file(s); may contain
               wildcards to indicate multiple files.
        :param extra_fields: Extra fields to read beyond U and V
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
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
                               **kwargs)

    @property
    def fields(self):
        """Returns a list of all the :class:`parcels.field.Field` objects
        associated with this FieldSet"""
        return [v for v in self.__dict__.values() if isinstance(v, Field)]

    def add_field(self, field):
        """Add a :class:`parcels.field.Field` object to the FieldSet

        :param field: :class:`parcels.field.Field` object to be added
        """
        setattr(self, field.name, field)

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

        for vnew in fieldset_new.fields:
            v = getattr(self, vnew.name)
            v.advancetime(vnew)
