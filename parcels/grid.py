from parcels.field import Field, UnitConverter, Geographic, GeographicPolar
from parcels.particleset import ParticleSet
import numpy as np
from py import path
from glob import glob
from collections import defaultdict


__all__ = ['Grid']


def unit_converters(mesh):
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
    """Grid class used to generate and read NEMO output files

    :param U: :class:`Field` for zonal velocity component
    :param V: :class:`Field` for meridional velocity component
    :param depth: Depth coordinates of the grid
    :param time: Time coordinates of the grid
    :param fields: Dictionary of additional fields
    """
    def __init__(self, U, V, depth, time, fields={}):
        self.U = U
        self.V = V
        self.depth = depth
        self.time = time
        self.fields = fields

        # Add additional fields as attributes
        for name, field in fields.items():
            setattr(self, name, field)

    @classmethod
    def from_data(cls, data_u, lon_u, lat_u, data_v, lon_v, lat_v,
                  depth=None, time=None, field_data={}, transpose=True,
                  mesh='spherical', **kwargs):
        """Initialise Grid object from raw data

        :param data_u: Zonal velocity data
        :param lon_u: Longitude coordinates of the U components
        :param lat_u: Latitude coordinates of the U components
        :param data_v: Meridional velocity data
        :param lon_v: Longitude coordinates of the V components
        :param lat_v: Latitude coordinates of the V components
        :param depth: Depth coordinates of the grid
        :param time: Time coordinates of the grid
        :param mesh: String indicating the type of mesh coordinates and
                     units used during velocity interpolation:
                       * sperical (default): Lat and lon in degree, with a
                         correction for zonal velocity U near the poles.
                       * flat: No conversion, lat/lon are assumed to be in m.
        """
        depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        time = np.zeros(1, dtype=np.float64) if time is None else time
        u_units, v_units = unit_converters(mesh)
        # Create velocity fields
        ufield = Field('U', data_u, lon_u, lat_u, depth=depth,
                       time=time, transpose=transpose,
                       units=u_units, **kwargs)
        vfield = Field('V', data_v, lon_v, lat_v, depth=depth,
                       time=time, transpose=transpose,
                       units=v_units, **kwargs)
        # Create additional data fields
        fields = {}
        for name, data in field_data.items():
            fields[name] = Field(name, data, lon_v, lat_u, depth=depth,
                                 time=time, transpose=transpose, **kwargs)
        return cls(ufield, vfield, depth, time, fields=fields)

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions,
                    mesh='spherical', **kwargs):
        """Initialises grid data from files using NEMO conventions.

        :param filenames: Dictionary mapping variables to file(s). The
        filepath may contain wildcards to indicate multiple files.
        :param variabels: Dictionary mapping variables to variable
        names in the netCDF file(s).
        :param dimensions: Dictionary mapping data dimensions (lon,
        lat, depth, time, data) to dimensions in the netCF file(s).
        :param mesh: String indicating the type of mesh coordinates and
                     units used during velocity interpolation:
                       * sperical (default): Lat and lon in degree, with a
                         correction for zonal velocity U near the poles.
                       * flat: No conversion, lat/lon are assumed to be in m.
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
            fields[var] = Field.from_netcdf(var, dimensions, paths,
                                            units=units[var], **kwargs)
        u = fields.pop('U')
        v = fields.pop('V')
        return cls(u, v, u.depth, u.time, fields=fields)

    @classmethod
    def from_nemo(cls, basename, uvar='vozocrtx', vvar='vomecrty',
                  extra_vars={}, **kwargs):
        """Initialises grid data from files using NEMO conventions.

        :param basename: Base name of the file(s); may contain
        wildcards to indicate multiple files.
        """
        dimensions = {'lon': 'nav_lon', 'lat': 'nav_lat',
                      'depth': 'depth', 'time': 'time_counter'}
        extra_vars.update({'U': uvar, 'V': vvar})
        filenames = dict([(v, str(path.local("%s%s.nc" % (basename, v))))
                          for v in extra_vars.keys()])
        return cls.from_netcdf(filenames, variables=extra_vars,
                               dimensions=dimensions, **kwargs)

    def add_field(self, field):
        self.fields.update({field.name: field})
        setattr(self, field.name, field)

    def ParticleSet(self, *args, **kwargs):
        return ParticleSet(*args, grid=self, **kwargs)

    def eval(self, x, y):
        u = self.U.eval(x, y)
        v = self.V.eval(x, y)
        return u, v

    def write(self, filename):
        """Write flow field to NetCDF file using NEMO convention

        :param filename: Basename of the output fileset"""
        print("Generating NEMO grid output with basename: %s" % filename)

        self.U.write(filename, varname='vozocrtx')
        self.V.write(filename, varname='vomecrty')

        for f in self.fields:
            field = getattr(self, f)
            field.write(filename)
