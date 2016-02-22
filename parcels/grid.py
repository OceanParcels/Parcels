from netCDF4 import Dataset
from parcels.field import Field
from parcels.particle import ParticleSet
from py import path
from glob import glob


__all__ = ['Grid']


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
                  depth, time, field_data={}, transpose=True, **kwargs):
        """Initialise Grid object from raw data

        :param data_u: Zonal velocity data
        :param lon_u: Longitude coordinates of the U components
        :param lat_u: Latitude coordinates of the U components
        :param data_v: Meridional velocity data
        :param lon_v: Longitude coordinates of the V components
        :param lat_v: Latitude coordinates of the V components
        :param depth: Depth coordinates of the grid
        :param time: Time coordinates of the grid
        """
        # Create velocity fields
        ufield = Field('U', data_u, lon_u, lat_u, depth=depth,
                       time=time, transpose=transpose, **kwargs)
        vfield = Field('V', data_v, lon_v, lat_v, depth=depth,
                       time=time, transpose=transpose, **kwargs)
        # Create additional data fields
        fields = {}
        for name, data in field_data.items():
            fields[name] = Field(name, data, lon_v, lat_u, depth=depth,
                                 time=time, transpose=transpose, **kwargs)
        return cls(ufield, vfield, depth, time, fields=fields)

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, **kwargs):
        """Initialises grid data from files using NEMO conventions.

        :param filenames: Dictionary mapping variables to file(s). The
        filepath may contain wildcards to indicate multiple files.
        :param variabels: Dictionary mapping variables to variable
        names in the netCDF file(s).
        :param dimensions: Dictionary mapping data dimensions (lon,
        lat, depth, time, data) to dimensions in the netCF file(s).
        """
        fields = {}
        for var, name in variables.items():
            # Resolve all matching paths for the current variable
            basepath = path.local(filenames[var])
            paths = [path.local(fp) for fp in glob(str(basepath))]
            for fp in paths:
                if not fp.exists():
                    raise IOError("Grid file not found: %s" % str(fp))
            dsets = [Dataset(str(fp), 'r', format="NETCDF4") for fp in paths]
            dimensions['data'] = name
            fields[var] = Field.from_netcdf(var, dimensions, dsets, **kwargs)
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
