import numpy as np
from netCDF4 import Dataset
from parcels.field import Field
from py import path


__all__ = ['NEMOGrid']


class NEMOGrid(object):
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
                  depth, time, field_data={}, transpose=True):
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
        ufield = Field('U', data_u, lon_u, lat_u, depth=depth, time=time, transpose=transpose)
        vfield = Field('V', data_v, lon_v, lat_v, depth=depth, time=time, transpose=transpose)
        # Create additional data fields
        fields = {}
        for name, data in field_data.items():
            fields[name] = Field(name, data, lon_v, lat_u, depth=depth,
                                 time=time, transpose=transpose)
        return cls(ufield, vfield, depth, time, fields=fields)

    @classmethod
    def from_file(cls, filename):
        """Initialises grid data from files using NEMO conventions.

        :param filename: Base name of a set of NEMO files
        """
        filepath_u = path.local("%s_U.nc" % filename)
        filepath_v = path.local("%s_V.nc" % filename)
        if not filepath_u.exists():
            raise IOError("Grid file not found: %s" % filepath_u)
        if not path.local(filepath_v).exists():
            raise IOError("Grid file not found: %s" % filepath_v)
        dset_u = Dataset(str(filepath_u), 'r', format="NETCDF4")
        dset_v = Dataset(str(filepath_v), 'r', format="NETCDF4")

        # Get U, V and flow-specific lat/lon from netCF file
        lon_u = dset_u['nav_lon'][0, :]
        lat_u = dset_u['nav_lat'][:, 0]
        lon_v = dset_v['nav_lon'][0, :]
        lat_v = dset_v['nav_lat'][:, 0]
        depth = np.zeros(1, dtype=np.float32)
        time = dset_v['time_counter'][:]

        u = dset_u['vozocrtx'][:, 0, :, :]
        v = dset_v['vomecrty'][:, 0, :, :]

        # Detect additional field data
        basedir = filepath_u.dirpath()
        fields = {}
        for fp in basedir.listdir('%s_*.nc' % filename):
            if not fp.samefile(filepath_u) and not fp.samefile(filepath_v):
                # Derive field name, read data and add to fields
                fname = fp.basename.split('.')[0].split('_')[-1]
                dset = Dataset(str(fp), 'r', format="NETCDF4")
                fields[fname] = dset[fname][:, 0, :, :]

        return cls.from_data(u, lon_u, lat_u, v, lon_v, lat_v, depth, time,
                             transpose=False, field_data=fields)

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
