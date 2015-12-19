import numpy as np
from netCDF4 import Dataset
from parcels.field import Field
from py import path


__all__ = ['NEMOGrid']


class NEMOGrid(object):
    """Grid class used to generate and read NEMO output files

    :param lon_u: Longitude coordinates of the U components
    :param lat_u: Latitude coordinates of the U components
    :param lon_v: Longitude coordinates of the V components
    :param lat_v: Latitude coordinates of the V components
    :param depth: Depth coordinates of the grid
    :param time: Time coordinates of the grid
    :param U: Zonal velocity component
    :param V: Meridional velocity component"""

    # Particle set
    _particles = []

    def __init__(self, lon_u, lat_u, lon_v, lat_v, depth, time,
                 U, V, transpose=True, fields=None):
        """Initialise Grid object from raw data"""
        # Grid dimension arrays
        self.depth = depth
        self.time_counter = time

        # Velocity data
        if transpose:
            # Make a copy of the transposed array to enforce
            # C-contiguous memory layout. This is required
            # for Cython and JIT mode.
            U = np.transpose(U).copy()
            V = np.transpose(V).copy()
        self.U = Field('U', U, lon_u, lat_u, depth=depth, time=time)
        self.V = Field('V', V, lon_v, lat_v, depth=depth, time=time)

        # Additional data fields
        self.fields = fields
        if self.fields is not None:
            for name, data in self.fields.items():
                if transpose:
                    data = np.transpose(data)
                field = Field(name, data, lon_v, lat_u, depth=depth, time=time)
                setattr(self, name, field)

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
        time = np.zeros(1, dtype=np.float32)

        u = dset_u['vozocrtx'][0, 0, :, :]
        v = dset_v['vomecrty'][0, 0, :, :]

        # Detect additional field data
        basedir = filepath_u.dirpath()
        fields = {}
        for fp in basedir.listdir('%s_*.nc' % filename):
            if not fp.samefile(filepath_u) and not fp.samefile(filepath_v):
                # Derive field name, read data and add to fields
                fname = fp.basename.split('.')[0].split('_')[1]
                dset = Dataset(str(fp), 'r', format="NETCDF4")
                fields[fname] = dset[fname][0, 0, :, :]

        return cls(lon_u, lat_u, lon_v, lat_v, depth, time,
                   u, v, transpose=False, fields=fields)

    def eval(self, x, y):
        u = self.U.eval(x, y)
        v = self.V.eval(x, y)
        return u, v

    def add_particle(self, p):
        self._particles.append(p)

    def write(self, filename):
        """Write flow field to NetCDF file using NEMO convention

        :param filename: Basename of the output fileset"""
        print("Generating NEMO grid output with basename:", filename)

        self.U.write(filename, varname='vozocrtx')
        self.V.write(filename, varname='vomecrty')

        for f in self.fields:
            field = getattr(self, f)
            field.write(filename)
