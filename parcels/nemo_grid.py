import numpy as np
from py import path
from netCDF4 import Dataset
from parcels.field import Field


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

    def __init__(self, lon_u, lat_u, lon_v, lat_v, depth, time, U, V):
        """Initialise Grid object from raw data"""
        # Grid dimension arrays
        self.depth = depth
        self.time_counter = time

        # Velocity data
        self.U = Field('U', U, lon_u, lat_u)
        self.V = Field('V', V, lon_v, lat_v)

    @classmethod
    def from_file(cls, filename):
        """Initialises grid data from files using NEMO conventions.

        :param filename: Base name of a set of NEMO files
        """
        dset_u = Dataset('%s_U.nc' % filename, 'r', format="NETCDF4")
        dset_v = Dataset('%s_V.nc' % filename, 'r', format="NETCDF4")

        # Get U, V and flow-specific lat/lon from netCF file
        lon_u = dset_u['nav_lon'][0, :]
        lat_u = dset_u['nav_lat'][:, 0]
        lon_v = dset_v['nav_lon'][0, :]
        lat_v = dset_v['nav_lat'][:, 0]
        depth = np.zeros(1, dtype=np.float32)
        time = np.zeros(1, dtype=np.float32)

        u = dset_u['vozocrtx'][0, 0, :, :]
        v = dset_v['vomecrty'][0, 0, :, :]

        # Hack around the fact that NaN values propagate in SciPy's interpolators
        u[np.isnan(u)] = 0.
        v[np.isnan(v)] = 0.

        return cls(lon_u, lat_u, lon_v, lat_v, depth, time, u, v)

    def eval(self, x, y):
        u = self.U.eval(y, x)
        v = self.V.eval(y, x)
        return u, v

    def add_particle(self, p):
        self._particles.append(p)

    def write(self, filename):
        """Write flow field to NetCDF file using NEMO convention"""
        filepath = path.local(filename)
        print "Generating NEMO grid output:", filepath

        # Generate NEMO-style output for U
        dset_u = Dataset('%s_U.nc' % filepath, 'w', format="NETCDF4")
        dset_u.createDimension('x', len(self.U.lon))
        dset_u.createDimension('y', len(self.U.lat))
        dset_u.createDimension('depthu', self.depth.size)
        dset_u.createDimension('time_counter', None)

        dset_u.createVariable('nav_lon', np.float32, ('y', 'x'))
        dset_u.createVariable('nav_lat', np.float32, ('y', 'x'))
        dset_u.createVariable('depthu', np.float32, ('depthu',))
        dset_u.createVariable('time_counter', np.float64, ('time_counter',))
        dset_u.createVariable('vozocrtx', np.float32, ('time_counter', 'depthu', 'y', 'x'))

        for y in range(len(self.U.lat)):
            dset_u['nav_lon'][y, :] = self.U.lon
        for x in range(len(self.U.lon)):
            dset_u['nav_lat'][:, x] = self.U.lat
        dset_u['depthu'][:] = self.depth
        dset_u['time_counter'][:] = self.time_counter
        dset_u['vozocrtx'][0, 0, :, :] = np.transpose(self.U.data)
        dset_u.close()

        # Generate NEMO-style output for V
        dset_v = Dataset('%s_V.nc' % filepath, 'w', format="NETCDF4")
        dset_v.createDimension('x', len(self.V.lon))
        dset_v.createDimension('y', len(self.V.lat))
        dset_v.createDimension('depthv', self.depth.size)
        dset_v.createDimension('time_counter', None)

        dset_v.createVariable('nav_lon', np.float32, ('y', 'x'))
        dset_v.createVariable('nav_lat', np.float32, ('y', 'x'))
        dset_v.createVariable('depthv', np.float32, ('depthv',))
        dset_v.createVariable('time_counter', np.float64, ('time_counter',))
        dset_v.createVariable('vomecrty', np.float32, ('time_counter', 'depthv', 'y', 'x'))

        for y in range(len(self.V.lat)):
            dset_v['nav_lon'][y, :] = self.V.lon
        for x in range(len(self.V.lon)):
            dset_v['nav_lat'][:, x] = self.V.lat
        dset_v['depthv'][:] = self.depth
        dset_v['time_counter'][:] = self.time_counter
        dset_v['vomecrty'][0, 0, :, :] = np.transpose(self.V.data)
        dset_v.close()
