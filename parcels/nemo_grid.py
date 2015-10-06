import numpy as np
from py import path
from netCDF4 import Dataset


__all__ = ['NEMOGrid']


class NEMOGrid(object):
    """Grid class used to generate and read NEMO output files"""

    # Dimension sizes
    x = 0
    y = 0
    depth = None
    time_counter = None

    # Variable data arrays
    lon_u = None
    lat_u = None
    lon_v = None
    lat_v = None

    # Flow field data
    U = None
    V = None

    # Particle set
    _particles = []

    def add_particle(self, p):
        self._particles.append(p)

    def write(self, filename):
        """Write flow field to NetCDF file using NEMO convention"""
        filepath = path.local(filename)
        print "Generating NEMO grid output:", filepath

        # Generate NEMO-style output for U
        dset_u = Dataset('%s_U' % filepath, 'w', format="NETCDF4")
        dset_u.createDimension('x', self.x-1)
        dset_u.createDimension('y', self.y)
        dset_u.createDimension('depthu', self.depth.size)
        dset_u.createDimension('time_counter', None)

        dset_u.createVariable('nav_lon', np.float32, ('x', 'y'))
        dset_u.createVariable('nav_lat', np.float32, ('x', 'y'))
        dset_u.createVariable('depthu', np.float32, ('depthu',))
        dset_u.createVariable('time_counter', np.float64, ('time_counter',))
        dset_u.createVariable('vozocrtx', np.float32, ('time_counter', 'depthu', 'x', 'y'))

        for y in range(self.y):
            dset_u['nav_lon'][:, y] = self.lon_u
        dset_u['nav_lon'].valid_min = self.lon_u[0]
        dset_u['nav_lon'].valid_max = self.lon_u[-1]
        for x in range(self.x-1):
            dset_u['nav_lat'][x, :] = self.lat_u
        dset_u['nav_lat'].valid_min = self.lat_u[0]
        dset_u['nav_lat'].valid_max = self.lat_u[-1]
        dset_u['depthu'][:] = self.depth
        dset_u['time_counter'][:] = self.time_counter
        dset_u['vozocrtx'][0, 0, :, :] = self.U
        dset_u.close()

        # Generate NEMO-style output for V
        dset_v = Dataset('%s_V' % filepath, 'w', format="NETCDF4")
        dset_v.createDimension('x', self.x)
        dset_v.createDimension('y', self.y-1)
        dset_v.createDimension('depthv', self.depth.size)
        dset_v.createDimension('time_counter', None)

        dset_v.createVariable('nav_lon', np.float32, ('x', 'y'))
        dset_v.createVariable('nav_lat', np.float32, ('x', 'y'))
        dset_v.createVariable('depthv', np.float32, ('depthv',))
        dset_v.createVariable('time_counter', np.float64, ('time_counter',))
        dset_v.createVariable('vomecrty', np.float32, ('time_counter', 'depthv', 'x', 'y'))

        for y in range(self.y-1):
            dset_v['nav_lon'][:, y] = self.lon_v
        dset_v['nav_lon'].valid_min = self.lon_u[0]
        dset_v['nav_lon'].valid_max = self.lon_u[-1]
        for x in range(self.x):
            dset_v['nav_lat'][x, :] = self.lat_v
        dset_v['nav_lat'].valid_min = self.lat_u[0]
        dset_v['nav_lat'].valid_max = self.lat_u[-1]
        dset_v['depthv'][:] = self.depth
        dset_v['time_counter'][:] = self.time_counter
        dset_v['vomecrty'][0, 0, :, :] = self.V
        dset_v.close()
