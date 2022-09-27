from parcels import FieldSet, GridCode
import numpy as np
import math
try:
    from pympler import asizeof
except:
    asizeof = None

from os import path
import xarray as xr
try:
    from parcels.tools import perlin2d as PERLIN
except:
    PERLIN = None
noctaves = 4
perlinres = (32, 8)
shapescale = (1, 1)
perlin_persistence = 0.3
scalefac = 2.0


def generate_testfieldset(xdim, ydim, zdim, tdim):
    lon = np.linspace(0., 2., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.linspace(0., 0.5, zdim, dtype=np.float32)
    time = np.linspace(0., tdim, tdim, dtype=np.float64)
    U = np.ones((xdim, ydim, zdim, tdim), dtype=np.float32)
    V = np.zeros((xdim, ydim, zdim, tdim), dtype=np.float32)
    P = 2.*np.ones((xdim, ydim, zdim, tdim), dtype=np.float32)
    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat, 'depth': depth, 'time': time}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)
    fieldset.write('testfields')


def generate_perlin_testfield():
    img_shape = (int(math.pow(2, noctaves)) * perlinres[0] * shapescale[0], int(math.pow(2, noctaves)) * perlinres[1] * shapescale[1])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(-180.0, 180.0, img_shape[0], dtype=np.float32)
    lat = np.linspace(-90.0, 90.0, img_shape[1], dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)
    time = np.array(time) if not isinstance(time, np.ndarray) else time

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    if PERLIN is not None:
        U = PERLIN.generate_fractal_noise_2d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
        V = PERLIN.generate_fractal_noise_2d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    else:
        U = np.ones(img_shape, dtype=np.float32)*scalefac
        V = np.ones(img_shape, dtype=np.float32)*scalefac
    U = np.transpose(U, (1, 0))
    U = np.expand_dims(U, 0)
    V = np.transpose(V, (1, 0))
    V = np.expand_dims(V, 0)
    data = {'U': U, 'V': V}
    dimensions = {'time': time, 'lon': lon, 'lat': lat}
    if asizeof is not None:
        print("Perlin U-field requires {} bytes of memory.".format(U.size * U.itemsize))
        print("Perlin V-field requires {} bytes of memory.".format(V.size * V.itemsize))
    fieldset = FieldSet.from_data(data, dimensions, mesh='spherical', transpose=False)
    # fieldset.write("perlinfields")  # can also be used, but then has a ghost depth dimension
    write_simple_2Dt(fieldset.U, path.join(path.dirname(__file__), 'perlinfields'), varname='vozocrtx')
    write_simple_2Dt(fieldset.V, path.join(path.dirname(__file__), 'perlinfields'), varname='vomecrty')


def write_simple_2Dt(field, filename, varname=None):
    """Write a :class:`Field` to a netcdf file

    :param filename: Basename of the file
    :param varname: Name of the field, to be appended to the filename"""
    filepath = str('%s%s.nc' % (filename, field.name))
    if varname is None:
        varname = field.name

    # Create DataArray objects for file I/O
    if field.grid.gtype == GridCode.RectilinearZGrid:
        nav_lon = xr.DataArray(field.grid.lon + np.zeros((field.grid.ydim, field.grid.xdim), dtype=np.float32),
                               coords=[('y', field.grid.lat), ('x', field.grid.lon)])
        nav_lat = xr.DataArray(field.grid.lat.reshape(field.grid.ydim, 1) + np.zeros(field.grid.xdim, dtype=np.float32),
                               coords=[('y', field.grid.lat), ('x', field.grid.lon)])
    elif field.grid.gtype == GridCode.CurvilinearZGrid:
        nav_lon = xr.DataArray(field.grid.lon, coords=[('y', range(field.grid.ydim)), ('x', range(field.grid.xdim))])
        nav_lat = xr.DataArray(field.grid.lat, coords=[('y', range(field.grid.ydim)), ('x', range(field.grid.xdim))])
    else:
        raise NotImplementedError('Field.write only implemented for RectilinearZGrid and CurvilinearZGrid')

    attrs = {'units': 'seconds since ' + str(field.grid.time_origin)} if field.grid.time_origin.calendar else {}
    time_counter = xr.DataArray(field.grid.time,
                                dims=['time_counter'],
                                attrs=attrs)
    vardata = xr.DataArray(field.data.reshape((field.grid.tdim, field.grid.ydim, field.grid.xdim)),
                           dims=['time_counter', 'y', 'x'])
    # Create xarray Dataset and output to netCDF format
    attrs = {'parcels_mesh': field.grid.mesh}
    dset = xr.Dataset({varname: vardata}, coords={'nav_lon': nav_lon,
                                                  'nav_lat': nav_lat,
                                                  'time_counter': time_counter}, attrs=attrs)
    dset.to_netcdf(filepath)
    if asizeof is not None:
        mem = 0
        mem += asizeof.asizeof(field)
        mem += asizeof.asizeof(field.data[:])
        mem += asizeof.asizeof(field.grid)
        mem += asizeof.asizeof(vardata)
        mem += asizeof.asizeof(nav_lat)
        mem += asizeof.asizeof(nav_lon)
        mem += asizeof.asizeof(time_counter)
        print("Field '{}' requires {} bytes of memory.".format(field.name, mem))


if __name__ == "__main__":
    generate_testfieldset(xdim=5, ydim=3, zdim=2, tdim=15)
    generate_perlin_testfield()
