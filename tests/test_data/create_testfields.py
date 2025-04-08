import math
import os

import numpy as np
import xarray as xr

from parcels import FieldSet, GridType

N_OCTAVES = 4
PERLIN_RES = (32, 8)
SHAPESCALE = (1, 1)
PERLIN_PERSISTENCE = 0.3
SCALE_FACTOR = 2.0


def generate_testfieldset(xdim: int, ydim: int, zdim: int, tdim: int) -> xr.Dataset:
    # TODO: Update function name and incorporate into testing data strategy
    U = np.ones((tdim, zdim, ydim, xdim), dtype=np.float32)
    V = np.zeros((tdim, zdim, ydim, xdim), dtype=np.float32)
    P = 2.0 * np.ones((tdim, zdim, ydim, xdim), dtype=np.float32)
    return xr.Dataset(
        {
            "U": (("time", "depth", "lat", "lon"), U),
            "V": (("time", "depth", "lat", "lon"), V),
            "P": (("time", "depth", "lat", "lon"), P),
        },
        coords={
            "lon": np.linspace(0.0, 2.0, xdim, dtype=np.float32),
            "lat": np.linspace(0.0, 1.0, ydim, dtype=np.float32),
            "depth": np.linspace(0.0, 0.5, zdim, dtype=np.float32),
            "time": np.linspace(0.0, tdim, tdim, dtype=np.float64),
        },
    )


def generate_perlin_testfield():
    img_shape = (
        int(math.pow(2, N_OCTAVES)) * PERLIN_RES[0] * SHAPESCALE[0],
        int(math.pow(2, N_OCTAVES)) * PERLIN_RES[1] * SHAPESCALE[1],
    )

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(-180.0, 180.0, img_shape[0], dtype=np.float32)
    lat = np.linspace(-90.0, 90.0, img_shape[1], dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.ones(img_shape, dtype=np.float32) * SCALE_FACTOR
    V = np.ones(img_shape, dtype=np.float32) * SCALE_FACTOR
    U = np.transpose(U, (1, 0))
    U = np.expand_dims(U, 0)
    V = np.transpose(V, (1, 0))
    V = np.expand_dims(V, 0)
    data = {"U": U, "V": V}
    dimensions = {"time": time, "lon": lon, "lat": lat}

    fieldset = FieldSet.from_data(data, dimensions, mesh="spherical")
    # fieldset.write("perlinfields")  # can also be used, but then has a ghost depth dimension
    write_simple_2Dt(fieldset.U, os.path.join(os.path.dirname(__file__), "perlinfields"), varname="vozocrtx")
    write_simple_2Dt(fieldset.V, os.path.join(os.path.dirname(__file__), "perlinfields"), varname="vomecrty")


def write_simple_2Dt(field, filename, varname=None):
    """Write a :class:`Field` to a netcdf file

    Parameters
    ----------
    field : parcels.field.Field
        Field to write to file
    filename : str
        Base name of the file to write to
    varname : str, optional
        Name of the variable to write to file. If None, defaults to field.name
    """
    filepath = str(f"{filename}{field.name}.nc")
    if varname is None:
        varname = field.name

    # Create DataArray objects for file I/O
    if field.grid._gtype == GridType.RectilinearZGrid:
        nav_lon = xr.DataArray(
            field.grid.lon + np.zeros((field.grid.ydim, field.grid.xdim), dtype=np.float32),
            coords=[("y", field.grid.lat), ("x", field.grid.lon)],
        )
        nav_lat = xr.DataArray(
            field.grid.lat.reshape(field.grid.ydim, 1) + np.zeros(field.grid.xdim, dtype=np.float32),
            coords=[("y", field.grid.lat), ("x", field.grid.lon)],
        )
    elif field.grid._gtype == GridType.CurvilinearZGrid:
        nav_lon = xr.DataArray(field.grid.lon, coords=[("y", range(field.grid.ydim)), ("x", range(field.grid.xdim))])
        nav_lat = xr.DataArray(field.grid.lat, coords=[("y", range(field.grid.ydim)), ("x", range(field.grid.xdim))])
    else:
        raise NotImplementedError("Field.write only implemented for RectilinearZGrid and CurvilinearZGrid")

    attrs = {"units": "seconds since " + str(field.grid.time_origin)} if field.grid.time_origin.calendar else {}
    time_counter = xr.DataArray(field.grid.time, dims=["time_counter"], attrs=attrs)
    vardata = xr.DataArray(
        field.data.reshape((field.grid.tdim, field.grid.ydim, field.grid.xdim)), dims=["time_counter", "y", "x"]
    )
    # Create xarray Dataset and output to netCDF format
    attrs = {"parcels_mesh": field.grid.mesh}
    dset = xr.Dataset(
        {varname: vardata}, coords={"nav_lon": nav_lon, "nav_lat": nav_lat, "time_counter": time_counter}, attrs=attrs
    )
    dset.to_netcdf(filepath)


if __name__ == "__main__":
    generate_testfieldset(xdim=5, ydim=3, zdim=2, tdim=15)
    generate_perlin_testfield()
