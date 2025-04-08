import numpy as np
import xarray as xr

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


def generate_perlin_testfield() -> xr.Dataset:
    # TODO: Update function name and incorporate into testing data strategy
    xdim = 512
    ydim = 128
    tdim = 1

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(-180.0, 180.0, xdim, dtype=np.float32)
    lat = np.linspace(-90.0, 90.0, ydim, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.random.rand(tdim, ydim, xdim) * SCALE_FACTOR
    V = np.random.rand(tdim, ydim, xdim) * SCALE_FACTOR
    U = U.astype(np.float32)
    V = V.astype(np.float32)

    nav_lon, nav_lat = np.meshgrid(lon, lat)

    return xr.Dataset(
        {"U": (("time", "lat", "lon"), U), "V": (("time", "lat", "lon"), V)},
        coords={
            "time": time,
            "lon": lon,
            "lat": lat,
            "nav_lon": (("lat", "lon"), nav_lon),
            "nav_lat": (("lat", "lon"), nav_lat),
        },
    )


if __name__ == "__main__":
    generate_testfieldset(xdim=5, ydim=3, zdim=2, tdim=15)
    generate_perlin_testfield()
