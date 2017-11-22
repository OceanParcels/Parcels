import argparse
import os
from parcels import Grid
import numpy as np
import math


def moving_eddies_grid(xdim=200, ydim=350):
    """Generate a grid encapsulating the flow field consisting of two
    moving eddies, one moving westward and the other moving northwestward.

    Note that this is not a proper geophysical flow. Rather, a Gaussian eddy
    is moved artificially with uniform velocities. Velocities are calculated
    from geostrophy.
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 8. * 86400., 86400., dtype=np.float64)

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, 4, xdim, dtype=np.float32)
    lat = np.linspace(45, 52, ydim, dtype=np.float32)

    # Grid spacing in m
    def cosd(x):
        return math.cos(math.radians(float(x)))
    dx = (lon[1] - lon[0]) * 1852 * 60 * cosd(lat.mean())
    dy = (lat[1] - lat[0]) * 1852 * 60

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    # Some constants
    corio_0 = 1.e-4  # Coriolis parameter
    h0 = 1  # Max eddy height
    sig = 0.5  # Eddy e-folding decay scale (in degrees)
    g = 10  # Gravitational constant
    eddyspeed = 0.1  # Translational speed in m/s
    dX = eddyspeed * 86400 / dx  # Grid cell movement of eddy max each day
    dY = eddyspeed * 86400 / dy  # Grid cell movement of eddy max each day

    [x, y] = np.mgrid[:lon.size, :lat.size]
    for t in range(time.size):
        hymax_1 = lat.size / 7.
        hxmax_1 = .75 * lon.size - dX * t
        hymax_2 = 3. * lat.size / 7. + dY * t
        hxmax_2 = .75 * lon.size - dX * t

        P[:, :, t] = h0 * np.exp(-(x-hxmax_1)**2/(sig*lon.size/4.)**2
                                 - (y-hymax_1)**2/(sig*lat.size/7.)**2)
        P[:, :, t] += h0 * np.exp(-(x-hxmax_2)**2/(sig*lon.size/4.)**2
                                  - (y-hymax_2)**2/(sig*lat.size/7.)**2)

        V[:-1, :, t] = -np.diff(P[:, :, t], axis=0) / dx / corio_0 * g
        V[-1, :, t] = V[-2, :, t]  # Fill in the last column

        U[:, :-1, t] = np.diff(P[:, :, t], axis=1) / dy / corio_0 * g
        U[:, -1, t] = U[:, -2, t]  # Fill in the last row

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time,
                          field_data={'P': P})


def main(target_path=None, overwrite_files=None):
    """Generate moving eddies example data."""

    # Add command line args
    parser = argparse.ArgumentParser(
        description="Create moving-eddies example.")
    parser.add_argument(
        "target_path",
        help="Where to put the data files?  (This path will be created.)")
    parser.add_argument(
        "-o", action="store_true",
        help="If set, existing files will be overwritten.")
    args = parser.parse_args()

    # Apply args if not already set
    if target_path is None:
        target_path = args.target_path
    if overwrite_files is None:
        overwrite_files = args.o

    # Check for existing files
    if any(os.path.exists(os.path.join(target_path,
                                       'moving_eddies{}.nc'.format(var)))
           for var in ["P", "U", "V"]) and not overwrite_files:
        print("Error: moving_eddies[P,U,V].nc already exist in " +
              "{}.".format(target_path))
        return

    # create fields and write to disk
    target_file = os.path.join(target_path, "moving_eddies")
    grid = moving_eddies_grid()
    grid.write(target_file)


if __name__ == "__main__":
    main()
