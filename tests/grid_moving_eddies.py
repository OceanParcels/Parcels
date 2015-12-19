from parcels import NEMOGrid
import numpy as np
import math
from argparse import ArgumentParser


class MovingEddiesGrid(NEMOGrid):
    """Grid encapsulating the flow field consisting of two moving
    eddies, one moving westward and the other moving northwestward.

    The original test description can be found in:
    K. Doos, J. Kjellsson and B. F. Jonsson. 2013
    TRACMASS - A Lagrangian Trajectory Model,
    in Preventive Methods for Coastal Protection,
    T. Soomere and E. Quak (Eds.),
    http://www.springer.com/gb/book/9783319004396"""

    def __init__(self, xdim, ydim):
        """Construct the two-eddy flow field on a NEMO grid
        """
        # Set NEMO grid variables
        depth = np.zeros(1, dtype=np.float32)
        time = np.arange(0., 25., dtype=np.float64)

        # Coordinates of the test grid (on A-grid in deg)
        # lon = np.linspace(0, 4, xdim, dtype=np.float32)
        # lat = np.linspace(45, 52, ydim, dtype=np.float32)
        lon = np.arange(0, 4, 0.02, dtype=np.float32)
        lat = np.arange(45, 52, 0.02, dtype=np.float32)

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
        sig = 30  # Eddy e-folding decay scale (in grid points)
        g = 10  # Gravitational constant
        eddyspeed = 0.1  # Translational speed in m/s
        dX = eddyspeed * 86400 / dx  # Grid cell movement of eddy max each day

        for t in range(time.size):
            hymax_1 = 50
            hxmax_1 = lon.size - 50 - dX * (t-2)
            hymax_2 = 150 + dX * (t-2)
            hxmax_2 = lon.size - 50 - dX * (t-2)
            for x in range(lon.size):
                for y in range(lat.size):
                    P[x, y, t] = h0 * np.exp(-((x-hxmax_1)**2+(y-hymax_1)**2)/sig**2)
                    P[x, y, t] += h0 * np.exp(-((x-hxmax_2)**2+(y-hymax_2)**2)/sig**2)

            for x in range(lon.size-1):
                for y in range(lat.size):
                    V[x, y, t] = -(P[x+1, y, t] - P[x, y, t]) / dx / corio_0 * g
                for y in range(lat.size):
                    V[-1, y, t] = V[-2, y, t]

            for x in range(lon.size):
                for y in range(lat.size-1):
                    U[x, y, t] = (P[x, y+1, t] - P[x, y, t]) / dy / corio_0 * g
                for x in range(lon.size):
                    U[x, -1, t] = V[x, -2, t]

        super(MovingEddiesGrid, self).__init__(lon, lat, lon, lat, depth, time,
                                               U, V, fields={'P': P})


def main():
    np.set_printoptions(linewidth=220)

    p = ArgumentParser(description="")
    p.add_argument('x', metavar='x', type=int, default=1,
                   help='Number of horizontal grid cells')
    p.add_argument('y', metavar='y', type=int, default=1,
                   help='Number of vertical grid cells')
    p.add_argument('-f', '--filename', default='moving_eddies',
                   help='Basename for the output grid files')
    args = p.parse_args()

    grid = MovingEddiesGrid(args.x, args.y)
    grid.write(args.filename)


if __name__ == "__main__":
    main()
