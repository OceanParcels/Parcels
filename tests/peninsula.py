from parcels import NEMOGrid
import numpy as np
from argparse import ArgumentParser


class PeninsulaGrid(NEMOGrid):
    """Grid encapsulating the flow field around an idealised
    peninsula.

    The original test description can be found in Fig. 2.2.3 in:
    North, E. W., Gallego, A., Petitgas, P. (Eds). 2009. Manual of
    recommended practices for modelling physical - biological
    interactions during fish early life.
    ICES Cooperative Research Report No. 295. 111 pp.
    http://archimer.ifremer.fr/doc/00157/26792/24888.pdf"""

    def __init__(self, xdim, ydim):
        """Construct the Peninsula flow field on a NEMO grid.

        :param xdim: Horizontal dimension of the generated grid
        :param xdim: Vertical dimension of the generated grid

        Note that the problem is defined on an A-grid while NEMO
        normally returns C-grids. However, to avoid accuracy
        problems with interpolation from A-grid to C-grid, we
        return NetCDF files that are on an A-grid.
        """
        # Set NEMO grid variables
        self.depth = np.zeros(1, dtype=np.float32)
        self.time_counter = np.zeros(1, dtype=np.float32)

        # Generate the original test setup on A-grid in km
        xoff = 100. / xdim / 2.
        yoff = 50. / ydim / 2.
        La = np.linspace(xoff, 100.-xoff, xdim, dtype=np.float)
        Wa = np.linspace(yoff, 50.-yoff, ydim, dtype=np.float)

        # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
        # surface height) all on A-grid
        self.U = np.zeros((xdim, ydim), dtype=np.float)
        self.V = np.zeros((xdim, ydim), dtype=np.float)
        self.W = np.zeros((xdim, ydim), dtype=np.float)
        self.P = np.zeros((xdim, ydim), dtype=np.float)

        u0 = 1
        x0 = 50.
        R = 0.32 * 50.

        # Create the fields
        for i, x in enumerate(La):
            for j, y in enumerate(Wa):
                self.P[i, j] = u0*R**2*y/((x-x0)**2+y**2)-u0*y
                self.U[i, j] = u0-u0*R**2*((x-x0)**2-y**2)/(((x-x0)**2+y**2)**2)
                self.V[i, j] = -2*u0*R**2*((x-x0)*y)/(((x-x0)**2+y**2)**2)

        # Set land points to NaN
        I = self.P >= 0.
        self.U[I] = np.nan
        self.V[I] = np.nan
        self.W[I] = np.nan

        # Convert from km to lat/lon
        self.lonA = La / 1.852 / 60.
        self.latA = Wa / 1.852 / 60.

        # Set the lon and lat mesh for U and V on c-grid
        self.lon_u = self.lonA
        self.lat_u = self.latA
        self.lon_v = self.lonA
        self.lat_v = self.latA


def main():
    p = ArgumentParser(description="")
    p.add_argument('x', metavar='x', type=int, default=1,
                   help='Number of horizontal grid cells')
    p.add_argument('y', metavar='y', type=int, default=1,
                   help='Number of vertical grid cells')
    p.add_argument('-f', '--filename', default='peninsula',
                   help='Basename for the output grid files')
    args = p.parse_args()

    grid = PeninsulaGrid(args.x, args.y)
    grid.write(args.filename)


if __name__ == "__main__":
    main()
