from parcels import NEMOGrid
import numpy as np
from argparse import ArgumentParser


class PeninsulaGrid(NEMOGrid):
    """Grid encapsulating the flow field around an idealised
    peninsula.

    The original test description can be found in Fig. 2 in:
    C. B. Paris, J. Helgers, E. van Sebille, A. Srinivasan,
    "Connectivity Modeling System: A probabilistic modeling tool for
    the multi-scale tracking of biotic and abiotic variability in the
    ocean",
    Environmental Modelling & Software, Volume 42,
    April 2013, Pages 47-54, ISSN 1364-8152,
    http://dx.doi.org/10.1016/j.envsoft.2012.12.006.
    """

    def __init__(self, xdim, ydim):
        """Construct the Peninsula flow field on a C-grid.

        :param xdim: Horizontal dimension of the generated C-grid
        :param xdim: Vertical dimension of the generated C-grid

        Note that an A-grid version of the flow field is generated
        first and then converted to the C-grid.
        """
        # Set NEMO grid variables
        self.x = xdim
        self.y = ydim
        self.depth = np.zeros(1, dtype=np.float32)
        self.time_counter = np.zeros(1, dtype=np.float32)

        # Generate lat/lon on A-grid and convert from km to degrees
        self.lonA = np.arange(self.x, dtype=np.int) / 1.852 / 60.
        self.latA = np.arange(self.y, dtype=np.int) / 1.852 / 60.

        # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
        # surface height) all on A-grid
        self.Ua = np.zeros((self.x, self.y), dtype=np.float)
        self.Va = np.zeros((self.x, self.y), dtype=np.float)
        self.Wa = np.zeros((self.x, self.y), dtype=np.float)
        self.P = np.zeros((self.x, self.y), dtype=np.float)

        u0 = 1
        x0 = round(self.x / 2.)
        R = 0.32 * self.y

        # Create the fields
        for x in range(1, self.x+1):
            for y in range(1, self.y+1):
                self.P[x-1, y-1] = u0*R**2*y/((x-x0)**2+y**2)-u0*y
                self.Ua[x-1, y-1] = u0-u0*R**2*((x-x0)**2-y**2)/(((x-x0)**2+y**2)**2)
                self.Va[x-1, y-1] = -2*u0*R**2*((x-x0)*y)/(((x-x0)**2+y**2)**2)

        # Set land points to NaN
        I = self.P >= 0.
        self.Ua[I] = np.nan
        self.Va[I] = np.nan
        self.Wa[I] = np.nan

        # Set the lon and lat mesh for U and V on c-grid
        self.lon_u = (self.lonA[1:] + self.lonA[:-1]) / 2.
        self.lat_u = self.latA
        self.lon_v = self.lonA
        self.lat_v = (self.latA[1:] + self.latA[:-1]) / 2.

        # Create c-grid fields
        self.U = np.zeros((self.x-1, self.y), dtype=np.float)
        self.V = np.zeros((self.x, self.y-1), dtype=np.float)

        # Set U on C-grid
        for x in range(self.x-1):
            for y in range(self.y):
                self.U[x, y] = (self.Ua[x+1, y] + self.Ua[x, y]) / 2.

        # Set Vc
        for x in range(self.x):
            for y in range(self.y-1):
                self.V[x, y] = (self.Va[x, y+1] + self.Va[x, y]) / 2.


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
