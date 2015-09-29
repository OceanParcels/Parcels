import numpy as np


class PeninsulaGrid(object):
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
        self.xdim = xdim
        self.ydim = ydim

        # Generate lat/lon on A-grid and convert from km to degrees
        self.lonA = np.arange(xdim, dtype=np.int) / 1.852 / 60.
        self.latA = np.arange(ydim, dtype=np.int) / 1.852 / 60.

        # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
        # surface height) all on A-grid
        self.Ua = np.zeros((self.xdim, self.ydim), dtype=np.float)
        self.Va = np.zeros((self.xdim, self.ydim), dtype=np.float)
        self.Wa = np.zeros((self.xdim, self.ydim), dtype=np.float)
        self.P = np.zeros((self.xdim, self.ydim), dtype=np.float)

        u0 = 1
        x0 = round(self.xdim / 2.)
        R = 0.32 * self.ydim

        # Create the fields
        for x in range(1, self.xdim+1):
            for y in range(1, self.ydim+1):
                self.P[x-1, y-1] = u0*R**2*y/((x-x0)**2+y**2)-u0*y
                self.Ua[x-1, y-1] = u0-u0*R**2*((x-x0)**2-y**2)/(((x-x0)**2+y**2)**2)
                self.Va[x-1, y-1] = -2*u0*R**2*((x-x0)*y)/(((x-x0)**2+y**2)**2)

        # Set land points to NaN
        I = self.P >= 0.
        self.Ua[I] = np.nan
        self.Va[I] = np.nan
        self.Wa[I] = np.nan
