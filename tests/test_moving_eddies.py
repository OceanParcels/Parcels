from parcels import Grid, Particle, JITParticle, AdvectionRK4
from argparse import ArgumentParser
import numpy as np
import math
import pytest


def moving_eddies_grid(xdim=200, ydim=350):
    """Generate a grid encapsulating the flow field consisting of two
    moving eddies, one moving westward and the other moving northwestward.

    The original test description can be found in: K. Doos,
    J. Kjellsson and B. F. Jonsson. 2013 TRACMASS - A Lagrangian
    Trajectory Model, in Preventive Methods for Coastal Protection,
    T. Soomere and E. Quak (Eds.),
    http://www.springer.com/gb/book/9783319004396
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 25. * 86400., 86400., dtype=np.float64)

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
    sig = 30  # Eddy e-folding decay scale (in grid points)
    g = 10  # Gravitational constant
    eddyspeed = 0.1  # Translational speed in m/s
    dX = eddyspeed * 86400 / dx  # Grid cell movement of eddy max each day

    [x, y] = np.mgrid[:lon.size, :lat.size]
    for t in range(time.size):
        hymax_1 = int(lat.size / 7)
        hxmax_1 = int(.75 * lon.size) - dX * (t-2)
        hymax_2 = int(3 * lat.size / 7) + dX * (t-2)
        hxmax_2 = int(.75 * lon.size) - dX * (t-2)

        P[:, :, t] = h0 * np.exp(-((x-hxmax_1)**2+(y-hymax_1)**2)/sig**2)
        P[:, :, t] += h0 * np.exp(-((x-hxmax_2)**2+(y-hymax_2)**2)/sig**2)

        V[:-1, :, t] = -np.diff(P[:, :, t], axis=0) / dx / corio_0 * g
        V[-1, :, t] = V[-2, :, t]  # Fill in the last column

        U[:, :-1, t] = np.diff(P[:, :, t], axis=1) / dy / corio_0 * g
        V[:, -1, t] = U[:, -2, t]  # Fill in the last row

    return Grid.from_data(U, lon, lat, V, lon, lat,
                          depth, time, field_data={'P': P})


def moving_eddies_example(grid, npart=2, mode='jit', verbose=False):
    """Configuration of a particle set that follows two moving eddies

    :arg grid: :class Grid: that defines the flow field
    :arg npart: Number of particles to intialise"""

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle

    pset = grid.ParticleSet(size=npart, pclass=ParticleClass,
                            start=(3.3, 46.), finish=(3.3, 47.8))

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execte for 25 days, with 5min timesteps and hourly output
    hours = 25*24
    substeps = 12
    print("MovingEddies: Advecting %d particles for %d timesteps"
          % (npart, hours * substeps))
    pset.execute(AdvectionRK4, timesteps=hours*substeps, dt=300.,
                 output_file=pset.ParticleFile(name="EddyParticle"),
                 output_steps=substeps)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_moving_eddies_grid(mode):
    grid = moving_eddies_grid()
    pset = moving_eddies_example(grid, 2, mode=mode)
    assert(pset[0].lon < 0.5 and 45.8 < pset[0].lat < 46.15)
    assert(pset[1].lon < 0.5 and 50.4 < pset[1].lat < 50.7)


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=2,
                   help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int, nargs=2, default=None,
                   help='Generate grid file with given dimensions')
    args = p.parse_args()
    filename = 'moving_eddies'

    # Generate grid files according to given dimensions
    if args.grid is not None:
        grid = moving_eddies_grid(args.grid[0], args.grid[1])
        grid.write(filename)

    # Open grid files
    grid = Grid.from_nemo(filename)

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("moving_eddies_example(grid, args.particles, mode=args.mode, \
                              verbose=args.verbose)",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        moving_eddies_example(grid, args.particles, mode=args.mode,
                              verbose=args.verbose)
