from parcels import Grid, Particle, JITParticle
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
from argparse import ArgumentParser
import numpy as np
import math
import pytest
from datetime import timedelta as delta


method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def moving_eddies_grid(xdim=200, ydim=350):
    """Generate a grid encapsulating the flow field consisting of two
    moving eddies, one moving westward and the other moving northwestward.

    Note that this is not a proper geophysical flow. Rather, a Gaussian eddy is moved
    artificially with uniform velocities. Velocities are calculated from geostrophy.
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

        P[:, :, t] = h0 * np.exp(-(x-hxmax_1)**2/(sig*lon.size/4.)**2-(y-hymax_1)**2/(sig*lat.size/7.)**2)
        P[:, :, t] += h0 * np.exp(-(x-hxmax_2)**2/(sig*lon.size/4.)**2-(y-hymax_2)**2/(sig*lat.size/7.)**2)

        V[:-1, :, t] = -np.diff(P[:, :, t], axis=0) / dx / corio_0 * g
        V[-1, :, t] = V[-2, :, t]  # Fill in the last column

        U[:, :-1, t] = np.diff(P[:, :, t], axis=1) / dy / corio_0 * g
        U[:, -1, t] = U[:, -2, t]  # Fill in the last row

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time, field_data={'P': P})


def moving_eddies_example(grid, npart=2, mode='jit', verbose=False,
                          method=AdvectionRK4):
    """Configuration of a particle set that follows two moving eddies

    :arg grid: :class Grid: that defines the flow field
    :arg npart: Number of particles to intialise"""

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle
    pset = grid.ParticleSet(size=npart, pclass=ParticleClass,
                            start=(3.3, 46.), finish=(3.3, 47.8))

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execte for 21 days, with 5min timesteps and hourly output
    endtime = delta(days=21)
    print("MovingEddies: Advecting %d particles for %s" % (npart, str(endtime)))
    pset.execute(method, endtime=endtime, dt=delta(minutes=5),
                 output_file=pset.ParticleFile(name="EddyParticle"),
                 interval=delta(hours=1), show_movie=False)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_moving_eddies_fwdbwd(mode, npart=2):
    method = AdvectionRK4
    grid = moving_eddies_grid()

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle

    pset = grid.ParticleSet(size=npart, pclass=ParticleClass,
                            start=(3.3, 46.), finish=(3.3, 47.8))

    # Execte for 14 days, with 30sec timesteps and hourly output
    endtime = delta(days=1)
    dt = delta(minutes=5)
    interval = delta(hours=1)
    print("MovingEddies: Advecting %d particles for %s" % (npart, str(endtime)))
    pset.execute(method, starttime=0, endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="EddyParticlefwd"))

    print("Now running in backward time mode")
    pset.execute(method, starttime=endtime, endtime=0, dt=-dt, interval=-interval,
                 output_file=pset.ParticleFile(name="EddyParticlebwd"))

    assert(pset[0].lon > 3.2 and 45.9 < pset[0].lat < 46.1)
    assert(pset[1].lon > 3.2 and 47.7 < pset[1].lat < 47.9)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_moving_eddies_grid(mode):
    grid = moving_eddies_grid()
    pset = moving_eddies_example(grid, 2, mode=mode)
    assert(pset[0].lon < 0.5 and 46.0 < pset[0].lat < 46.35)
    assert(pset[1].lon < 0.5 and 49.4 < pset[1].lat < 49.8)


@pytest.fixture(scope='module')
def gridfile():
    """Generate grid files for moving_eddies test"""
    filename = 'moving_eddies'
    grid = moving_eddies_grid(200, 350)
    grid.write(filename)
    return filename


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_moving_eddies_file(gridfile, mode):
    grid = Grid.from_nemo(gridfile, extra_vars={'P': 'P'})
    pset = moving_eddies_example(grid, 2, mode=mode)
    assert(pset[0].lon < 0.5 and 46.0 < pset[0].lat < 46.35)
    assert(pset[1].lon < 0.5 and 49.4 < pset[1].lat < 49.8)


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
    p.add_argument('-m', '--method', choices=('RK4', 'EE', 'RK45'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args()
    filename = 'examples/MovingEddies_data/moving_eddies'

    # Generate grid files according to given dimensions
    if args.grid is not None:
        grid = moving_eddies_grid(args.grid[0], args.grid[1])
        grid.write(filename)

    # Open grid files
    grid = Grid.from_nemo(filename, extra_vars={'P': 'P'})

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("moving_eddies_example(grid, args.particles, mode=args.mode, \
                              verbose=args.verbose, method=method[args.method])",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        moving_eddies_example(grid, args.particles, mode=args.mode,
                              verbose=args.verbose, method=method[args.method])
