from parcels import Grid, Particle, JITParticle
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
from argparse import ArgumentParser
import numpy as np
import math
import pytest


method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def ground_truth(x_0, y_0, time, dt, pset, output_file):
    f = 0.0001  # Coriolis frequency
    u_0 = 0.3
    u_g = 0.04
    gamma = 1 / (86400 * 2.89)
    gamma_g = 1 / (86400 * 28.9)
    for t in range(dt, time+1, dt):
        # Circular eddy
        if 0:
            pset[0].lat = y_0 - u_0 / f * (1 - math.cos(f * t)) / 1852. / 60.
            pset[0].lon = x_0 + u_0 / f * math.sin(f * t) / 1852. / 60. /\
                math.cos(pset[0].lat*math.pi/180)
        # Moving circular eddy
        if 0:
            pset[0].lat = y_0 - (u_0 - u_g) / f * (1 - math.cos(f * t)) / 1852. / 60.
            pset[0].lon = x_0 + (u_g * t + (u_0 - u_g) / f * math.sin(f * t)) /\
                1852. / 60. / math.cos(pset[0].lat*math.pi/180)
        # Moving decaying eddy
        if 1:
            pset[0].lat = y_0 - ((u_0 - u_g) * f / (f ** 2 + gamma ** 2) *
                                 (1 - np.exp(-gamma * t) * (np.cos(f * t) +
                                  gamma / f * np.sin(f * t)))) / 1852. / 60.
            pset[0].lon = x_0 + (u_g / gamma_g * (1 - np.exp(-gamma_g * t)) +
                                 (u_0 - u_g) * f / (f ** 2 + gamma ** 2) *
                                 (gamma / f + np.exp(-gamma * t) * (math.sin(f
                                  * t) - gamma / f * math.cos(f * t)))) / 1852.\
                / 60. / math.cos(pset[0].lat*math.pi/180)
        output_file.write(pset, t)


def analytical_eddies_grid(xdim=20, ydim=20):
    """Generate a grid encapsulating the flow field describing a stationary
    eddy, an eddy moving eastward or an decaying eddy moving eastward.

    The original test description can be found in: N. Fabbroni 2009 Numerical
    simulations of passive tracers dispersion in the sea
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 20. * 86400., 150., dtype=np.float64)

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, 4, xdim, dtype=np.float32)
    lat = np.linspace(40, 50, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    # Some constants
    f = 1.e-4
    u_0 = 0.3
    u_g = 0.04
    gamma = 1/(86400. * 2.89)
    gamma_g = 1/(86400. * 28.9)

    [x, y] = np.mgrid[:lon.size, :lat.size]
    for t in range(time.size):
        # Circular eddy
        if 0:
            U[:, :, t] = u_0 * math.cos(f * time[t])
            V[:, :, t] = -u_0 * math.sin(f * time[t])
        # Moving circular eddy
        if 0:
            U[:, :, t] = u_g + (u_0 - u_g) * math.cos(f * time[t])
            V[:, :, t] = -(u_0 - u_g) * math.sin(f * time[t])
        # Moving decaying eddy
        if 1:
            U[:, :, t] = u_g * np.exp(-gamma_g * time[t]) + (u_0 - u_g) *\
                np.exp(-gamma * time[t]) * math.cos(f * time[t])
            V[:, :, t] = -(u_0 - u_g) * np.exp(-gamma * time[t]) *\
                math.sin(f * time[t])

    return Grid.from_data(U, lon, lat, V, lon, lat,
                          depth, time, field_data={'P': P})


def analytical_eddies_example(grid, npart=1, mode='jit', verbose=False,
                              method=AdvectionRK4):
    """Configuration of a particle set that follows the eddy

    :arg grid: :class NEMOGrid: that defines the flow field
    :arg npart: Number of particles to intialise"""

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle
    pset = grid.ParticleSet(size=npart, pclass=ParticleClass,
                            start=(1., 45.), finish=(1., 45.))

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execute for 3 days, with 5min timesteps and hourly output
    hours = 3*24
    substeps = 1
    dt = 300

    if method == AdvectionRK45:
        for particle in pset:
            particle.time = 0.
            particle.dt = dt
        tol = 1e-10
        print("Analytical: Advecting %d particles with adaptive timesteps"
              % (npart))
        pset.execute(method, timesteps=hours*substeps*3600/dt, dt=dt,
                     output_file=pset.ParticleFile(name="AnalyticalParticle" +
                                                   method.__name__),
                     output_steps=substeps, tol=tol)
    else:
        print("Analytical: Advecting %d particles for %d timesteps"
              % (npart, hours * substeps * 3600 / dt))
        pset.execute(method, timesteps=hours*substeps*3600/dt, dt=dt,
                     output_file=pset.ParticleFile(name="AnalyticalParticle" +
                                                   method.__name__),
                     output_steps=substeps)

    # Analytical solution
    if 1:
        pset_a = grid.ParticleSet(size=npart, pclass=ParticleClass,
                                  start=(1., 45.), finish=(1., 45.))
        ground_truth(1., 45., hours*substeps*3600, dt, pset_a,
                     output_file=pset_a.ParticleFile(name="GroundTruthParticle"))

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_analytic_eddies_grid(mode):
    grid = analytical_eddies_grid()
    pset = analytical_eddies_example(grid, 1, mode=mode)
    assert(1.12 < pset[0].lon < 1.14 and 44.98 < pset[0].lat < 44.99)


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
    p.add_argument('-p', '--particles', type=int, default=1,
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
    filename = 'analytical_eddies'

    # Generate grid files according to given dimensions
    if args.grid is not None:
        grid = analytical_eddies_grid(args.grid[0], args.grid[1])
        grid.write(filename)

    # Open grid files
    grid = Grid.from_nemo(filename)

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("analytical_eddies_example(grid, args.particles, mode=args.mode, \
                verbose=args.verbose)", globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        analytical_eddies_example(grid, args.particles, mode=args.mode,
                                  verbose=args.verbose, method=method[args.method])
