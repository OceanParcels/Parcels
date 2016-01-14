from parcels import Particle, ParticleSet, JITParticle, JITParticleSet
from parcels import NEMOGrid, ParticleFile, AdvectionRK4
from argparse import ArgumentParser
import numpy as np


pclasses = {'scipy': (Particle, ParticleSet),
            'jit': (JITParticle, JITParticleSet)}


def moving_eddies(grid, npart=2, mode='jit', verbose=False):
    """Configuration of a particle set that follows two moving eddies

    :arg grid: :class NEMOGrid: that defines the flow field
    :arg npart: Number of particles to intialise"""

    # Determine particle and set classes according to mode
    ParticleClass, ParticleSetClass = pclasses[mode]

    lon = 3.3 * np.ones(npart, dtype=np.float)
    lat = np.linspace(46., 47.8, npart, dtype=np.float)
    pset = ParticleSetClass(npart, grid, lon=lon, lat=lat)

    if verbose:
        print("Initial particle positions:")
        for p in pset:
            print(p)

    out = ParticleFile(name="EddyParticle", particleset=pset)
    out.write(pset, 0.)

    # 25 days, with 5min timesteps and hourly output
    hours = 24 * 25
    timesteps = 12
    dt = 300.
    current = 0.
    print("MovingEddies: Advecting %d particles for %d timesteps"
          % (npart, hours * timesteps))
    for _ in range(hours):
        pset.execute(AdvectionRK4, time=current,
                     timesteps=timesteps, dt=dt)
        out.write(pset, current)
        current += timesteps * dt

    if verbose:
        print("Final particle positions:")
        for p in pset:
            print(p)


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
    args = p.parse_args()

    # Open grid file set
    grid = NEMOGrid.from_file('moving_eddies')

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("moving_eddies(grid, args.particles, mode=args.mode, \
                              verbose=args.verbose)",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        moving_eddies(grid, args.particles, mode=args.mode, verbose=args.verbose)
