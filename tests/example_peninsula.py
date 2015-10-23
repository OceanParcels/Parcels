from parcels import NEMOGrid, Particle, ParticleSet
from argparse import ArgumentParser
import numpy as np


class MyParticle(Particle):
    p = None

    def __repr__(self):
        return "P(%.4f, %.4f)[p=%.5f]" % (self.lon, self.lat, self.p)


def pensinsula_example(filename, npart, degree=3, verbose=False):
    """Example configuration of particle flow around an idealised Peninsula

    :arg filename: Basename of the input grid file set
    :arg npart: Number of particles to intialise"""

    # Open grid file set
    grid = NEMOGrid.from_file(filename)

    # Initialise particles
    pset = ParticleSet(npart, grid)
    km2deg = 1. / 1.852 / 60
    min_y = grid.U.lat[0] + 3. * km2deg
    max_y = grid.U.lat[-1] - 3. * km2deg
    for lat in np.linspace(min_y, max_y, npart, dtype=np.float):
        lon = 3. * km2deg
        particle = MyParticle(lon=lon, lat=lat)
        particle.p = grid.P.eval(lon, lat)
        pset.add_particle(particle)

    if verbose:
        print "Initial particle positions:"
        for p in pset._particles:
            print p

    # Advect the particles for 24h
    time = 86400.
    dt = 36.
    timesteps = int(time / dt)
    pset.advect(timesteps=timesteps, dt=dt)

    if verbose:
        print "Final particle positions:"
        for p in pset._particles:
            p_local = grid.P.eval(p.lon, p.lat)
            print p, "\tP(final)%.5f \tdelta(P): %0.5g" % (p_local, p_local - p.p)

if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('-p', '--particles', type=int, default=20,
                   help='Number of particles to advect')
    p.add_argument('-d', '--degree', type=int, default=3,
                   help='Degree of spatial interpolation')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    args = p.parse_args()

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("pensinsula_example('peninsula', args.particles, degree=args.degree, verbose=args.verbose)",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        pensinsula_example('peninsula', args.particles, degree=args.degree,
                           verbose=args.verbose)
