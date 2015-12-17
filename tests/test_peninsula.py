from parcels import NEMOGrid, Particle, ParticleSet, JITParticle, JITParticleSet
from parcels import CythonParticle, CythonParticleSet
from grid_peninsula import PeninsulaGrid
from argparse import ArgumentParser
import numpy as np
import pytest


def pensinsula_example(grid, npart, mode='cython', degree=3, verbose=False):
    """Example configuration of particle flow around an idealised Peninsula

    :arg filename: Basename of the input grid file set
    :arg npart: Number of particles to intialise"""

    # Determine particle and set classes according to mode
    if mode == 'jit':
        ParticleClass = JITParticle
        PSetClass = JITParticleSet
    elif mode == 'cython':
        ParticleClass = CythonParticle
        PSetClass = CythonParticleSet
    else:
        ParticleClass = Particle
        PSetClass = ParticleSet

    # First, we define a custom Particle class to which we add a
    # custom variable, the initial stream function value p
    class MyParticle(ParticleClass):
        # JIT compilation requires a-priori knowledge of the particle
        # data structure, so we define additional variables here.
        user_vars = [('p', np.float32)]

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(MyParticle, self).__init__(*args, **kwargs)
            self.p = None

        def __repr__(self):
            """Custom print function which overrides the built-in"""
            return "P(%.4f, %.4f)[p=%.5f]" % (self.lon, self.lat, self.p)

    # Initialise particles
    km2deg = 1. / 1.852 / 60
    min_y = grid.U.lat[0] + 3. * km2deg
    max_y = grid.U.lat[-1] - 3. * km2deg
    lon = 3. * km2deg * np.ones(npart)
    lat = np.linspace(min_y, max_y, npart, dtype=np.float)
    pset = PSetClass(npart, grid, pclass=MyParticle, lon=lon, lat=lat)
    for particle in pset:
        particle.p = grid.P.eval(particle.lon, particle.lat)

    if verbose:
        print "Initial particle positions:"
        for p in pset:
            print p

    # Advect the particles for 24h
    time = 86400.
    dt = 36.
    timesteps = int(time / dt)
    pset.advect(timesteps=timesteps, dt=dt)

    if verbose:
        print "Final particle positions:"
        for p in pset:
            p_local = grid.P.eval(p.lon, p.lat)
            print p, "\tP(final)%.5f \tdelta(P): %0.5g" % (p_local, p_local - p.p)

    return np.array([abs(p.p - grid.P.eval(p.lon, p.lat)) for p in pset])


@pytest.mark.parametrize('mode', ['scipy', 'cython', 'jit'])
def test_peninsula_grid(mode):
    """Execute peninsula test from grid generated in memory"""
    grid = PeninsulaGrid(100, 50)
    error = pensinsula_example(grid, 100, mode=mode, degree=1)
    assert(error <= 2.e-4).all()


@pytest.fixture(scope='module')
def gridfile():
    """Generate grid files for peninsula test"""
    filename = 'peninsula'
    grid = PeninsulaGrid(100, 50)
    grid.write(filename)
    return filename


@pytest.mark.parametrize('mode', ['scipy', 'cython', 'jit'])
def test_peninsula_file(gridfile, mode):
    """Open grid files and execute"""
    grid = NEMOGrid.from_file(gridfile)
    error = pensinsula_example(grid, 100, mode=mode, degree=1)
    assert(error <= 2.e-4).all()


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'cython', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=20,
                   help='Number of particles to advect')
    p.add_argument('-d', '--degree', type=int, default=3,
                   help='Degree of spatial interpolation')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    args = p.parse_args()

    # Open grid file set
    grid = NEMOGrid.from_file('peninsula')

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("pensinsula_example(grid, args.particles, mode=args.mode,\
                                   degree=args.degree, verbose=args.verbose)",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        pensinsula_example(grid, args.particles, mode=args.mode,
                           degree=args.degree, verbose=args.verbose)
