from parcels import NEMOGrid, Particle
from argparse import ArgumentParser


def pensinsula_example(filename, npart, degree=3):
    """Example configuration of particle flow around an idealised Peninsula

    :arg filename: Basename of the input grid file set
    :arg npart: Number of particles to intialise"""

    # Open grid file set
    grid = NEMOGrid(filename, degree=degree)

    # Initialise particles
    for p in range(npart):
        lat = p * grid.lat_u.valid_max / npart + 0.45 / 1.852 / 60.
        grid.add_particle(Particle(lon=3 / 1.852 / 60., lat=lat))

    print "Initial particle positions:"
    for p in grid._particles:
        print p

    # Advect the particles for 24h
    time = 86400.
    dt = 36.
    ntimesteps = int(time / dt)
    for t in range(ntimesteps):
        for p in grid._particles:
            p.advect_rk4(grid, dt)

    print "Final particle positions:"
    for p in grid._particles:
        print p

if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('-p', '--particles', type=int, default=20,
                   help='Number of particles to advect')
    p.add_argument('-d', '--degree', type=int, default=3,
                   help='Degree of spatial interpolation')
    args = p.parse_args()
    pensinsula_example('peninsula', args.particles, degree=args.degree)
