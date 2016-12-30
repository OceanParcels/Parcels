from parcels import *
import numpy as np
from argparse import ArgumentParser


def CreateArbitraryField(grid):
    """Generate a grid that contains the physical eddy dynamics
    from the original moving eddies test example, overlaying a
    habitat map containing a single maxima and gaussian gradient
    """
    depth = grid.U.depth
    time = grid.U.time

    # Coordinates of the test grid (on A-grid in deg)
    lon = grid.U.lon
    lat = grid.U.lat

    K = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    K = np.random.uniform(0, 1., size=np.shape(K))

    return Field('K', K, lon, lat, depth, time, transpose=True)


def CreateDummyUV(xdim=200, ydim=200):
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 100000., 100000/2., dtype=np.float64)

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, 0.1, xdim, dtype=np.float32)
    lat = np.linspace(0, 0.1, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    ufield = Field('U', U, lon, lat, depth=depth,
                   time=time, transpose=True)
    vfield = Field('V', V, lon, lat, depth=depth,
                   time=time, transpose=True)

    return {'U': ufield, 'V': vfield}


def CreateStartField(lon, lat):
    time = np.arange(0., 100000, 100000/2., dtype=np.float64)

    data = np.ones((lon.size, lat.size, time.size), dtype=np.float32)

    return data


def TestSample(particle, grid, time, dt):
    if grid.K[time, particle.lon, particle.lat] < 0:
        print("Field value < zero! At [%s,%s] K = %s" % (particle.lon, particle.lat, grid.K[time, particle.lon, particle.lat]))


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=5000,
                   help='Number of particles to advect')
    p.add_argument('-t', '--timesteps', type=int, default=1,
                   help='Timesteps of one second to run simulation over')
    p.add_argument('-o', '--output', type=str, default='fieldsampletest',
                   help='Output filename')
    args = p.parse_args()

    # Generate grid files according to given dimensions
    gridx = 100
    gridy = 100
    print('Generating grid')
    forcing_fields = CreateDummyUV(gridx, gridy)
    grid = Grid(forcing_fields['U'], forcing_fields['V'], forcing_fields['U'].depth,
                forcing_fields['U'].time, fields=forcing_fields)
    print('Creating K Field')
    grid.add_field(CreateArbitraryField(grid))
    # Evenly spread starting distribution
    print('Creating start field')
    Start_Field = Field('Start', CreateStartField(grid.U.lon, grid.U.lat),
                        grid.U.lon, grid.U.lat, depth=grid.U.depth, time=grid.U.time, transpose=True)
    timestep = 500
    print('Timestep = %s' % timestep)
    steps = args.timesteps

    print("Minimum K = %s" % np.min(grid.K.data))
    print("Maximum K = %s" % np.max(grid.K.data))

    ParticleClass = JITParticle if args.mode == 'jit' else Particle

    pset = grid.ParticleSet(size=args.particles, pclass=ParticleClass, start_field=Start_Field)

    sample = pset.Kernel(TestSample)

    pset.execute(sample, endtime=grid.U.time[0]+timestep*steps, dt=timestep,
                 output_file=pset.ParticleFile(name=args.output+'_resultsF'),
                 output_interval=timestep)
