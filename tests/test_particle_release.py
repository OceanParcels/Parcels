from test_moving_eddies import moving_eddies_grid
from argparse import ArgumentParser
from parcels.field import Field
import numpy as np
from parcels.particle import Particle, JITParticle, AdvectionRK4_2D
from datetime import timedelta as delta


def CreateInitialPositionField(grid):
    depth = grid.U.depth
    time = grid.U.time[0]
    lon = grid.U.lon
    lat = grid.U.lat
    Num = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    for i, x in enumerate(lon):
        for j, y in enumerate(lat):
            if i == j:
                Num[i, j, :] = 1

    return Field('Start', Num, lon, lat, depth, time, transpose=True)


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int, nargs=2, default=None,
                   help='Generate grid file with given dimensions')
    p.add_argument('-m', '--method', choices=('RK4', 'EE'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args()

    # Generate grid files according to given dimensions
    if args.grid is not None:
        grid = moving_eddies_grid(args.grid[0], args.grid[1])

    grid.add_field(CreateInitialPositionField(grid))

    ParticleClass = JITParticle if args.mode == 'jit' else Particle

    print(grid.fields)

    pset = grid.ParticleSet(size=args.particles, pclass=ParticleClass,
                            start_field=grid.Start)

    dt = delta(seconds=800)
    pset.execute(AdvectionRK4_2D, endtime=delta(days=25), dt=dt,
                 output_file=pset.ParticleFile(name="ReleaseTestParticle"),
                 output_interval=12 * dt)
