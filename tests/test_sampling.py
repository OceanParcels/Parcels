from test_moving_eddies import moving_eddies_grid
from argparse import ArgumentParser
from parcels.field import Field
import numpy as np
from parcels.particle import Particle, JITParticle, AdvectionRK4_2D


def updateUserVars(particle, grid, time, dt):
    for var in particle.user_vars.keys():
        if hasattr(grid, var):
            setattr(particle, var, getattr(grid, var)[time, particle.lon, particle.lat])
        else:
            raise AttributeError('The current grid does contain the "%s" field defined in particle.user_vars.' % var)


def CreateHabitatGrid(mu=None, xdim=200, ydim=350):
    """Generate a grid that contains the physical eddy dynamics
    from the original moving eddies test example, overlaying a
    habitat map containing a single maxima and gaussian gradient
    """

    grid = moving_eddies_grid(xdim, ydim)
    depth = grid.U.depth
    time = grid.U.time
    lon = grid.U.lon
    lat = grid.U.lat

    H = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    if mu is not None:
        mu = [2, 46.5]

    def MVNorm(x, y, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
        mu_x = mu[0]
        mu_y = mu[1]
        sigma = np.array(sigma)
        sig_x = sigma[0, 0]
        sig_y = sigma[1, 1]
        sig_xy = sigma[1, 0]

        pd = 1/(2 * np.pi * sig_x * sig_y * np.sqrt(1 - sig_xy))
        pd = pd * np.exp(-1 / (2 * (1 - np.power(sig_xy, 2))) * (
            ((np.power(x - mu_x, 2)) / np.power(sig_x, 2)) +
            ((np.power(y - mu_y, 2)) / np.power(sig_y, 2)) -
            ((2 * sig_xy * (x - mu_x) * (y - mu_y)) / (sig_x * sig_y))))

        return pd

    sig = [[1, 0], [0, 1]]
    for i, x in enumerate(lon):
        for j, y in enumerate(lat):
            H[i, j, :] = MVNorm(x, y, mu, sig)

    H_field = Field('H', H, lon, lat, depth, time, transpose=True)

    # grid.add_field(H_field)
    grid.fields.update({'H': H_field})
    setattr(grid, 'H', H_field)

    return grid


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
    if args.grid is None:
        args.grid = [200, 200]

    grid = CreateHabitatGrid(args.grid[0], args.grid[1])
    grid.write('sampler')

    ParticleClass = JITParticle if args.mode == 'jit' else Particle

    class Sampler(ParticleClass):
        user_vars = {'H': np.float32}

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(Sampler, self).__init__(*args, **kwargs)

    pset = grid.ParticleSet(size=args.particles, pclass=Sampler, start=(0, 45), finish=(4, 47))

    endtime = 25*24
    dt = 800
    output_interval = 12 * dt

    record_user_vars = pset.Kernel(updateUserVars)

    pset.execute(AdvectionRK4_2D + record_user_vars, endtime=endtime, dt=dt,
                 output_file=pset.ParticleFile(name="SamplerParticle"),
                 output_interval=output_interval)
