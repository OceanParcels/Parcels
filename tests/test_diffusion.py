from parcels import *
import numpy as np
import math
from argparse import ArgumentParser


def CreateDiffusionField(grid, mu=None):
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

    if mu is None:
        mu = [np.mean(lon), np.mean(lat)]
    print(mu)

    def MVNorm(x, y, mu=[0, 0], sigma=[[100, 0], [0, 100]]):
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

    sig = [[0.006, 0], [0, 0.006]]
    for i, x in enumerate(lon):
        for j, y in enumerate(lat):
            K[i, j, :] = MVNorm(x, y, mu, sig)

    # Scale and invert (to make bowl of low diffusivity)
    K /= np.max(K) - np.min(K)
    K -= np.min(K)-0.2
    K = 1/K

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

    data = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    data[(round(lon.size/2)-round(lon.size/10)):-(round(lon.size/2)-round(lon.size/10)),
         (round(lat.size/2)-round(lat.size/10)):-(round(lat.size/2)-round(lat.size/10)), :] = 1

    return data


def RK4(fieldx, fieldy, lon, lat, time, dt):
    f_lat = dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(lat*math.pi/180)
    u1 = fieldx[time, lon, lat]
    v1 = fieldy[time, lon, lat]
    lon1, lat1 = (lon + u1*.5*f_lon, lat + v1*.5*f_lat)
    u2, v2 = (fieldx[time + .5 * dt, lon1, lat1], fieldy[time + .5 * dt, lon1, lat1])
    lon2, lat2 = (lon + u2*.5*f_lon, lat + v2*.5*f_lat)
    u3, v3 = (fieldx[time + .5 * dt, lon2, lat2], fieldy[time + .5 * dt, lon2, lat2])
    lon3, lat3 = (lon + u3*f_lon, lat + v3*f_lat)
    u4, v4 = (fieldx[time + dt, lon3, lat3], fieldy[time + dt, lon3, lat3])
    Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
    Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
    return [Vx, Vy]


def LagrangianDiffusion(particle, grid, time, dt):
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r = 1/3.
    Rx = np.random.uniform(-1., 1.)
    Ry = np.random.uniform(-1., 1.)
    dK = RK4(grid.dK_dx, grid.dK_dy, particle.lon, particle.lat, time, dt)
    half_dx = 0.5 * dK[0] * dt * to_lon
    half_dy = 0.5 * dK[1] * dt * to_lat
    Rx_component = Rx * np.sqrt(2 * RK4(grid.K, grid.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)[0] * dt / r) * to_lon
    Ry_component = Ry * np.sqrt(2 * RK4(grid.K, grid.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)[1] * dt / r) * to_lat
    CorrectionX = dK[0] * dt * to_lon
    CorrectionY = dK[1] * dt * to_lat
    particle.lon += Rx_component + CorrectionX
    particle.lat += Ry_component + CorrectionY


def LagrangianDiffusionNoCorrection(particle, grid, time, dt):
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r = 1/3.
    Rx = np.random.uniform(-1., 1.)
    Ry = np.random.uniform(-1., 1.)
    dK = RK4(grid.dK_dx, grid.dK_dy, particle.lon, particle.lat, time, dt)
    half_dx = 0.5 * dK[0] * dt * to_lon
    half_dy = 0.5 * dK[1] * dt * to_lat
    Rx_component = Rx * np.sqrt(2 * RK4(grid.K, grid.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)[0] * dt / r) * to_lon
    Ry_component = Ry * np.sqrt(2 * RK4(grid.K, grid.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)[1] * dt / r) * to_lat
    particle.lon += Rx_component
    particle.lat += Ry_component


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=100,
                   help='Number of particles to advect')
    p.add_argument('-t', '--timesteps', type=int, default=100,
                   help='Timesteps of one second to run simulation over')
    p.add_argument('-o', '--output', type=str, default='diffusion_test',
                   help='Output filename')
    args = p.parse_args()

    # Generate grid files according to given dimensions
    gridx = 100
    gridy = 100
    print('Generating grid')
    forcing_fields = CreateDummyUV(gridx, gridy)
    grid = Grid(forcing_fields['U'], forcing_fields['V'], forcing_fields['U'].depth,
                forcing_fields['U'].time, fields=forcing_fields)
    grid.add_field(CreateDiffusionField(grid))
    divK = grid.K.gradient()
    grid.add_field(divK[0])
    grid.add_field(divK[1])
    grid.add_field(grid.dK_dx.gradient(name="d2K")[0])
    grid.add_field(grid.dK_dy.gradient(name="d2K")[1])
    # Evenly spread starting distribution
    Start_Field = Field('Start', CreateStartField(grid.U.lon, grid.U.lat),
                        grid.U.lon, grid.U.lat, depth=grid.U.depth, time=grid.U.time, transpose=True)

    print('Calculating timestep')
    # Calculate appropriate timestep
    print('Timestep should be < %s' % np.floor(np.min([np.min(1/np.abs(grid.d2K_dx.data)), np.min(1/np.abs(grid.d2K_dy.data))])))
    timestep = 500
    print('Timestep = %s' % timestep)
    steps = args.timesteps

    ParticleClass = JITParticle if args.mode == 'jit' else Particle

    pset = grid.ParticleSet(size=args.particles, pclass=ParticleClass, start_field=Start_Field)
    pset2 = grid.ParticleSet(size=args.particles, pclass=ParticleClass, start_field=Start_Field)

    diffuse = pset.Kernel(LagrangianDiffusion)
    random_walk = pset.Kernel(LagrangianDiffusionNoCorrection)

    dtime = np.arange(0, timestep*steps+1, timestep*steps/20)
    grid.add_field(Field('DensityDiffusion', np.full([grid.U.lon.size, grid.U.lat.size, dtime.size], -1, dtype=np.float32),
                         grid.U.lon, grid.U.lat, depth=grid.U.depth, time=dtime, transpose=True))
    grid.add_field(Field('DensityRandomWalk', np.full([grid.U.lon.size, grid.U.lat.size, dtime.size], -1, dtype=np.float32),
                         grid.U.lon, grid.U.lat, depth=grid.U.depth, time=dtime, transpose=True))

    pset.execute(diffuse, endtime=grid.U.time[0]+timestep*steps, dt=timestep,
                 output_file=pset.ParticleFile(name=args.output+'_Diffusers'),
                 output_interval=timestep, density_field=grid.DensityDiffusion)
    pset2.execute(random_walk, endtime=grid.U.time[0]+timestep*steps, dt=timestep,
                  output_file=pset.ParticleFile(name=args.output+'_RandomWalkers'),
                  output_interval=timestep, density_field=grid.DensityRandomWalk)

    # Density fields of random_walkers should show accumulation in the low-diffusivity centre of the space
    # Diffusers should spread out evenly regardless of this same low-diffusivity

    grid.write(args.output)
