from argparse import ArgumentParser
from parcels.field import Field
from parcels.grid import Grid
import numpy as np
import math
from parcels.particle import Particle, JITParticle


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


def GradientRK4(particle, grid, time, dt):
    f_lat = dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(particle.lat*math.pi/180)
    V = RK4(grid.dK_dx, grid.dK_dy, particle.lon, particle.lat, time, dt)
    particle.lon += V[0] * f_lon
    particle.lat += V[1] * f_lat


def CreateForcingFields(xdim=200, ydim=350, time=25):
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(1., time, time/2, dtype=np.float64)

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(1, 2, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional) forcing as zero
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    ufield = Field('U', U, lon, lat, depth=depth,
                   time=time, transpose=True)
    vfield = Field('V', V, lon, lat, depth=depth,
                   time=time, transpose=True)

    return {'U': ufield, 'V': vfield}


def CreateDiffusionField(grid, mu=None):
    """Generate a simple gradient field containing a single maxima and gaussian gradient
    """
    depth = np.zeros(1, dtype=np.float32)
    time = grid.time

    lon, lat = grid.U.lon, grid.U.lat

    K = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    if mu is None:
        mu = [np.mean(grid.U.lon), np.mean(grid.U.lat)]

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

    sig = [[0.3, 0], [0, 0.3]]
    for i, x in enumerate(lon):
        for j, y in enumerate(lat):
            K[i, j, :] = MVNorm(x, y, mu, sig)

    # Boost to provide enough force for our gradient climbers
    K *= 10000
    K_Field = Field('K', K, lon, lat, depth, time, transpose=True)

    return K_Field


def CreateInitialPositionField(grid):
    # Simple evenly distribution starting density field
    depth = grid.U.depth
    time = grid.U.time
    lon = grid.U.lon
    lat = grid.U.lat
    Num = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    for i, x in enumerate(lon):
        for j, y in enumerate(lat):
            Num[i, j, :] = 1

    return Field('Start', Num, lon, lat, depth, time, transpose=True)


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=1000,
                   help='Number of particles to advect')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int, default=None,
                   help='Generate grid file with given dimensions')
    p.add_argument('-o', '--output', default='density_test',
                   help='List of NetCDF files to load')
    args = p.parse_args()
    filename = args.output
    time = 100

    # Generate grid files according to given dimensions
    gridx = 5
    gridy = 5
    forcing_fields = CreateForcingFields(gridx, gridy, time)
    grid = Grid(forcing_fields['U'], forcing_fields['V'], forcing_fields['U'].depth,
                forcing_fields['U'].time, fields=forcing_fields)
    K = CreateDiffusionField(grid, mu=[np.mean(grid.U.lon), np.mean(grid.U.lat)])
    grid.add_field(K)
    Kprime = K.gradient()
    for field in Kprime:
        grid.add_field(field)
    grid.add_field(CreateInitialPositionField(grid))

    # grid.add_field(CreateInitialPositionField(grid))
    ParticleClass = JITParticle if args.mode == 'jit' else Particle

    climbers = grid.ParticleSet(size=args.particles, pclass=ParticleClass,
                                start_field=grid.Start)
    # Calculate the initial particle density and save to grid as a field
    # (should be fairly homogenous, but with some random structure due to
    # stochasitic nature of particle start positions!)
    StartDensity = climbers.density()

    timestep = 10000
    substeps = timestep

    climb = climbers.Kernel(GradientRK4)

    climbers.execute(climb, endtime=climbers.grid.time[0]+time*timestep, dt=timestep,
                     output_file=climbers.ParticleFile(name=filename+"_particle"),
                     output_interval=substeps)

    # Calculate final particle density and save to grid
    # (density should be maximal in the centre! Neat way to test this quantitatively??)
    EndDensity = climbers.density()

    Densities = np.full((grid.U.lon.size, grid.U.lat.size, grid.U.time.size), -1, dtype=np.float32)
    Densities[:, :, 0] = StartDensity
    Densities[:, :, -1] = EndDensity
    grid.add_field(Field('Density', Densities, grid.U.lon, grid.U.lat,
                         depth=grid.U.depth, time=grid.U.time, transpose=True))

    grid.write(args.output)
