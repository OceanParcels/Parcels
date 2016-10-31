from argparse import ArgumentParser
import numpy as np
import math
import pytest
from parcels import Field, Grid, JITParticle, ScipyParticle, Variable


def GradientClimber(particle, grid, time, dt):
    f_lat = dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(particle.lat*math.pi/180)
    u1 = grid.dK_dx[time, particle.lon, particle.lat]
    v1 = grid.dK_dy[time, particle.lon, particle.lat]
    lon1, lat1 = (particle.lon + u1*.5*f_lon, particle.lat + v1*.5*f_lat)
    u2, v2 = (grid.dK_dx[time + .5 * dt, lon1, lat1], grid.dK_dy[time + .5 * dt, lon1, lat1])
    lon2, lat2 = (particle.lon + u2*.5*f_lon, particle.lat + v2*.5*f_lat)
    u3, v3 = (grid.dK_dx[time + .5 * dt, lon2, lat2], grid.dK_dy[time + .5 * dt, lon2, lat2])
    lon3, lat3 = (particle.lon + u3*f_lon, particle.lat + v3*f_lat)
    u4, v4 = (grid.dK_dx[time + dt, lon3, lat3], grid.dK_dy[time + dt, lon3, lat3])
    Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
    Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
    particle.lon += Vx * f_lon
    particle.lat += Vy * f_lat


def CreateForcingFields(xdim=200, ydim=350, time=25):
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0, time+1, time/2, dtype=np.float64)

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


def CreateGradientField(grid, mu=None):
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
    boost = 50000
    K *= boost
    K_Field = Field('K', K, lon, lat, depth, time, transpose=True)

    return K_Field


def CreateInitialPositionField(grid):
    # Simple evenly distribution starting density field
    depth = grid.U.depth
    time = grid.U.time
    lon = grid.U.lon
    lat = grid.U.lat
    Num = np.ones((lon.size, lat.size, time.size), dtype=np.float32)
    return Field('Start', Num, lon, lat, depth, time, transpose=True)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_density_calculation(mode, pnum):
    time = 100000

    # Generate grid files according to given dimensions
    gridx = 5
    gridy = 5
    forcing_fields = CreateForcingFields(gridx, gridy, time)
    grid = Grid(forcing_fields['U'], forcing_fields['V'], forcing_fields['U'].depth,
                forcing_fields['U'].time, fields=forcing_fields)
    K = CreateGradientField(grid, mu=[np.mean(grid.U.lon), np.mean(grid.U.lat)])
    grid.add_field(K)
    K_gradients = grid.K.gradient()
    for field in K_gradients:
        grid.add_field(field)

    grid.add_field(CreateInitialPositionField(grid))

    # grid.add_field(CreateInitialPositionField(grid))
    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class DensityP(ParticleClass):
        weight = Variable('weight', dtype=np.float32)

    climbers = grid.ParticleSet(size=pnum, pclass=DensityP,
                                start_field=grid.Start)

    for p in climbers.particles:
        p.weight = 2

    dtime = np.array([grid.U.time[0], grid.U.time[-1]], dtype=np.float32)
    Densities = np.full((grid.U.lon.size, grid.U.lat.size, dtime.size), -1, dtype=np.float32)
    grid.add_field(Field('Density', Densities, grid.U.lon, grid.U.lat,
                   depth=grid.U.depth, time=dtime, transpose=True))

    grid.Density.data[0, :, :] = climbers.density(particle_val="weight")

    print("-- Initial Particle Density --")
    print(grid.Density.data[0, :, :])

    timestep = 1000
    substeps = 10000

    climb = climbers.Kernel(GradientClimber)

    climbers.execute(climb, endtime=climbers.grid.time[-1], dt=timestep,
                     output_file=climbers.ParticleFile(name="density_test_particle"),
                     interval=substeps)

    grid.Density.data[-1, :, :] = climbers.density(particle_val="weight")

    # Final densities should be zero everywhere except central vertex
    print("-- Final Particle Density --")
    print(grid.Density.data[-1, :, :])

    mask = np.ones(np.shape(grid.Density.data), dtype=bool)
    mask[0, :, :] = 0
    mask[-1, 2, 2] = 0
    assert np.all(grid.Density.data[mask] == 0), "Particle density is non-zero away from central vertex"


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=100,
                   help='Number of particles to advect')
    args = p.parse_args()

    test_density_calculation(args.mode, args.particles)
