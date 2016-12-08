from parcels import Field, Grid, JITParticle, ScipyParticle, Variable, LagrangianDiffusion, random, ErrorCode
import numpy as np
import math
from argparse import ArgumentParser
import pytest


def CreateDiffusionField(grid, mu=None):
    """Generates a non-uniform diffusivity field
    """
    depth = grid.U.depth
    time = grid.U.time
    lon = grid.U.lon
    lat = grid.U.lat

    K = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    if mu is None:
        mu = [np.mean(lon), np.mean(lat)]

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

    sig = [[0.01, 0], [0, 0.01]]
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
    lon = np.linspace(0, 0.1, xdim, dtype=np.float32)
    lat = np.linspace(0, 0.1, ydim, dtype=np.float32)

    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    ufield = Field('U', U, lon, lat, depth=depth,
                   time=time, transpose=True)
    vfield = Field('V', V, lon, lat, depth=depth,
                   time=time, transpose=True)

    return {'U': ufield, 'V': vfield}


def CreateStartField(lon, lat):
    time = np.arange(0., 100000, 100000/2., dtype=np.float64)
    # An evenly distributed starting density
    data = np.ones((lon.size, lat.size, time.size), dtype=np.float32)
    return data


def LagrangianDiffusionNoCorrection(particle, grid, time, dt):
    # Version of diffusion equation with no determenistic term i.e. brownian motion
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r_var = 1/3.
    Rx = random.uniform(-1., 1.)
    Ry = random.uniform(-1., 1.)
    Kfield = grid.K[time, particle.lon, particle.lat]
    Rx_component = Rx * math.sqrt(2 * Kfield * dt / r_var) * to_lon
    Ry_component = Ry * math.sqrt(2 * Kfield * dt / r_var) * to_lat
    particle.lon += Rx_component
    particle.lat += Ry_component


def UpdatePosition(particle, grid, time, dt):
    particle.prev_lon = particle.new_lon
    particle.new_lon = particle.lon
    particle.prev_lat = particle.new_lat
    particle.new_lat = particle.lat


# Recovery Kernal for particles that diffuse outside boundary
def Send2PreviousPoint(particle):
    # print("Recovery triggered at %s | %s!" % (particle.lon, particle.lat))
    # print("Moving particle back to %s | %s" % (particle.prev_lat, particle.prev_lat))
    particle.lon = particle.prev_lon
    particle.lat = particle.prev_lat


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def diffusion_test(mode, type='true_diffusion', particles=1000, timesteps=1000, output_file='diffusion_test'):
    # Generate grid files according to given dimensions
    gridx = 100
    gridy = 100
    # Generating grid with zero horizontal velocities
    forcing_fields = CreateDummyUV(gridx, gridy)
    grid = Grid(forcing_fields['U'], forcing_fields['V'], forcing_fields['U'].depth,
                forcing_fields['U'].time, fields=forcing_fields)
    # Create a non-uniform field of diffusivity
    grid.add_field(CreateDiffusionField(grid))
    # Calculate first differential of diffusivity field (needed for diffusion)
    divK = grid.K.gradient()
    grid.add_field(divK[0])
    grid.add_field(divK[1])
    # Calculate second differential (needed to estimate the appropriate minimum timestep to approximate Eulerian diffusion)
    grid.add_field(grid.dK_dx.gradient(name="d2K")[0])
    grid.add_field(grid.dK_dy.gradient(name="d2K")[1])

    # Evenly spread starting distribution
    Start_Field = Field('Start', CreateStartField(grid.U.lon, grid.U.lat),
                        grid.U.lon, grid.U.lat, depth=grid.U.depth, time=grid.U.time, transpose=True)

    # Calculate appropriate timestep
    min_timestep = np.floor(np.min([np.min(1/np.abs(grid.d2K_dx.data)), np.min(1/np.abs(grid.d2K_dy.data))]))
    print('Timestep should be < %s' % min_timestep)
    timestep = 500

    steps = timesteps

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    # Simply particle superclass that remembers previous positions for kernel error recovery
    class Diffuser(ParticleClass):
        prev_lon = Variable("prev_lon", to_write=False)
        prev_lat = Variable("prev_lat", to_write=False)
        new_lon = Variable("new_lon", to_write=False)
        new_lat = Variable("new_lat", to_write=False)

        def __init__(self, *args, **kwargs):
            super(Diffuser, self).__init__(*args, **kwargs)
            self.prev_lon = 0.05
            self.prev_lat = 0.05

    diffusers = grid.ParticleSet(size=particles, pclass=Diffuser, start_field=Start_Field)

    # Particle density at simulation start should be more or less uniform
    DensityField = Field('temp', np.zeros((5, 5), dtype=np.float32),
                         np.linspace(np.min(grid.U.lon), np.max(grid.U.lon), 5, dtype=np.float32),
                         np.linspace(np.min(grid.U.lat), np.max(grid.U.lat), 5, dtype=np.float32))
    StartDensity = diffusers.density(DensityField, area_scale=False, relative=True)
    grid.add_field(Field(type + 'StartDensity', StartDensity,
                         DensityField.lon,
                         DensityField.lat))

    diffuse = diffusers.Kernel(LagrangianDiffusion) if type == 'true_diffusion' else diffusers.Kernel(LagrangianDiffusionNoCorrection)

    diffusers.execute(diffusers.Kernel(UpdatePosition) + diffuse, endtime=grid.U.time[0]+timestep*steps, dt=timestep,
                      output_file=diffusers.ParticleFile(name=args.output+type),
                      interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: Send2PreviousPoint})

    EndDensity = diffusers.density(DensityField, area_scale=False, relative=True)
    grid.add_field(Field(type+'FinalDensity', EndDensity,
                         DensityField.lon,
                         DensityField.lat))
    grid.write(output_file)

    print(type + ' start variation = %s' % np.var(StartDensity))
    print(type + ' end variation = %s' % np.var(EndDensity))

    return [StartDensity, EndDensity]


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of diffusion in a non-uniform diffusivity field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=1000,
                   help='Number of particles to advect')
    p.add_argument('-t', '--timesteps', type=int, default=1000,
                   help='Timesteps of one second to run simulation over')
    p.add_argument('-o', '--output', type=str, default='diffusion_test',
                   help='Output filename')
    args = p.parse_args()

    # Density fields of random_walkers should show accumulation in the low-diffusivity centre of the space
    # Diffusers should spread out evenly regardless of this same low-diffusivity
    densities1 = diffusion_test(args.mode, 'true_diffusion', args.particles, args.timesteps, args.output)

    tol = 0.0001
    assert np.abs(np.var(densities1[0]) - np.var(densities1[1])) < tol, \
        'Variance in diffuser particle density across cells is significantly different from start to end of simulation!'

    densities2 = diffusion_test(args.mode, 'brownian_motion', args.particles, args.timesteps, args.output)
    print(np.abs(np.var(densities2[0]) - np.var(densities2[1])))
    assert np.abs(np.var(densities2[0]) - np.var(densities2[1])) > tol, \
        'Variance in brownian motion particle density across cells is not significantly different from start to end of simulation!'
