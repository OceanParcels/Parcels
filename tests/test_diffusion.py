from parcels import Field, FieldSet, JITParticle, ScipyParticle, Variable, SaptiallyVaryingDiffusion2D, random, ErrorCode
from operator import attrgetter
import numpy as np
import math
from argparse import ArgumentParser
import pytest


def CreateDiffusionField(fieldset, mu=None):
    """Generates a non-uniform diffusivity field using a multivariate normal distribution
    """
    depth = fieldset.U.depth
    time = fieldset.U.time
    lon = fieldset.U.lon
    lat = fieldset.U.lat

    K = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    # Simple multivariate normal pdf function
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
    # Define multivariate mean (mu) and covariance matrix (sig) parameters
    if mu is None:
        mu = [np.mean(lon), np.mean(lat)]
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


def LagrangianDiffusionNoCorrection(particle, fieldset, time, dt):
    # Version of diffusion equation with no determenistic term i.e. brownian motion
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r_var = 1/3.
    Rx = random.uniform(-1., 1.)
    Ry = random.uniform(-1., 1.)
    Kfield = fieldset.K[time, particle.lon, particle.lat]
    Rx_component = Rx * math.sqrt(2 * Kfield * dt / r_var) * to_lon
    Ry_component = Ry * math.sqrt(2 * Kfield * dt / r_var) * to_lat
    particle.lon += Rx_component
    particle.lat += Ry_component


def UpdatePosition(particle, fieldset, time, dt):
    particle.prev_lon = particle.new_lon
    particle.new_lon = particle.lon
    particle.prev_lat = particle.new_lat
    particle.new_lat = particle.lat


# Recovery Kernal for particles that diffuse outside boundary
def Send2PreviousPoint(particle):
    # print("Recovery triggered at %s | %s!" % (particle.lon, particle.lat))
    # print("Moving particle back to %s | %s" % (particle.prev_lon, particle.prev_lat))
    particle.lon = particle.prev_lon
    particle.lat = particle.prev_lat


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def diffusion_test(mode, type='true_diffusion', particles=1000, timesteps=1000, output_file='diffusion_test'):
    # Generating fieldset with zero horizontal velocities
    forcing_fields = CreateDummyUV(100, 100)
    fieldset = FieldSet(forcing_fields['U'], forcing_fields['V'], forcing_fields['U'].depth,
                        forcing_fields['U'].time, fields=forcing_fields)
    # Create a non-uniform field of diffusivity
    fieldset.add_field(CreateDiffusionField(fieldset))
    # Calculate first differential of diffusivity field (needed for diffusion)
    divK = fieldset.K.gradient()
    fieldset.add_field(divK[0])
    fieldset.add_field(divK[1])
    # Calculate second differential (needed to estimate the appropriate minimum timestep to approximate Eulerian diffusion)
    fieldset.add_field(fieldset.dK_dx.gradient(name="d2K")[0])
    fieldset.add_field(fieldset.dK_dy.gradient(name="d2K")[1])

    # Evenly spread starting distribution
    Start_Field = Field('Start', CreateStartField(fieldset.U.lon, fieldset.U.lat),
                        fieldset.U.lon, fieldset.U.lat, depth=fieldset.U.depth, time=fieldset.U.time, transpose=True)

    timestep = 500

    steps = timesteps

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    # Simply particle superclass that remembers previous positions for kernel error recovery
    class Diffuser(ParticleClass):
        prev_lon = Variable("prev_lon", to_write=False, initial=attrgetter('lon'))
        prev_lat = Variable("prev_lat", to_write=False, initial=attrgetter('lat'))
        new_lon = Variable("new_lon", to_write=False, initial=attrgetter('lon'))
        new_lat = Variable("new_lat", to_write=False, initial=attrgetter('lat'))

    diffusers = fieldset.ParticleSet(size=particles, pclass=Diffuser, start_field=Start_Field)

    # Particle density at simulation start should be more or less uniform
    DensityField = Field('temp', np.zeros((5, 5), dtype=np.float32),
                         np.linspace(np.min(fieldset.U.lon), np.max(fieldset.U.lon), 5, dtype=np.float32),
                         np.linspace(np.min(fieldset.U.lat), np.max(fieldset.U.lat), 5, dtype=np.float32))
    StartDensity = diffusers.density(DensityField, area_scale=False, relative=True)
    fieldset.add_field(Field(type + 'StartDensity', StartDensity,
                             DensityField.lon,
                             DensityField.lat))

    diffuse = diffusers.Kernel(SaptiallyVaryingDiffusion2D) if type == 'true_diffusion' else diffusers.Kernel(LagrangianDiffusionNoCorrection)

    diffusers.execute(diffusers.Kernel(UpdatePosition) + diffuse, endtime=fieldset.U.time[0]+timestep*steps, dt=timestep,
                      output_file=diffusers.ParticleFile(name=args.output+type),
                      interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: Send2PreviousPoint})

    EndDensity = diffusers.density(DensityField, area_scale=False, relative=True)
    fieldset.add_field(Field(type+'FinalDensity', EndDensity,
                             DensityField.lon,
                             DensityField.lat))
    fieldset.write(output_file)

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
    assert np.abs(np.var(densities2[0]) - np.var(densities2[1])) > tol, \
        'Variance in brownian motion particle density across cells is not significantly different from start to end of simulation!'
