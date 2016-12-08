from parcels.rng import random
import math


__all__ = ['LagrangianDiffusion']


def LagrangianDiffusion(particle, grid, time, dt):
    # Diffusion equations for particles in non-uniform diffusivity fields
    # from Ross &  Sharples 2004 and Spagnol et al. 2002
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r_var = 1/3.
    Rx = random.uniform(-1., 1.)
    Ry = random.uniform(-1., 1.)
    dKdx, dKdy = (grid.dK_dx[time, particle.lon, particle.lat], grid.dK_dy[time, particle.lon, particle.lat])
    Kfield = grid.K[time, particle.lon, particle.lat]
    Rx_component = Rx * math.sqrt(2 * Kfield * dt / r_var) * to_lon
    Ry_component = Ry * math.sqrt(2 * Kfield * dt / r_var) * to_lat
    # Deterministic 'boost' out of areas of low diffusivity
    CorrectionX = dKdx * dt * to_lon
    CorrectionY = dKdy * dt * to_lat
    # diffuse particle
    particle.lon += Rx_component + CorrectionX
    particle.lat += Ry_component + CorrectionY
