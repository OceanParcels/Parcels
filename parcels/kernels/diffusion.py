from parcels import rng as random
import math


__all__ = ['BrownianMotion2D', 'SpatiallyVaryingBrownianMotion2D']


def BrownianMotion2D(particle, fieldset, time, dt):
    # Kernel for simple Brownian particle diffusion in zonal and meridional direction.
    # Assumes that fieldset has fields Kh_zonal and Kh_meridional

    r = 1/3.
    kh_meridional = fieldset.Kh_meridional[time, particle.lon, particle.lat, particle.depth]
    particle.lat += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dt)*kh_meridional/r)
    kh_zonal = fieldset.Kh_zonal[time, particle.lon, particle.lat, particle.depth]
    particle.lon += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dt)*kh_zonal/r)


def SpatiallyVaryingBrownianMotion2D(particle, fieldset, time, dt):
    # Diffusion equations for particles in non-uniform diffusivity fields
    # from Ross & Sharples 2004 and Spagnol et al. 2002

    # regular Brownian motion step
    r = 1/3.
    kh_meridional = fieldset.Kh_meridional[time, particle.lon, particle.lat, particle.depth]
    Ry = random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dt)*kh_meridional/r)
    kh_zonal = fieldset.Kh_zonal[time, particle.lon, particle.lat, particle.depth]
    Rx = random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dt)*kh_zonal/r)

    # Deterministic 'boost' out of areas of low diffusivity
    dKdx = fieldset.dKh_zonal_dx[time, particle.lon, particle.lat, particle.depth]
    dKdy = fieldset.dKh_meridional_dy[time, particle.lon, particle.lat, particle.depth]
    CorrectionX = dKdx * math.fabs(dt)
    CorrectionY = dKdy * math.fabs(dt)

    # diffuse particle as sum of Brownian motion and deterministic 'boost'
    particle.lon += Rx + CorrectionX
    particle.lat += Ry + CorrectionY
