"""Collection of pre-built diffusion kernels"""
from parcels import rng as random
import math


__all__ = ['BrownianMotion2D', 'SpatiallyVaryingBrownianMotion2D']


def BrownianMotion2D(particle, fieldset, time):
    """Kernel for simple Brownian particle diffusion in zonal and meridional direction.
    Assumes that fieldset has fields Kh_zonal and Kh_meridional"""

    r = 1/3.
    kh_meridional = fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon]
    particle.lat += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(particle.dt)*kh_meridional/r)
    kh_zonal = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon]
    particle.lon += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(particle.dt)*kh_zonal/r)


def SpatiallyVaryingBrownianMotion2D(particle, fieldset, time):
    """Diffusion equations for particles in non-uniform diffusivity fields
    from Ross & Sharples (2004, doi:10.4319/lom.2004.2.289)
    and Spagnol et al. (2002, doi:10.3354/meps235299)"""

    # regular Brownian motion step
    r = 1/3.
    kh_meridional = fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon]
    Ry = random.uniform(-1., 1.) * math.sqrt(2*math.fabs(particle.dt)*kh_meridional/r)
    kh_zonal = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon]
    Rx = random.uniform(-1., 1.) * math.sqrt(2*math.fabs(particle.dt)*kh_zonal/r)

    # Deterministic 'boost' out of areas of low diffusivity
    dKdx = fieldset.dKh_zonal_dx[time, particle.depth, particle.lat, particle.lon]
    dKdy = fieldset.dKh_meridional_dy[time, particle.depth, particle.lat, particle.lon]
    CorrectionX = dKdx * math.fabs(particle.dt)
    CorrectionY = dKdy * math.fabs(particle.dt)

    # diffuse particle as sum of Brownian motion and deterministic 'boost'
    particle.lon += Rx + CorrectionX
    particle.lat += Ry + CorrectionY
