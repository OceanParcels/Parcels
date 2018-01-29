from parcels import rng as random
import math


__all__ = ['BrownianMotion2D']


def BrownianMotion2D(particle, fieldset, time, dt):
    # Kernel for simple Brownian particle diffusion in zonal and meridional direction.
    # Assumes that fieldset has fields Kh_zonal and Kh_meridional

    r = 1/3.
    kh_meridional = fieldset.Kh_meridional[time, particle.lon, particle.lat, particle.depth]
    particle.lat += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dt)*kh_meridional/r)
    kh_zonal = fieldset.Kh_zonal[time, particle.lon, particle.lat, particle.depth]
    particle.lon += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dt)*kh_zonal/r)
