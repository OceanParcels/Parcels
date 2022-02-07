from ctypes import c_float
from ctypes import c_int

__all__ = ['seed', 'random', 'uniform', 'randint', 'normalvariate', 'expovariate', 'vonmisesvariate']


_parcels_random_ccodeconverter = None


def _assign_parcels_random_ccodeconverter():
    global _parcels_random_ccodeconverter
    if _parcels_random_ccodeconverter is None:
        _parcels_random_ccodeconverter = RandomC()


def seed(seed):
    """Sets the seed for parcels internal RNG"""
    _assign_parcels_random_ccodeconverter()
    _parcels_random_ccodeconverter.lib.pcls_seed(c_int(seed))


def random():
    """Returns a random float between 0. and 1."""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_random
    rnd.argtype = []
    rnd.restype = c_float
    return rnd()


def uniform(low, high):
    """Returns a random float between `low` and `high`"""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_uniform
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(low), c_float(high))


def randint(low, high):
    """Returns a random int between `low` and `high`"""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_randint
    rnd.argtype = [c_int, c_int]
    rnd.restype = c_int
    return rnd(c_int(low), c_int(high))


def normalvariate(loc, scale):
    """Returns a random float on normal distribution with mean `loc` and width `scale`"""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_normalvariate
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(loc), c_float(scale))


def expovariate(lamb):
    """Returns a randome float of an exponential distribution with parameter lamb"""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_expovariate
    rnd.argtype = c_float
    rnd.restype = c_float
    return rnd(c_float(lamb))


def vonmisesvariate(mu, kappa):
    """Returns a randome float of a Von Mises distribution
    with mean angle mu and concentration parameter kappa"""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_vonmisesvariate
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(mu), c_float(kappa))
