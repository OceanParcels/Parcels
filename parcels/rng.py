from parcels.compiler import get_cache_dir, GNUCompiler
from parcels.loggers import logger
from os import path
import numpy.ctypeslib as npct
from ctypes import c_int, c_float


__all__ = ['seed', 'random', 'uniform', 'randint', 'normalvariate']


class Random(object):
    stmt_import = """#include "parcels.h"\n\n"""
    fnct_seed = """
extern void pcls_seed(int seed){
  parcels_seed(seed);
}
"""
    fnct_random = """
extern float pcls_random(){
  return parcels_random();
}
"""
    fnct_uniform = """
extern float pcls_uniform(float low, float high){
  return parcels_uniform(low, high);
}
"""
    fnct_randint = """
extern int pcls_randint(int low, int high){
  return parcels_randint(low, high);
}
"""
    fnct_normalvariate = """
extern float pcls_normalvariate(float loc, float scale){
  return parcels_normalvariate(loc, scale);
}
"""
    ccode = stmt_import + fnct_seed
    ccode += fnct_random + fnct_uniform + fnct_randint + fnct_normalvariate
    src_file = path.join(get_cache_dir(), "random.c")
    lib_file = path.join(get_cache_dir(), "random.so")
    log_file = path.join(get_cache_dir(), "random.log")

    def __init__(self):
        self._lib = None

    @property
    def lib(self, compiler=GNUCompiler()):
        if self._lib is None:
            with open(self.src_file, 'w') as f:
                f.write(self.ccode)
            compiler.compile(self.src_file, self.lib_file, self.log_file)
            logger.info("Compiled %s ==> %s" % ("random", self.lib_file))
            self._lib = npct.load_library(self.lib_file, '.')
        return self._lib


parcels_random = Random()


def seed(seed):
    """Sets the seed for parcels internal RNG"""
    parcels_random.lib.pcls_seed(c_int(seed))


def random():
    """Returns a random float between 0. and 1."""
    rnd = parcels_random.lib.pcls_random
    rnd.argtype = []
    rnd.restype = c_float
    return rnd()


def uniform(low, high):
    """Returns a random float between `low` and `high`"""
    rnd = parcels_random.lib.pcls_uniform
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(low), c_float(high))


def randint(low, high):
    """Returns a random int between `low` and `high`"""
    rnd = parcels_random.lib.pcls_randint
    rnd.argtype = [c_int, c_int]
    rnd.restype = c_int
    return rnd(c_int(low), c_int(high))


def normalvariate(loc, scale):
    """Returns a random float on normal distribution with mean `loc` and width `scale`"""
    rnd = parcels_random.lib.pcls_normalvariate
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(loc), c_float(scale))
