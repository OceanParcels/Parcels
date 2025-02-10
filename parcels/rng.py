import _ctypes
import os
import sys
import uuid
from ctypes import c_float, c_int

import numpy.ctypeslib as npct

from parcels.compilation.codecompiler import GNUCompiler
from parcels.tools import get_cache_dir, get_package_dir
from parcels.tools.loggers import logger

__all__ = ["expovariate", "normalvariate", "randint", "random", "seed", "uniform", "vonmisesvariate"]


class RandomC:
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
    fnct_expovariate = """
extern float pcls_expovariate(float lamb){
  return parcels_expovariate(lamb);
}
"""
    fnct_vonmisesvariate = """
extern float pcls_vonmisesvariate(float mu, float kappa){
  return parcels_vonmisesvariate(mu, kappa);
}
"""
    _lib = None
    ccode = None
    src_file = None
    lib_file = None
    log_file = None

    def __init__(self):
        self._lib = None
        self.ccode = ""
        self.ccode += self.stmt_import
        self.ccode += self.fnct_seed
        self.ccode += self.fnct_random
        self.ccode += self.fnct_uniform
        self.ccode += self.fnct_randint
        self.ccode += self.fnct_normalvariate
        self.ccode += self.fnct_expovariate
        self.ccode += self.fnct_vonmisesvariate
        self._loaded = False
        self.compile()
        self.load_lib()

    def __del__(self):
        self.unload_lib()
        self.remove_lib()

    def unload_lib(self):
        # Unload the currently loaded dynamic linked library to be secure
        if self._lib is not None and self._loaded and _ctypes is not None:
            _ctypes.FreeLibrary(self._lib._handle) if sys.platform == "win32" else _ctypes.dlclose(self._lib._handle)
            del self._lib
            self._lib = None
            self._loaded = False

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, ".")
        self._loaded = True

    def remove_lib(self):
        # If file already exists, pull new names. This is necessary on a Windows machine, because
        # Python's ctype does not deal in any sort of manner well with dynamic linked libraries on this OS.
        if self._lib is not None and self._loaded and _ctypes is not None and os.path.isfile(self.lib_file):
            [os.remove(s) for s in [self.src_file, self.lib_file, self.log_file]]

    def compile(self, compiler=None):
        if self.src_file is None or self.lib_file is None or self.log_file is None:
            basename = f"parcels_random_{uuid.uuid4()}"
            lib_filename = "lib" + basename
            basepath = os.path.join(get_cache_dir(), f"{basename}")
            libpath = os.path.join(get_cache_dir(), f"{lib_filename}")
            self.src_file = f"{basepath}.c"
            self.lib_file = f"{libpath}.so"
            self.log_file = f"{basepath}.log"
        ccompiler = compiler
        if ccompiler is None:
            cppargs = []
            incdirs = [os.path.join(get_package_dir(), "include")]
            ccompiler = GNUCompiler(cppargs=cppargs, incdirs=incdirs)
        if self._lib is None:
            with open(self.src_file, "w+") as f:
                f.write(self.ccode)
            ccompiler.compile(self.src_file, self.lib_file, self.log_file)
            logger.info(f"Compiled ParcelsRandom ==> {self.src_file}")

    @property
    def lib(self):
        if self.src_file is None or self.lib_file is None or self.log_file is None:
            self.compile()
        if self._lib is None or not self._loaded:
            self.load_lib()
            # self._lib = npct.load_library(self.lib_file, '.')
        return self._lib


_parcels_random_ccodeconverter = None


def _assign_parcels_random_ccodeconverter():
    global _parcels_random_ccodeconverter
    if _parcels_random_ccodeconverter is None:
        _parcels_random_ccodeconverter = RandomC()


def seed(seed):
    """Sets the seed for parcels internal RNG."""
    _assign_parcels_random_ccodeconverter()
    _parcels_random_ccodeconverter.lib.pcls_seed(c_int(seed))


def random():
    """Returns a random float between 0.0 and 1.0."""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_random
    rnd.argtype = []
    rnd.restype = c_float
    return rnd()


def uniform(low, high):
    """Returns a random float between `low` and `high`."""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_uniform
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(low), c_float(high))


def randint(low, high):
    """Returns a random int between `low` and `high`."""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_randint
    rnd.argtype = [c_int, c_int]
    rnd.restype = c_int
    return rnd(c_int(low), c_int(high))


def normalvariate(loc, scale):
    """Returns a random float on normal distribution with mean `loc` and width `scale`."""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_normalvariate
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(loc), c_float(scale))


def expovariate(lamb):
    """Returns a random float of an exponential distribution with parameter lamb."""
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_expovariate
    rnd.argtype = c_float
    rnd.restype = c_float
    return rnd(c_float(lamb))


def vonmisesvariate(mu, kappa):
    """Returns a random float of a Von Mises distribution
    with mean angle mu and concentration parameter kappa.
    """
    _assign_parcels_random_ccodeconverter()
    rnd = _parcels_random_ccodeconverter.lib.pcls_vonmisesvariate
    rnd.argtype = [c_float, c_float]
    rnd.restype = c_float
    return rnd(c_float(mu), c_float(kappa))
