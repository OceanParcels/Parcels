from collections import OrderedDict
import numpy as np


__all__ = ['Particle', 'JITParticle']


class ParticleType(object):
    """Class encapsulating the type information for custom particles

    :param user_vars: Optional list of (name, dtype) tuples for custom variables
    """

    def __init__(self, pclass):
        if not isinstance(pclass, type):
            raise TypeError("Class object required to derive ParticleType")
        if not issubclass(pclass, Particle):
            raise TypeError("Class object does not inherit from parcels.Particle")

        self.name = pclass.__name__
        self.uses_jit = issubclass(pclass, JITParticle)
        self.var_types = None
        if self.uses_jit:
            self.var_types = pclass.base_vars.copy()
            self.var_types.update(pclass.user_vars)

        self.user_vars = pclass.user_vars

    def __repr__(self):
        return "PType<%s>::%s" % (self.name, str(self.var_types))

    @property
    def _cache_key(self):
        return"-".join(["%s:%s" % v for v in self.var_types.items()])

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        type_list = list(self.var_types.items())
        if self.size % 8 > 0:
            # Add padding to be 64-bit aligned
            type_list += [('pad', np.float32)]
        return np.dtype(type_list)

    @property
    def size(self):
        """Size of the underlying particle struct in bytes"""
        return sum([8 if vt == np.float64 else 4
                    for vt in self.var_types.values()])


class Particle(object):
    """Class encapsualting the basic attributes of a particle

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param grid: :Class Grid: object to track this particle on
    :param user_vars: Dictionary of any user variables that might be defined in subclasses
    """
    user_vars = OrderedDict()

    def __init__(self, lon, lat, grid, dt=3600., time=0., cptr=None):
        self.lon = lon
        self.lat = lat
        self.time = time
        self.dt = dt

        self.xi = np.where(self.lon >= grid.U.lon)[0][-1]
        self.yi = np.where(self.lat >= grid.U.lat)[0][-1]
        self.active = 1

        for var in self.user_vars:
            setattr(self, var, 0)

    def __repr__(self):
        return "P(%f, %f, %f)[%d, %d]" % (self.lon, self.lat, self.time,
                                          self.xi, self.yi)

    @classmethod
    def getPType(cls):
        return ParticleType(cls)

    def delete(self):
        self.active = 0


class JITParticle(Particle):
    """Particle class for JIT-based Particle objects

    Users should extend this type for custom particles with fast
    advection computation. Additional variables need to be defined
    via the :user_vars: list of (name, dtype) tuples.

    :param user_vars: Class variable that defines additional particle variables
    """

    base_vars = OrderedDict([('lon', np.float32), ('lat', np.float32),
                             ('time', np.float64), ('dt', np.float32),
                             ('xi', np.int32), ('yi', np.int32),
                             ('active', np.int32)])
    user_vars = OrderedDict()

    def __init__(self, *args, **kwargs):
        self._cptr = kwargs.pop('cptr', None)
        if self._cptr is None:
            # Allocate data for a single particle
            ptype = super(JITParticle, self).getPType()
            self._cptr = np.empty(1, dtype=ptype.dtype)[0]
        super(JITParticle, self).__init__(*args, **kwargs)

    def __getattr__(self, attr):
        if attr == "_cptr":
            return super(JITParticle, self).__getattr__(attr)
        else:
            return self._cptr.__getitem__(attr)

    def __setattr__(self, key, value):
        if key == "_cptr":
            super(JITParticle, self).__setattr__(key, value)
        else:
            self._cptr.__setitem__(key, value)
