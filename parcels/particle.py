import numpy as np
from parcels.jit_module import Kernel, GNUCompiler

__all__ = ['Particle', 'ParticleSet', 'JITParticle', 'JITParticleSet']

ctype = {np.int32: 'int', np.float32: 'float'}


class Particle(object):
    """Class encapsualting the basic attributes of a particle

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param grid: :Class Grid: object to track this particle on
    """

    def __init__(self, lon, lat, grid):
        self.lon = lon
        self.lat = lat

        self.xi = np.where(self.lon > grid.U.lon)[0][-1]
        self.yi = np.where(self.lat > grid.U.lat)[0][-1]

    def __repr__(self):
        return "P(%f, %f)[%d, %d]" % (self.lon, self.lat, self.xi, self.yi)

    def advect_rk4(self, grid, dt):
        f = dt / 1000. / 1.852 / 60.
        u1, v1 = grid.eval(self.lon, self.lat)
        lon1, lat1 = (self.lon + u1*.5*f, self.lat + v1*.5*f)
        u2, v2 = grid.eval(lon1, lat1)
        lon2, lat2 = (self.lon + u2*.5*f, self.lat + v2*.5*f)
        u3, v3 = grid.eval(lon2, lat2)
        lon3, lat3 = (self.lon + u3*f, self.lat + v3*f)
        u4, v4 = grid.eval(lon3, lat3)
        self.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f
        self.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f


class ParticleSet(object):
    """Container class for storing and executing over sets of particles.

    Please note that this currently only supports fixed size particle sets.

    :param size: Initial size of particle set
    :param grid: Grid object from which to sample velocity"""

    def __init__(self, size, grid, pclass=Particle, lon=None, lat=None):
        self._grid = grid
        self._particles = np.empty(size, dtype=pclass)

        if lon is not None and lat is not None:
            for i in range(size):
                self._particles[i] = pclass(lon=lon[i], lat=lat[i], grid=grid)
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")

    @property
    def size(self):
        return self._particles.size

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return self._particles[key]

    def __setitem__(self, key, value):
        self._particles[key] = value

    def advect(self, timesteps=1, dt=None):
        print "Parcels::ParticleSet: Advecting %d particles for %d timesteps" \
            % (len(self), timesteps)
        for t in range(timesteps):
            for p in self._particles:
                p.advect_rk4(self._grid, dt)


class ParticleType(object):
    """Class encapsulating the type information for custom particles

    :param user: Optional list of (name, dtype) tuples for custom variables
    """

    def __init__(self, user=[]):
        self.base = [('lon', np.float32), ('lat', np.float32),
                     ('xi', np.int32), ('yi', np.int32)]
        self.user = user

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        return np.dtype(self.base + self.user)

    @property
    def code(self, name='Particle'):
        """Type definition for the corresponding C struct"""
        tdef = '\n'.join(['  %s %s;' % (ctype[t], v) for v, t in self.base + self.user])
        return """#define PARCELS_PTYPE
typedef struct
{
%s
} %s;""" % (tdef, name)


class JITParticle(Particle):
    """Particle class for JIT-based Particle objects

    Users should extend this type for custom particles with fast
    advection computation. Additional variables need to be defined
    via the :user_vars: list of (name, dtype) tuples.

    :param user_vars: Class variable that defines additional particle variables
    """

    user_vars = []

    def __init__(self, *args, **kwargs):
        self._cptr = kwargs.pop('cptr', None)
        super(JITParticle, self).__init__(*args, **kwargs)

    def __getattr__(self, attr):
        if hasattr(self, '_cptr'):
            return self._cptr.__getitem__(attr)
        else:
            return super(JITParticle, self).__getattr__(attr)

    def __setattr__(self, key, value):
        if hasattr(self, '_cptr'):
            self._cptr.__setitem__(key, value)
        else:
            super(JITParticle, self).__setattr__(key, value)


class JITParticleSet(ParticleSet):
    """Container class for storing and executing over sets of
    particles using Just-in-Time (JIT) compilation techniques.

    Please note that this currently only supports fixed size particle
    sets.

    :param size: Initial size of particle set
    :param grid: Grid object from which to sample velocity
    :param pclass: Optional class object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    """

    def __init__(self, size, grid, pclass=JITParticle, lon=None, lat=None):
        self._grid = grid
        self.ptype = ParticleType(pclass.user_vars)
        self._particles = np.empty(size, dtype=pclass)
        self._particle_data = np.empty(size, dtype=self.ptype.dtype)

        for i in range(size):
            self._particles[i] = pclass(lon[i], lat[i], grid=grid,
                                        cptr=self._particle_data[i])

    def advect(self, timesteps=1, dt=None):
        print "Parcels::JITParticleSet: Advecting %d particles for %d timesteps" \
            % (len(self), timesteps)

        # Generate, compile and execute JIT kernel
        self._kernel = Kernel("particle_kernel")
        self._kernel.generate_code(self._grid, ptype=self.ptype)
        self._kernel.compile(compiler=GNUCompiler())
        self._kernel.load_lib()
        self._kernel.execute(self, timesteps, dt)
