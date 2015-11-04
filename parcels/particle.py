import numpy as np
from parcels.jit_module import Kernel, GNUCompiler

__all__ = ['Particle', 'ParticleSet', 'JITParticleSet']


class Particle(object):
    """Class encapsualting the basic attributes of a particle"""

    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat
        self.xi = None
        self.yi = None

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

    def __init__(self, size, grid):
        self._grid = grid
        self._particles = np.empty(size, dtype=Particle)
        self._npart = 0

    def add_particle(self, p):
        p.xi = np.where(p.lon > self._grid.U.lon)[0][-1]
        p.yi = np.where(p.lat > self._grid.U.lat)[0][-1]
        self._particles[self._npart] = p
        self._npart += 1

    def advect(self, timesteps=1, dt=None):
        print "Parcels::ParticleSet: Advecting %d particles for %d timesteps" \
            % (self._npart, timesteps)
        for t in range(timesteps):
            for p in self._particles:
                p.advect_rk4(self._grid, dt)


class JITParticleSet(object):
    """Container class for storing and executing over sets of
    particles using Just-in-Time (JIT) compialtion techniques.

    Please note that this currently only supports fixed size particle
    sets.

    :param size: Initial size of particle set
    :param grid: Grid object from which to sample velocity"""

    def __init__(self, size, grid):
        self._grid = grid
        self._particles = np.empty(size, dtype=Particle)
        self._npart = 0

        # Particle array for JIT kernel
        self._kernel = None
        self._p_dtype = np.dtype([('lon', np.float32), ('lat', np.float32),
                                  ('xi', np.int32), ('yi', np.int32)])
        self._p_array = np.empty(size, dtype=self._p_dtype)

    def add_particle(self, p):
        p.xi = np.where(p.lon > self._grid.U.lon)[0][-1]
        p.yi = np.where(p.lat > self._grid.U.lat)[0][-1]
        self._particles[self._npart] = p

        # Populate the partcile's struct
        self._p_array[self._npart]['lon'] = p.lon
        self._p_array[self._npart]['lat'] = p.lat
        self._p_array[self._npart]['xi'] = p.xi
        self._p_array[self._npart]['yi'] = p.yi

        self._npart += 1

    def generate_jit_kernel(self, filename):
        self._kernel = Kernel(filename)
        self._kernel.generate_code(self._grid)
        self._kernel.compile(compiler=GNUCompiler())
        self._kernel.load_lib()

    def advect(self, timesteps=1, dt=None):
        print "Parcels::JITParticleSet: Advecting %d particles for %d timesteps" \
            % (self._npart, timesteps)

        # Generate, compile and execute JIT kernel
        self.generate_jit_kernel("particle_kernel")
        self._kernel.execute(self, timesteps, dt)

        # Transferrring particle data back onto original array
        for i, p in enumerate(self._particles):
            p.lon = self._p_array[i]['lon']
            p.lat = self._p_array[i]['lat']
            p.xi = self._p_array[i]['xi']
            p.yi = self._p_array[i]['yi']
