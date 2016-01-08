from parcels.kernel import Kernel
from parcels.compiler import GNUCompiler
import numpy as np
import netCDF4
from collections import OrderedDict

__all__ = ['Particle', 'ParticleSet', 'JITParticle', 'JITParticleSet',
           'ParticleFile', 'AdvectionRK4']


def AdvectionRK4(particle, grid, dt):
    f = dt / 1000. / 1.852 / 60.
    u1 = grid.U[particle.lon, particle.lat]
    v1 = grid.V[particle.lon, particle.lat]
    lon1, lat1 = (particle.lon + u1*.5*f, particle.lat + v1*.5*f)
    u2, v2 = (grid.U[lon1, lat1], grid.V[lon1, lat1])
    lon2, lat2 = (particle.lon + u2*.5*f, particle.lat + v2*.5*f)
    u3, v3 = (grid.U[lon2, lat2], grid.V[lon2, lat2])
    lon3, lat3 = (particle.lon + u3*f, particle.lat + v3*f)
    u4, v4 = (grid.U[lon3, lat3], grid.V[lon3, lat3])
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f


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
                AdvectionRK4(p, self._grid, dt)


class ParticleType(object):
    """Class encapsulating the type information for custom particles

    :param user: Optional list of (name, dtype) tuples for custom variables
    """

    def __init__(self, pclass):
        if not isinstance(pclass, type):
            raise TypeError("Class object required to derive ParticleType")
        if not issubclass(pclass, Particle):
            raise TypeError("Class object does not inherit from parcels.Particle")

        self.name = pclass.__name__
        self.base = pclass.base_vars
        self.user = pclass.user_vars or {}

    def __repr__(self):
        return self.name

    @property
    def var_types(self):
        return self.base.items() + self.user.items()

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        return np.dtype(self.var_types)


class JITParticle(Particle):
    """Particle class for JIT-based Particle objects

    Users should extend this type for custom particles with fast
    advection computation. Additional variables need to be defined
    via the :user_vars: list of (name, dtype) tuples.

    :param user_vars: Class variable that defines additional particle variables
    """

    base_vars = OrderedDict([('lon', np.float32), ('lat', np.float32),
                             ('xi', np.int32), ('yi', np.int32)])
    user_vars = {}

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
        self.ptype = ParticleType(pclass)
        self._particles = np.empty(size, dtype=pclass)
        self._particle_data = np.empty(size, dtype=self.ptype.dtype)
        self.kernel = None

        for i in range(size):
            self._particles[i] = pclass(lon[i], lat[i], grid=grid,
                                        cptr=self._particle_data[i])

    def advect(self, timesteps=1, dt=None, pyfunc=AdvectionRK4):
        print "Parcels::JITParticleSet: Advecting %d particles for %d timesteps" \
            % (len(self), timesteps)

        if self.kernel is None:
            # Generate and compile JIT kernel
            self.kernel = Kernel(pyfunc.__name__)
            self.kernel.generate_code(self._grid, self.ptype, pyfunc=pyfunc)
            self.kernel.compile(compiler=GNUCompiler())
            self.kernel.load_lib()

        # Execute JIT kernel
        self.kernel.execute(self, timesteps, dt)


class ParticleFile(object):

    def __init__(self, name, particleset):
        """Initialise netCDF4.Dataset for trajectory output.

        The output follows the format outlined in the Discrete
        Sampling Geometries section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries

        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

        Developer note: We cannot use xray.Dataset here, since it does
        not yet allow incremental writes to disk:
        https://github.com/xray/xray/issues/199
        """
        self.dataset = netCDF4.Dataset("%s.nc" % name, "w", format="NETCDF4")
        self.dataset.createDimension("obs", None)
        self.dataset.createDimension("trajectory", particleset.size)
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"

        # Create ID variable according to CF conventions
        self.trajectory = self.dataset.createVariable("trajectory", "i4", ("trajectory",))
        self.trajectory.long_name = "Unique identifier for each particle"
        self.trajectory.cf_role = "trajectory_id"
        self.trajectory[:] = np.arange(particleset.size, dtype=np.int32)

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", ("trajectory", "obs"), fill_value=0.)
        self.time.long_name = ""
        self.time.standard_name = "time"
        self.time.units = "seconds since 1970-01-01 00:00:00 0:00"
        self.time.calendar = "julian"
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", ("trajectory", "obs"), fill_value=0.)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", ("trajectory", "obs"), fill_value=0.)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", ("trajectory", "obs"), fill_value=0.)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        self.idx = 0

    def __del__(self):
        self.dataset.close()

    def write(self, data, time):
        if isinstance(data, ParticleSet):
            # Write multiple particles at once
            pset = data
            self.time[:, self.idx] = time
            self.lat[:, self.idx] = np.array([p.lat for p in pset])
            self.lon[:, self.idx] = np.array([p.lon for p in pset])
            self.z[:, self.idx] = np.zeros(pset.size, dtype=np.float32)
        else:
            raise TypeError("NetCDF output is only enabled for ParticleSet obects")

        self.idx += 1
