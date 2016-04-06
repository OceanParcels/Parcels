from parcels.kernel import Kernel
from parcels.field import Field
from parcels.compiler import GNUCompiler
import numpy as np
import netCDF4
from collections import OrderedDict, Iterable
import matplotlib.pyplot as plt
import math
import datetime

__all__ = ['Particle', 'ParticleSet', 'JITParticle',
           'ParticleFile', 'AdvectionRK4', 'AdvectionEE']


def AdvectionRK4(particle, grid, time, dt):
    f_lat = dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(particle.lat*math.pi/180)
    u1 = grid.U[time, particle.lon, particle.lat]
    v1 = grid.V[time, particle.lon, particle.lat]
    lon1, lat1 = (particle.lon + u1*.5*f_lon, particle.lat + v1*.5*f_lat)
    u2, v2 = (grid.U[time + .5 * dt, lon1, lat1], grid.V[time + .5 * dt, lon1, lat1])
    lon2, lat2 = (particle.lon + u2*.5*f_lon, particle.lat + v2*.5*f_lat)
    u3, v3 = (grid.U[time + .5 * dt, lon2, lat2], grid.V[time + .5 * dt, lon2, lat2])
    lon3, lat3 = (particle.lon + u3*f_lon, particle.lat + v3*f_lat)
    u4, v4 = (grid.U[time + dt, lon3, lat3], grid.V[time + dt, lon3, lat3])
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * f_lon
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * f_lat


def AdvectionEE(particle, grid, time, dt):
    f_lat = dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(particle.lat*math.pi/180)
    u1 = grid.U[time, particle.lon, particle.lat]
    v1 = grid.V[time, particle.lon, particle.lat]
    particle.lon += u1 * f_lon
    particle.lat += v1 * f_lat


def positions_from_density_field(pnum, startfield, mode='monte_carlo'):
    # initialise particles from a field
    print("Initialising particles from " + startfield.name + " field")
    total = np.sum(startfield.data[0, :, :])
    startfield.data[0, :, :] = startfield.data[0, :, :]/total
    lonwidth = (startfield.lon[1] - startfield.lon[0])/2
    latwidth = (startfield.lat[1] - startfield.lat[0])/2

    def jitter_pos(pos, width, list=[]):
        list[-1] = pos + np.random.uniform(-width, width)
        return list

    if(mode is 'monte_carlo'):
        probs = np.random.uniform(size=pnum)
        lon = []
        lat = []
        for p in probs:
            cell = np.unravel_index(np.where([p < i for i in np.cumsum(startfield.data[0, :, :])])[0][0],
                                    np.shape(startfield.data[0, :, :]))
            lon.append(None)
            while np.max(startfield.lon) > jitter_pos(startfield.lon[cell[1]], lonwidth, lon)[-1] < np.min(startfield.lon):
                pass
            lat.append(None)
            while np.max(startfield.lat) > jitter_pos(startfield.lat[cell[0]], latwidth, lat)[-1] < np.min(startfield.lat):
                pass

    return lon, lat


class Particle(object):
    """Class encapsualting the basic attributes of a particle

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param grid: :Class Grid: object to track this particle on
    :param user_vars: Dictionary of any user variables that might be defined in subclasses
    """
    user_vars = OrderedDict()

    def __init__(self, lon, lat, grid, cptr=None):
        self.lon = lon
        self.lat = lat
        self.xi = np.where(self.lon >= grid.U.lon)[0][-1]
        self.yi = np.where(self.lat >= grid.U.lat)[0][-1]
        self.active = 1

        for var in self.user_vars:
            setattr(self, var, 0)

    def __repr__(self):
        return "P(%f, %f)[%d, %d]" % (self.lon, self.lat, self.xi, self.yi)

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
        if self.uses_jit:
            self.var_types = pclass.base_vars
            self.var_types.update(pclass.user_vars)

        self.user_vars = pclass.user_vars

    def __repr__(self):
        return "PType<self.name>"

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        return np.dtype(list(self.var_types.items()))


class ParticleSet(object):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle
    sets.

    :param size: Initial size of particle set
    :param grid: Grid object from which to sample velocity
    :param pclass: Optional class object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    """

    def __init__(self, size, grid, pclass=JITParticle,
                 lon=None, lat=None, start=None, finish=None, start_field=None):
        self.grid = grid
        self.particles = np.empty(size, dtype=pclass)
        self.ptype = ParticleType(pclass)
        self.kernel = None

        if self.ptype.uses_jit:
            # Allocate underlying data for C-allocated particles
            self._particle_data = np.empty(size, dtype=self.ptype.dtype)

            def cptr(i):
                return self._particle_data[i]
        else:
            def cptr(i):
                return None

        if start is not None and finish is not None:
            # Initialise from start/finish coordinates with equidistant spacing
            assert(lon is None and lat is None)
            lon = np.linspace(start[0], finish[0], size, dtype=np.float32)
            lat = np.linspace(start[1], finish[1], size, dtype=np.float32)

        if start_field is not None:
            lon, lat = positions_from_density_field(size, start_field)

        if lon is not None and lat is not None:
            # Initialise from lists of lon/lat coordinates
            assert(size == len(lon) and size == len(lat))

            for i in range(size):
                self.particles[i] = pclass(lon[i], lat[i], grid=grid, cptr=cptr(i))
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")

    @property
    def size(self):
        return self.particles.size

    def __repr__(self):
        return "\n".join([str(p) for p in self])

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return self.particles[key]

    def __setitem__(self, key, value):
        self.particles[key] = value

    def __iadd__(self, particles):
        self.add(particles)
        return self

    def add(self, particles):
        if isinstance(particles, ParticleSet):
            particles = particles.particles
        if not isinstance(particles, Iterable):
            particles = [particles]
        self.particles = np.append(self.particles, particles)
        if self.ptype.uses_jit:
            particles_data = [p._cptr for p in particles]
            self._particle_data = np.append(self._particle_data, particles_data)

    def remove(self, indices):
        if isinstance(indices, Iterable):
            particles = [self.particles[i] for i in indices]
        else:
            particles = self.particles[indices]
        if self.ptype.uses_jit:
            self._particle_data = np.delete(self._particle_data, indices)
        self.particles = np.delete(self.particles, indices)
        return particles

    def execute(self, pyfunc=AdvectionRK4, time=None, dt=1., timesteps=1,
                output_file=None, show_movie=False, output_steps=-1):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel funtion to execute
        :param time: Starting time for the timestepping loop
        :param dt: Timestep interval to be passed to the kernel
        :param timesteps: Number of individual timesteps to execute
        :param output_file: ParticleFile object for particle output
        :param output_steps: Size of output intervals in timesteps
        :param show_movie: True shows particles; name of field plots that field as background
        """
        if self.kernel is None:
            # Generate and store Kernel
            if isinstance(pyfunc, Kernel):
                self.kernel = pyfunc
            else:
                self.kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self.ptype.uses_jit:
                self.kernel.compile(compiler=GNUCompiler())
                self.kernel.load_lib()

        # Check if output is required and compute outer leaps
        if output_file is None or output_steps <= 0:
            output_steps = timesteps
        timeleaps = int(timesteps / output_steps)
        # Execute kernel in sub-stepping intervals (leaps)
        current = time or self.grid.time[0]
        for _ in range(timeleaps):
            self.kernel.execute(self, output_steps, current, dt)
            current += output_steps * dt
            if output_file:
                output_file.write(self, current)
            if show_movie:
                self.show(field=show_movie, t=current)
        to_remove = [i for i, p in enumerate(self.particles) if p.active == 0]
        self.remove(to_remove)

    def show(self, **kwargs):
        field = kwargs.get('field', True)
        t = kwargs.get('t', 0)
        lon = [p.lon for p in self]
        lat = [p.lat for p in self]
        plt.ion()
        plt.clf()
        plt.plot(np.transpose(lon), np.transpose(lat), 'ko')
        if field is True:
            axes = plt.gca()
            axes.set_xlim([self.grid.U.lon[0], self.grid.U.lon[-1]])
            axes.set_ylim([self.grid.U.lat[0], self.grid.U.lat[-1]])
            namestr = ''
        else:
            if not isinstance(field, Field):
                field = getattr(self.grid, field)
            field.show(**kwargs)
            namestr = ' on ' + field.name
        if self.grid.U.time_origin == 0:
            timestr = ' after ' + str(datetime.timedelta(seconds=t)) + ' hours'
        else:
            timestr = ' on ' + str(self.grid.U.time_origin + datetime.timedelta(seconds=t))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Particles' + namestr + timestr)
        plt.show()
        plt.pause(0.0001)

    def Kernel(self, pyfunc):
        return Kernel(self.grid, self.ptype, pyfunc)

    def ParticleFile(self, *args, **kwargs):
        return ParticleFile(*args, particleset=self, **kwargs)


class ParticleFile(object):

    def __init__(self, name, particleset, initial_dump=True):
        """Initialise netCDF4.Dataset for trajectory output.

        The output follows the format outlined in the Discrete
        Sampling Geometries section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries

        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

        Developer note: We cannot use xray.Dataset here, since it does
        not yet allow incremental writes to disk:
        https://github.com/xray/xray/issues/199

        :param name: Basename of the output file
        :param particlset: ParticleSet to output
        :param initial_dump: Perform initial output at time 0.
        :param user_vars: A list of additional user defined particle variables to write
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

        if particleset.ptype.user_vars is not None:
            self.user_vars = particleset.ptype.user_vars.keys()
            for var in self.user_vars:
                setattr(self, var, self.dataset.createVariable(var, "f4", ("trajectory", "obs"), fill_value=0.))
                getattr(self, var).long_name = ""
                getattr(self, var).standard_name = var
                getattr(self, var).units = "unknown"
        else:
            self.user_vars = {}

        self.idx = 0

        if initial_dump:
            self.write(particleset, 0.)

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
            for var in self.user_vars:
                getattr(self, var)[:, self.idx] = np.array([getattr(p, var) for p in pset])
        else:
            raise TypeError("NetCDF output is only enabled for ParticleSet obects")

        self.idx += 1
