from parcels.kernel import Kernel, KernelOp
from parcels.field import Field
from parcels.compiler import GNUCompiler
import numpy as np
import netCDF4
from collections import OrderedDict, Iterable
from datetime import timedelta as delta
from datetime import datetime
import math
try:
    import matplotlib.pyplot as plt
except:
    plt = None


__all__ = ['Particle', 'ParticleSet', 'JITParticle',
           'ParticleFile', 'AdvectionRK4', 'AdvectionEE', 'AdvectionRK45']


def AdvectionRK4(particle, grid, time, dt):
    u1 = grid.U[time, particle.lon, particle.lat]
    v1 = grid.V[time, particle.lon, particle.lat]
    lon1, lat1 = (particle.lon + u1*.5*dt, particle.lat + v1*.5*dt)
    u2, v2 = (grid.U[time + .5 * dt, lon1, lat1], grid.V[time + .5 * dt, lon1, lat1])
    lon2, lat2 = (particle.lon + u2*.5*dt, particle.lat + v2*.5*dt)
    u3, v3 = (grid.U[time + .5 * dt, lon2, lat2], grid.V[time + .5 * dt, lon2, lat2])
    lon3, lat3 = (particle.lon + u3*dt, particle.lat + v3*dt)
    u4, v4 = (grid.U[time + dt, lon3, lat3], grid.V[time + dt, lon3, lat3])
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * dt


def AdvectionEE(particle, grid, time, dt):
    u1 = grid.U[time, particle.lon, particle.lat]
    v1 = grid.V[time, particle.lon, particle.lat]
    particle.lon += u1 * dt
    particle.lat += v1 * dt


def AdvectionRK45(particle, grid, time, dt):
    tol = [1e-9]
    c = [1./4., 3./8., 12./13., 1., 1./2.]
    A = [[1./4., 0., 0., 0., 0.],
         [3./32., 9./32., 0., 0., 0.],
         [1932./2197., -7200./2197., 7296./2197., 0., 0.],
         [439./216., -8., 3680./513., -845./4104., 0.],
         [-8./27., 2., -3544./2565., 1859./4104., -11./40.]]
    b4 = [25./216., 0., 1408./2565., 2197./4104., -1./5.]
    b5 = [16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.]

    u1 = grid.U[time, particle.lon, particle.lat]
    v1 = grid.V[time, particle.lon, particle.lat]
    lon1, lat1 = (particle.lon + u1 * A[0][0] * dt,
                  particle.lat + v1 * A[0][0] * dt)
    u2, v2 = (grid.U[time + c[0] * dt, lon1, lat1],
              grid.V[time + c[0] * dt, lon1, lat1])
    lon2, lat2 = (particle.lon + (u1 * A[1][0] + u2 * A[1][1]) * dt,
                  particle.lat + (v1 * A[1][0] + v2 * A[1][1]) * dt)
    u3, v3 = (grid.U[time + c[1] * dt, lon2, lat2],
              grid.V[time + c[1] * dt, lon2, lat2])
    lon3, lat3 = (particle.lon + (u1 * A[2][0] + u2 * A[2][1] + u3 * A[2][2]) * dt,
                  particle.lat + (v1 * A[2][0] + v2 * A[2][1] + v3 * A[2][2]) * dt)
    u4, v4 = (grid.U[time + c[2] * dt, lon3, lat3],
              grid.V[time + c[2] * dt, lon3, lat3])
    lon4, lat4 = (particle.lon + (u1 * A[3][0] + u2 * A[3][1] + u3 * A[3][2] + u4 * A[3][3]) * dt,
                  particle.lat + (v1 * A[3][0] + v2 * A[3][1] + v3 * A[3][2] + v4 * A[3][3]) * dt)
    u5, v5 = (grid.U[time + c[3] * dt, lon4, lat4],
              grid.V[time + c[3] * dt, lon4, lat4])
    lon5, lat5 = (particle.lon + (u1 * A[4][0] + u2 * A[4][1] + u3 * A[4][2] + u4 * A[4][3] + u5 * A[4][4]) * dt,
                  particle.lat + (v1 * A[4][0] + v2 * A[4][1] + v3 * A[4][2] + v4 * A[4][3] + v5 * A[4][4]) * dt)
    u6, v6 = (grid.U[time + c[4] * dt, lon5, lat5],
              grid.V[time + c[4] * dt, lon5, lat5])

    lon_4th = particle.lon + (u1 * b4[0] + u2 * b4[1] + u3 * b4[2] + u4 * b4[3] + u5 * b4[4]) * dt
    lat_4th = particle.lat + (v1 * b4[0] + v2 * b4[1] + v3 * b4[2] + v4 * b4[3] + v5 * b4[4]) * dt
    lon_5th = particle.lon + (u1 * b5[0] + u2 * b5[1] + u3 * b5[2] + u4 * b5[3] + u5 * b5[4] + u6 * b5[5]) * dt
    lat_5th = particle.lat + (v1 * b5[0] + v2 * b5[1] + v3 * b5[2] + v4 * b5[3] + v5 * b5[4] + v6 * b5[5]) * dt

    kappa = math.sqrt(math.pow(lon_5th - lon_4th, 2) + math.pow(lat_5th - lat_4th, 2))
    if kappa <= dt * tol[0]:
        particle.lon = lon_4th
        particle.lat = lat_4th
        if kappa <= dt * tol[0] / 10:
            particle.dt *= 2
        return KernelOp.SUCCESS
    else:
        particle.dt /= 2
        return KernelOp.FAILURE


def positions_from_density_field(pnum, field, mode='monte_carlo'):
    """Initialise particles from a given density field"""
    print("Initialising particles from " + field.name + " field")
    total = np.sum(field.data[0, :, :])
    field.data[0, :, :] = field.data[0, :, :] / total
    lonwidth = (field.lon[1] - field.lon[0]) / 2
    latwidth = (field.lat[1] - field.lat[0]) / 2

    def add_jitter(pos, width, min, max):
        value = pos + np.random.uniform(-width, width)
        while not (min <= value <= max):
            value = pos + np.random.uniform(-width, width)
        return value

    if mode == 'monte_carlo':
        probs = np.random.uniform(size=pnum)
        lon = []
        lat = []
        for p in probs:
            cell = np.unravel_index(np.where([p < i for i in np.cumsum(field.data[0, :, :])])[0][0],
                                    np.shape(field.data[0, :, :]))
            lon.append(add_jitter(field.lon[cell[1]], lonwidth,
                                  field.lon.min(), field.lon.max()))
            lat.append(add_jitter(field.lat[cell[0]], latwidth,
                                  field.lat.min(), field.lat.max()))
    else:
        raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

    return lon, lat


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


class ParticleSet(object):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle
    sets.

    :param size: Initial size of particle set
    :param grid: Grid object from which to sample velocity
    :param pclass: Optional class object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    :param start: Optional starting point for initilisation of particles
                 on a straight line. Use start/finish instead of lat/lon.
    :param finish: Optional end point for initilisation of particles on a
                 straight line. Use start/finish instead of lat/lon.
    :param start_field: Optional field for initialising particles stochastically
                 according to the presented density field. Use instead of lat/lon.
    """

    def __init__(self, size, grid, pclass=JITParticle,
                 lon=None, lat=None, start=None, finish=None, start_field=None):
        self.grid = grid
        self.particles = np.empty(size, dtype=pclass)
        self.ptype = ParticleType(pclass)
        self.kernel = None
        self.time_origin = grid.U.time_origin

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

    def execute(self, pyfunc=AdvectionRK4, starttime=None, endtime=None, dt=1.,
                runtime=None, interval=None, output_file=None, tol=None,
                show_movie=False):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel funtion to execute. This can be the name of a
                       defined Python function of a parcels.Kernel.
        :param starttime: Starting time for the timestepping loop. Defaults to 0.0.
        :param endtime: End time for the timestepping loop
        :param runtime: Length of the timestepping loop. Use instead of endtime.
        :param dt: Timestep interval to be passed to the kernel
        :param interval: Interval for inner sub-timestepping (leap), which dictates
                         the update frequency of file output and animation.
        :param output_file: ParticleFile object for particle output
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

        # Convert all time variables to seconds
        if isinstance(starttime, delta):
            starttime = starttime.total_seconds()
        if isinstance(endtime, delta):
            endtime = endtime.total_seconds()
        if isinstance(runtime, delta):
            runtime = runtime.total_seconds()
        if isinstance(dt, delta):
            dt = dt.total_seconds()
        if isinstance(interval, delta):
            interval = interval.total_seconds()
        if isinstance(starttime, datetime):
            starttime = (starttime - self.time_origin).total_seconds()
        if isinstance(endtime, datetime):
            endtime = (endtime - self.time_origin).total_seconds()

        # Derive starttime, endtime and interval from arguments or grid defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')
        if starttime is None:
            starttime = self.grid.time[0] if dt > 0 else self.grid.time[-1]
        if runtime is not None:
            endtime = starttime + runtime if dt > 0 else starttime - runtime
        else:
            if endtime is None:
                endtime = self.grid.time[-1] if dt > 0 else self.grid.time[0]
        if interval is None:
            interval = endtime - starttime

        # Ensure that dt and interval have the correct sign
        if endtime > starttime:  # Time-forward mode
            if dt < 0:
                dt *= -1.
                print("negating dt because running in time-forward mode")
            if interval < 0:
                interval *= -1.
                print("negating interval because running in time-forward mode")
        if endtime < starttime:  # Time-backward mode
            if dt > 0.:
                dt *= -1.
                print("negating dt because running in time-backward mode")
            if interval > 0.:
                interval *= -1.
                print("negating interval because running in time-backward mode")

        # Initialise particle timestepping
        for p in self:
            p.time = starttime
            p.dt = dt
        # Execute time loop in sub-steps (timeleaps)
        timeleaps = int((endtime - starttime) / interval)
        assert(timeleaps >= 0)
        leaptime = starttime
        for _ in range(timeleaps):
            leaptime += interval
            self.kernel.execute(self, endtime=leaptime, dt=dt)
            if output_file:
                output_file.write(self, leaptime)
            if show_movie:
                self.show(field=show_movie, t=leaptime)
        # Remove deactivated particles
        to_remove = [i for i, p in enumerate(self.particles) if p.active == 0]
        if len(to_remove) > 0:
            self.remove(to_remove)

    def show(self, **kwargs):
        if plt is None:
            raise RuntimeError("Visualisation not possible: matplotlib not found!")

        field = kwargs.get('field', True)
        lon = [p.lon for p in self]
        lat = [p.lat for p in self]
        time = [p.time for p in self]
        t = int(kwargs.get('t', time[0]))
        plt.ion()
        plt.clf()
        plt.plot(np.transpose(lon), np.transpose(lat), 'ko')
        if field is True:
            axes = plt.gca()
            axes.set_xlim([self.grid.U.lon[0], self.grid.U.lon[-1]])
            axes.set_ylim([self.grid.U.lat[0], self.grid.U.lat[-1]])
            namestr = ''
            time_origin = self.grid.U.time_origin
        else:
            if not isinstance(field, Field):
                field = getattr(self.grid, field)
            field.show(with_particles=True, **dict(kwargs, t=t))
            namestr = ' on ' + field.name
            time_origin = field.time_origin
        if time_origin is 0:
            timestr = ' after ' + str(delta(seconds=t)) + ' hours'
        else:
            timestr = ' on ' + str(time_origin + delta(seconds=t))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Particles' + namestr + timestr)
        plt.show()
        plt.pause(0.0001)

    def Kernel(self, pyfunc):
        return Kernel(self.grid, self.ptype, pyfunc=pyfunc)

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
        self.time = self.dataset.createVariable("time", "f8", ("trajectory", "obs"), fill_value=np.nan)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if particleset.time_origin == 0:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(particleset.time_origin)
            self.time.calendar = "julian"
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", ("trajectory", "obs"), fill_value=np.nan)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", ("trajectory", "obs"), fill_value=np.nan)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", ("trajectory", "obs"), fill_value=np.nan)
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
