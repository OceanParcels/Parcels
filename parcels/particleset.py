from parcels.kernel import Kernel, KernelOp as op
from parcels.field import Field
from parcels.particle import JITParticle
from parcels.compiler import GNUCompiler
from parcels.kernels.advection import AdvectionRK4
from parcels.particlefile import ParticleFile
import numpy as np
import bisect
from collections import Iterable
from datetime import timedelta as delta
from datetime import datetime
try:
    import matplotlib.pyplot as plt
except:
    plt = None
try:
    from mpl_toolkits.basemap import Basemap
except:
    Basemap = None

__all__ = ['ParticleSet']


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


def nearest_index(array, value):
    """returns index of the nearest value in array using O(log n) bisection method"""
    y = bisect.bisect(array, value)
    if y == len(array):
        return y-1
    elif(abs(array[y-1] - value) < abs(array[y] - value)):
        return y-1
    else:
        return y


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
        self.ptype = pclass.getPType()
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
                self.particles[i] = pclass(lon[i], lat[i], grid=grid, cptr=cptr(i), time=grid.time[0])
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
            # Update C-pointer on particles
            for p, pdata in zip(self.particles, self._particle_data):
                p._cptr = pdata

    def remove(self, indices):
        if isinstance(indices, Iterable):
            particles = [self.particles[i] for i in indices]
        else:
            particles = self.particles[indices]
        if self.ptype.uses_jit:
            self._particle_data = np.delete(self._particle_data, indices)
            # Update C-pointer on particles
            for p, pdata in zip(self.particles, self._particle_data):
                p._cptr = pdata
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
            endtime = starttime + runtime
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

    def show(self, **kwargs):
        savefile = kwargs.get('savefile', None)
        field = kwargs.get('field', True)
        domain = kwargs.get('domain', None)
        particles = kwargs.get('particles', True)
        plon = np.array([p.lon for p in self])
        plat = np.array([p.lat for p in self])
        time = [p.time for p in self]
        t = kwargs.get('t', time[0])
        if isinstance(t, datetime):
            t = (t - self.grid.U.time_origin).total_seconds()
        if isinstance(t, delta):
            t = t.total_seconds()
        if domain is not None:
            latN = nearest_index(self.grid.U.lat, domain[0])
            latS = nearest_index(self.grid.U.lat, domain[1])
            lonE = nearest_index(self.grid.U.lon, domain[2])
            lonW = nearest_index(self.grid.U.lon, domain[3])
        else:
            latN, latS, lonE, lonW = (-1, 0, -1, 0)
        if field is not 'vector':
            t = int(t)
            if plt is None:
                raise RuntimeError("Visualisation not possible: matplotlib not found!")
            field = kwargs.get('field', True)
            plt.ion()
            plt.clf()
            if particles:
                plt.plot(np.transpose(plon), np.transpose(plat), 'ko')
            if field is True:
                axes = plt.gca()
                axes.set_xlim([self.grid.U.lon[lonW], self.grid.U.lon[lonE]])
                axes.set_ylim([self.grid.U.lat[latS], self.grid.U.lat[latN]])
                namestr = ''
                time_origin = self.grid.U.time_origin
            else:
                if not isinstance(field, Field):
                    field = getattr(self.grid, field)
                field.show(with_particles=True, **dict(kwargs, t=t))
                namestr = field.name
                time_origin = field.time_origin
            if time_origin is 0:
                timestr = ' after ' + str(delta(seconds=t)) + ' hours'
            else:
                timestr = ' on ' + str(time_origin + delta(seconds=t))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
        else:
            if Basemap is None:
                raise RuntimeError("Visualisation not possible: Basemap module not found!")
            land = kwargs.get('land', False)
            vmax = kwargs.get('vmax', None)
            time_origin = self.grid.U.time_origin
            idx = self.grid.U.time_index(t)
            U = np.array(self.grid.U.temporal_interpolate_fullfield(idx, t))
            V = np.array(self.grid.V.temporal_interpolate_fullfield(idx, t))
            lon = self.grid.U.lon
            lat = self.grid.U.lat
            lon = lon[lonW:lonE]
            lat = lat[latS:latN]
            U = U[latS:latN, lonW:lonE]
            V = V[latS:latN, lonW:lonE]

            # configuring plot
            lat_median = np.median(lat)
            lon_median = np.median(lon)
            plt.figure()
            m = Basemap(projection='merc', lat_0=lat_median, lon_0=lon_median,
                        resolution='h', area_thresh=100,
                        llcrnrlon=lon[0], llcrnrlat=lat[0],
                        urcrnrlon=lon[-1], urcrnrlat=lat[-1])
            if land:
                m.drawcoastlines()
                m.fillcontinents(color='burlywood')
            parallels = np.arange(lat[0], lat[-1], abs(lat[0]-lat[-1])/5)
            parallels = np.around(parallels, 2)
            m.drawparallels(parallels, labels=[1, 0, 0, 0])
            meridians = np.arange(lon[0], lon[-1], abs(lon[0]-lon[-1])/5)
            meridians = np.around(meridians, 2)
            m.drawmeridians(meridians, labels=[0, 0, 0, 1])

            # formating velocity data for quiver plotting
            U = np.array([U[y, x] for x in range(len(lon)) for y in range(len(lat))])
            V = np.array([V[y, x] for x in range(len(lon)) for y in range(len(lat))])
            speed = np.sqrt(U**2 + V**2)
            normU = U/speed
            normV = V/speed
            x = np.repeat(lon, len(lat))
            y = np.tile(lat, len(lon))

            # plotting velocity vector field
            vecs = m.quiver(x, y, normU, normV, speed, cmap=plt.cm.gist_ncar, clim=[0, vmax], scale=50, latlon=True)
            m.colorbar(vecs, "right", size="5%", pad="2%")
            # plotting particle data
            if particles:
                xs, ys = m(plon, plat)
                m.scatter(xs, ys, color='black')

        if time_origin is 0:
            timestr = ' after ' + str(delta(seconds=t)) + ' hours'
        else:
            timestr = ' on ' + str(time_origin + delta(seconds=t))

        if particles:
            if field:
                plt.title('Particles' + timestr)
            elif field is 'vector':
                plt.title('Particles and velocity field' + timestr)
            else:
                plt.title('Particles and '+namestr + timestr)
        else:
            if field is 'vector':
                plt.title('Velocity field' + timestr)
            else:
                plt.title(namestr + timestr)

        if savefile is None:
            plt.show()
            plt.pause(0.0001)
        else:
            plt.savefig(savefile)
            print('Plot saved to '+savefile+'.png')
            plt.close()

    def Kernel(self, pyfunc):
        return Kernel(self.grid, self.ptype, pyfunc=pyfunc)

    def ParticleFile(self, *args, **kwargs):
        return ParticleFile(*args, particleset=self, **kwargs)
