from parcels.kernel import Kernel
from parcels.field import Field, UnitConverter
from parcels.particle import JITParticle
from parcels.compiler import GNUCompiler
from parcels.kernels.advection import AdvectionRK4
from parcels.particlefile import ParticleFile
from parcels.loggers import logger
import numpy as np
import bisect
from collections import Iterable
from datetime import timedelta as delta
from datetime import datetime

__all__ = ['ParticleSet']


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

    Please note that this currently only supports fixed size particle sets.

    :param grid: :mod:`parcels.grid.Grid` object from which to sample velocity
    :param pclass: Optional :mod:`parcels.particle.JITParticle` or
                 :mod:`parcels.particle.ScipyParticle` object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    :param time: Optional list of initial time values for particles. Default is grid.U.time[0]
    """

    def __init__(self, grid, pclass=JITParticle, lon=None, lat=None, time=None):
        # Convert numpy arrays to one-dimensional lists
        lon = lon.flatten() if isinstance(lon, np.ndarray) else lon
        lat = lat.flatten() if isinstance(lat, np.ndarray) else lat
        assert len(lon) == len(lat)

        time = grid.U.time[0] if time is None else time
        time = time.flatten() if isinstance(time, np.ndarray) else time
        time = [time] * len(lat) if not isinstance(time, list) else time
        assert len(lon) == len(time)

        size = len(lon)
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

        if lon is not None and lat is not None:
            # Initialise from lists of lon/lat coordinates
            assert(size == len(lon) and size == len(lat))

            for i in range(size):
                self.particles[i] = pclass(lon[i], lat[i], grid=grid, cptr=cptr(i), time=time[i])
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")

    @classmethod
    def from_list(cls, grid, pclass, lon, lat, time=None):
        """Initialise the ParticleSet from lists of lon and lat

        :param grid: :mod:`parcels.grid.Grid` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param lon: List of initial longitude values for particles
        :param lat: List of initial latitude values for particles
        :param time: Optional list of start time values for particles. Default is grid.U.time[0]
       """
        return cls(grid=grid, pclass=pclass, lon=lon, lat=lat, time=time)

    @classmethod
    def from_line(cls, grid, pclass, start, finish, size, time=None):
        """Initialise the ParticleSet from start/finish coordinates with equidistant spacing
        Note that this method uses simple numpy.linspace calls and does not take into account
        great circles, so may not be a exact on a globe

        :param grid: :mod:`parcels.grid.Grid` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start: Starting point for initialisation of particles on a straight line.
        :param finish: End point for initialisation of particles on a straight line.
        :param size: Initial size of particle set
        :param time: Optional start time value for particles. Default is grid.U.time[0]
        """
        lon = np.linspace(start[0], finish[0], size, dtype=np.float32)
        lat = np.linspace(start[1], finish[1], size, dtype=np.float32)
        return cls(grid=grid, pclass=pclass, lon=lon, lat=lat, time=time)

    @classmethod
    def from_field(cls, grid, pclass, start_field, size, mode='monte_carlo', time=None):
        """Initialise the ParticleSet randomly drawn according to distribution from a field

        :param grid: :mod:`parcels.grid.Grid` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start_field: Field for initialising particles stochastically according to the presented density field.
        :param size: Initial size of particle set
        :param mode: Type of random sampling. Currently only 'monte_carlo' is implemented
        :param time: Optional start time value for particles. Default is grid.U.time[0]
        """
        total = np.sum(start_field.data[0, :, :])
        start_field.data[0, :, :] = start_field.data[0, :, :] / total
        lonwidth = (start_field.lon[1] - start_field.lon[0]) / 2
        latwidth = (start_field.lat[1] - start_field.lat[0]) / 2

        def add_jitter(pos, width, min, max):
            value = pos + np.random.uniform(-width, width)
            while not (min <= value <= max):
                value = pos + np.random.uniform(-width, width)
            return value

        if mode == 'monte_carlo':
            probs = np.random.uniform(size=size)
            lon = []
            lat = []
            for p in probs:
                cell = np.unravel_index(np.where([p < i for i in np.cumsum(start_field.data[0, :, :])])[0][0],
                                        np.shape(start_field.data[0, :, :]))
                lon.append(add_jitter(start_field.lon[cell[1]], lonwidth,
                                      start_field.lon.min(), start_field.lon.max()))
                lat.append(add_jitter(start_field.lat[cell[0]], latwidth,
                                      start_field.lat.min(), start_field.lat.max()))
        else:
            raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

        return cls(grid=grid, pclass=pclass, lon=lon, lat=lat, time=time)

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
        """Method to add particles to the ParticleSet"""
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
        """Method to remove particles from the ParticleSet, based on their `indices`"""
        if isinstance(indices, Iterable):
            particles = [self.particles[i] for i in indices]
        else:
            particles = self.particles[indices]
        self.particles = np.delete(self.particles, indices)
        if self.ptype.uses_jit:
            self._particle_data = np.delete(self._particle_data, indices)
            # Update C-pointer on particles
            for p, pdata in zip(self.particles, self._particle_data):
                p._cptr = pdata
        return particles

    def execute(self, pyfunc=AdvectionRK4, starttime=None, endtime=None, dt=1.,
                runtime=None, interval=None, recovery=None, output_file=None,
                show_movie=False):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.Kernel` object.
                       Kernels can be concatenated using the + operator
        :param starttime: Starting time for the timestepping loop. Defaults to 0.0.
        :param endtime: End time for the timestepping loop
        :param runtime: Length of the timestepping loop. Use instead of endtime.
        :param dt: Timestep interval to be passed to the kernel
        :param interval: Interval for inner sub-timestepping (leap), which dictates
                         the update frequency of file output and animation.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param recovery: Dictionary with additional `:mod:parcels.kernels.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
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
            starttime = self.grid.U.time[0] if dt > 0 else self.grid.U.time[-1]
        if runtime is not None:
            if runtime < 0:
                runtime = np.abs(runtime)
                print("negating runtime  as it has to be positive")
            endtime = starttime + runtime * np.sign(dt)
        else:
            if endtime is None:
                endtime = self.grid.U.time[-1] if dt > 0 else self.grid.U.time[0]
        if interval is None:
            interval = endtime - starttime

        # Ensure that dt and interval have the correct sign
        if endtime > starttime:  # Time-forward mode
            if dt < 0:
                dt *= -1.
                logger.warning("Negating dt because running in time-forward mode")
            if interval < 0:
                interval *= -1.
                logger.warning("Negating interval because running in time-forward mode")
        if endtime < starttime:  # Time-backward mode
            if dt > 0.:
                dt *= -1.
                logger.warning("Negating dt because running in time-backward mode")
            if interval > 0.:
                interval *= -1.
                logger.warning("Negating interval because running in time-backward mode")

        # Initialise particle timestepping
        for p in self:
            p.time = starttime
            p.dt = dt
        # Execute time loop in sub-steps (timeleaps)
        timeleaps = int((endtime - starttime) / interval)
        assert(timeleaps >= 0)
        leaptime = starttime
        for _ in range(timeleaps):
            # First write output_file, because particles could have been added
            if output_file:
                output_file.write(self, leaptime)
            if show_movie:
                self.show(field=show_movie, show_time=leaptime)
            leaptime += interval
            self.kernel.execute(self, endtime=leaptime, dt=dt,
                                recovery=recovery)
        # Write out a final output_file
        if output_file:
            output_file.write(self, leaptime)

    def show(self, particles=True, show_time=None, field=True, domain=None,
             land=False, vmin=None, vmax=None, savefile=None):
        """Method to 'show' a Parcels ParticleSet

        :param particles: Boolean whether to show particles
        :param show_time: Time at which to show the ParticleSet
        :param field: Field to plot under particles (either True, a Field object, or 'vector')
        :param domain: Four-vector (latN, latS, lonE, lonW) defining domain to show
        :param land: Boolean whether to show land (in field='vector' mode only)
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        """
        try:
            import matplotlib.pyplot as plt
        except:
            logger.info("Visualisation is not possible. Matplotlib not found.")
            return
        try:
            from mpl_toolkits.basemap import Basemap
        except:
            Basemap = None

        plon = np.array([p.lon for p in self])
        plat = np.array([p.lat for p in self])
        show_time = self[0].time if show_time is None else show_time
        if isinstance(show_time, datetime):
            show_time = (show_time - self.grid.U.time_origin).total_seconds()
        if isinstance(show_time, delta):
            show_time = show_time.total_seconds()
        if domain is not None:
            latN = nearest_index(self.grid.U.lat, domain[0])
            latS = nearest_index(self.grid.U.lat, domain[1])
            lonE = nearest_index(self.grid.U.lon, domain[2])
            lonW = nearest_index(self.grid.U.lon, domain[3])
        else:
            latN, latS, lonE, lonW = (-1, 0, -1, 0)
        if field is not 'vector':
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
                field.show(with_particles=True, show_time=show_time, vmin=vmin, vmax=vmax)
                namestr = field.name
                time_origin = field.time_origin
            if time_origin is 0:
                timestr = ' after ' + str(delta(seconds=show_time)) + ' hours'
            else:
                timestr = ' on ' + str(time_origin + delta(seconds=show_time))
            xlbl = 'Zonal distance [m]' if type(self.grid.U.units) is UnitConverter else 'Longitude [degrees]'
            ylbl = 'Meridional distance [m]' if type(self.grid.U.units) is UnitConverter else 'Latitude [degrees]'
            plt.xlabel(xlbl)
            plt.ylabel(ylbl)
        elif Basemap is None:
            logger.info("Visualisation is not possible. Basemap not found.")
        else:
            time_origin = self.grid.U.time_origin
            idx = self.grid.U.time_index(show_time)
            U = np.array(self.grid.U.temporal_interpolate_fullfield(idx, show_time))
            V = np.array(self.grid.V.temporal_interpolate_fullfield(idx, show_time))
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
            vecs = m.quiver(x, y, normU, normV, speed, cmap=plt.cm.gist_ncar, clim=[vmin, vmax], scale=50, latlon=True)
            m.colorbar(vecs, "right", size="5%", pad="2%")
            # plotting particle data
            if particles:
                xs, ys = m(plon, plat)
                m.scatter(xs, ys, color='black')

        if time_origin is 0:
            timestr = ' after ' + str(delta(seconds=show_time)) + ' hours'
        else:
            timestr = ' on ' + str(time_origin + delta(seconds=show_time))

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
            logger.info('Plot saved to '+savefile+'.png')
            plt.close()

    def density(self, field=None, particle_val=None, relative=False, area_scale=True):
        """Method to calculate the density of particles in a ParticleSet from their locations,
        through a 2D histogram

        :param field: Optional :mod:`parcels.field.Field` object to calculate the histogram
                    on. Default is `grid.U`
        :param particle_val: Optional list of values to weigh each particlewith
        :param relative: Boolean to control whether the density is scaled by the total
                    number of particles
        :param area_scale: Boolean to control whether the density is scaled by the area
                    (in m^2) of each grid cell"""
        lons = [p.lon for p in self.particles]
        lats = [p.lat for p in self.particles]
        # Code for finding nearest vertex for each particle is currently very inefficient
        # once cell tracking is implemented for SciPy particles, the below use of np.min/max
        # will be replaced (see PR #111)
        if field is not None:
            # Kick out particles that are not within the limits of our density field
            half_lon = (field.lon[1] - field.lon[0])/2
            half_lat = (field.lat[1] - field.lat[0])/2
            dparticles = (lons > (np.min(field.lon)-half_lon)) * (lons < (np.max(field.lon)+half_lon)) * \
                         (lats > (np.min(field.lat)-half_lat)) * (lats < (np.max(field.lat)+half_lat))
            dparticles = np.where(dparticles)[0]
        else:
            field = self.grid.U
            dparticles = range(len(self.particles))
        Density = np.zeros((field.lon.size, field.lat.size), dtype=np.float32)

        # For each particle, find closest vertex in x and y and add 1 or val to the count
        if particle_val is not None:
            for p in dparticles:
                Density[np.argmin(np.abs(lons[p] - field.lon)), np.argmin(np.abs(lats[p] - field.lat))] \
                    += getattr(self.particles[p], particle_val)
        else:
            for p in dparticles:
                nearest_lon = np.argmin(np.abs(lons[p] - field.lon))
                nearest_lat = np.argmin(np.abs(lats[p] - field.lat))
                Density[nearest_lon, nearest_lat] += 1
            if relative:
                Density /= len(dparticles)

        if area_scale:
            area = np.zeros(np.shape(field.data[0, :, :]), dtype=np.float32)
            U = self.grid.U
            V = self.grid.V
            dy = (V.lon[1] - V.lon[0])/V.units.to_target(1, V.lon[0], V.lat[0])
            for y in range(len(U.lat)):
                dx = (U.lon[1] - U.lon[0])/U.units.to_target(1, U.lon[0], U.lat[y])
                area[y, :] = dy * dx
            # Scale by cell area
            Density /= np.transpose(area)

        return Density

    def Kernel(self, pyfunc):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `grid` and `ptype` of the ParticleSet"""
        return Kernel(self.grid, self.ptype, pyfunc=pyfunc)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFile(*args, particleset=self, **kwargs)
