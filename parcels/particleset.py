from parcels.kernel import Kernel
from parcels.particle import JITParticle
from parcels.compiler import GNUCompiler
from parcels.kernels.advection import AdvectionRK4
from parcels.particlefile import ParticleFile
from parcels.tools.loggers import logger
from parcels.grid import GridCode
from parcels.field import NestedField, SummedField
import numpy as np
import progressbar
import time as time_module
import collections
from datetime import timedelta as delta
from datetime import datetime, date

__all__ = ['ParticleSet']


class ParticleSet(object):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets.

    :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
    :param pclass: Optional :mod:`parcels.particle.JITParticle` or
                 :mod:`parcels.particle.ScipyParticle` object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    :param depth: Optional list of initial depth values for particles. Default is 0m
    :param time: Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
    :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
           It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
           and np.float64 if the interpolation method is 'cgrid_velocity'
    Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(self, fieldset, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        self.fieldset = fieldset
        self.fieldset.check_complete()

        def convert_to_list(var):
            # Convert numpy arrays and single integers/floats to one-dimensional lists
            if isinstance(var, (int, float)):
                return [var]
            elif isinstance(var, np.ndarray):
                return var.flatten()
            return var
        if lon is None:
            lon = []
        if lat is None:
            lat = []

        lon = convert_to_list(lon)
        lat = convert_to_list(lat)
        if depth is None:
            mindepth, _ = self.fieldset.gridset.dimrange('depth')
            depth = np.ones(len(lon)) * mindepth
        depth = convert_to_list(depth)
        assert len(lon) == len(lat) and len(lon) == len(depth), (
            'lon, lat, depth don''t all have the same lenghts')

        time = time.tolist() if isinstance(time, np.ndarray) else time
        time = [time] * len(lat) if not isinstance(time, list) else time
        time = [np.datetime64(t) if isinstance(t, datetime) else t for t in time]
        time = [np.datetime64(t) if isinstance(t, date) else t for t in time]
        self.time_origin = fieldset.time_origin
        if len(time) > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = [self.time_origin.reltime(t) if isinstance(t, np.datetime64) else t for t in time]

        for kwvar in kwargs:
            kwargs[kwvar] = convert_to_list(kwargs[kwvar])

        assert len(lon) == len(time), (
            'time and positions (lon, lat, depth) don''t have the same '
            'lengths.')

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        if self.repeatdt:
            if self.repeatdt <= 0:
                raise('Repeatdt should be > 0')
            if time[0] and not np.allclose(time, time[0]):
                raise ('All Particle.time should be the same when repeatdt is not None')
            self.repeat_starttime = time[0]
            self.repeatlon = lon
            self.repeatlat = lat
            self.repeatdepth = depth
            self.repeatpclass = pclass
            self.repeatkwargs = kwargs

        if lonlatdepth_dtype is None:
            self.lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
        else:
            self.lonlatdepth_dtype = lonlatdepth_dtype
        assert self.lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'
        JITParticle.set_lonlatdepth_dtype(self.lonlatdepth_dtype)

        size = len(lon)
        self.particles = np.empty(size, dtype=pclass)
        self.ptype = pclass.getPType()
        self.kernel = None

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
            assert size == len(lon) and size == len(lat), (
                'Size of ParticleSet does not match lenght of lon and lat.')

            for i in range(size):
                self.particles[i] = pclass(lon[i], lat[i], fieldset=fieldset, depth=depth[i], cptr=cptr(i), time=time[i])
                # Set other Variables if provided
                for kwvar in kwargs:
                    if not hasattr(self.particles[i], kwvar):
                        raise RuntimeError('Particle class does not have Variable %s' % kwvar)
                    setattr(self.particles[i], kwvar, kwargs[kwvar][i])
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")

    @classmethod
    def from_list(cls, fieldset, pclass, lon, lat, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from lists of lon and lat

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param lon: List of initial longitude values for particles
        :param lat: List of initial latitude values for particles
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional list of start time values for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
       """
        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, repeatdt=repeatdt, lonlatdepth_dtype=lonlatdepth_dtype, **kwargs)

    @classmethod
    def from_line(cls, fieldset, pclass, start, finish, size, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None):
        """Initialise the ParticleSet from start/finish coordinates with equidistant spacing
        Note that this method uses simple numpy.linspace calls and does not take into account
        great circles, so may not be a exact on a globe

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start: Starting point for initialisation of particles on a straight line.
        :param finish: End point for initialisation of particles on a straight line.
        :param size: Initial size of particle set
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional start time value for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        lon = np.linspace(start[0], finish[0], size)
        lat = np.linspace(start[1], finish[1], size)
        if type(depth) in [int, float]:
            depth = [depth] * size
        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, repeatdt=repeatdt, lonlatdepth_dtype=lonlatdepth_dtype)

    @classmethod
    def from_field(cls, fieldset, pclass, start_field, size, mode='monte_carlo', depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None):
        """Initialise the ParticleSet randomly drawn according to distribution from a field

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start_field: Field for initialising particles stochastically (horizontally)  according to the presented density field.
        :param size: Initial size of particle set
        :param mode: Type of random sampling. Currently only 'monte_carlo' is implemented
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional start time value for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        if mode == 'monte_carlo':
            if start_field.interp_method == 'cgrid_tracer':
                p_interior = np.squeeze(start_field.data[0, 1:, 1:])
            else:  # if A-grid
                d = start_field.data
                p_interior = (d[0, :-1, :-1] + d[0, 1:, :-1] + d[0, :-1, 1:] + d[0, 1:, 1:])/4.
                p_interior = np.where(d[0, :-1, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, 1:] == 0, 0, p_interior)
                p_interior = np.where(d[0, :-1, 1:] == 0, 0, p_interior)
            p = np.reshape(p_interior, (1, p_interior.size))
            inds = np.random.choice(p_interior.size, size, replace=True, p=p[0] / np.sum(p))
            xsi = np.random.uniform(size=len(inds))
            eta = np.random.uniform(size=len(inds))
            j, i = np.unravel_index(inds, p_interior.shape)
            grid = start_field.grid
            if grid.gtype in [GridCode.RectilinearZGrid, GridCode.RectilinearSGrid]:
                lon = grid.lon[i] + xsi * (grid.lon[i + 1] - grid.lon[i])
                lat = grid.lat[j] + eta * (grid.lat[j + 1] - grid.lat[j])
            else:
                lons = np.array([grid.lon[j, i], grid.lon[j, i+1], grid.lon[j+1, i+1], grid.lon[j+1, i]])
                if grid.mesh == 'spherical':
                    lons[1:] = np.where(lons[1:] - lons[0] > 180, lons[1:]-360, lons[1:])
                    lons[1:] = np.where(-lons[1:] + lons[0] > 180, lons[1:]+360, lons[1:])
                lon = (1-xsi)*(1-eta) * lons[0] +\
                    xsi*(1-eta) * lons[1] +\
                    xsi*eta * lons[2] +\
                    (1-xsi)*eta * lons[3]
                lat = (1-xsi)*(1-eta) * grid.lat[j, i] +\
                    xsi*(1-eta) * grid.lat[j, i+1] +\
                    xsi*eta * grid.lat[j+1, i+1] +\
                    (1-xsi)*eta * grid.lat[j+1, i]
        else:
            raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt)

    @staticmethod
    def lonlatdepth_dtype_from_field_interp_method(field):
        if type(field) in [SummedField, NestedField]:
            for f in field:
                if f.interp_method == 'cgrid_velocity':
                    return np.float64
        else:
            if field.interp_method == 'cgrid_velocity':
                return np.float64
        return np.float32

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
        if not isinstance(particles, collections.Iterable):
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
        if isinstance(indices, collections.Iterable):
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

    def execute(self, pyfunc=AdvectionRK4, endtime=None, runtime=None, dt=1.,
                moviedt=None, recovery=None, output_file=None, movie_background_field=None,
                verbose_progress=None):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.Kernel` object.
                       Kernels can be concatenated using the + operator
        :param endtime: End time for the timestepping loop.
                        It is either a datetime object or a positive double.
        :param runtime: Length of the timestepping loop. Use instead of endtime.
                        It is either a timedelta object or a positive double.
        :param dt: Timestep interval to be passed to the kernel.
                   It is either a timedelta object or a double.
                   Use a negative value for a backward-in-time simulation.
        :param moviedt:  Interval for inner sub-timestepping (leap), which dictates
                         the update frequency of animation.
                         It is either a timedelta object or a positive double.
                         None value means no animation.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param recovery: Dictionary with additional `:mod:parcels.tools.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
        :param movie_background_field: field plotted as background in the movie if moviedt is set.
                                       'vector' shows the velocity as a vector field.
        :param verbose_progress: Boolean for providing a progress bar for the kernel execution loop.

        """

        # check if pyfunc has changed since last compile. If so, recompile
        if self.kernel is None or (self.kernel.pyfunc is not pyfunc and self.kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, Kernel):
                self.kernel = pyfunc
            else:
                self.kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self.ptype.uses_jit:
                self.kernel.remove_lib()
                cppargs = ['-DDOUBLE_COORD_VARIABLES'] if self.lonlatdepth_dtype == np.float64 else None
                self.kernel.compile(compiler=GNUCompiler(cppargs=cppargs))
                self.kernel.load_lib()

        # Convert all time variables to seconds
        if isinstance(endtime, delta):
            raise RuntimeError('endtime must be either a datetime or a double')
        if isinstance(endtime, datetime):
            endtime = np.datetime64(endtime)
        if isinstance(endtime, np.datetime64):
            if self.time_origin.calendar is None:
                raise NotImplementedError('If fieldset.time_origin is not a date, execution endtime must be a double')
            endtime = self.time_origin.reltime(endtime)
        if isinstance(runtime, delta):
            runtime = runtime.total_seconds()
        if isinstance(dt, delta):
            dt = dt.total_seconds()
        outputdt = output_file.outputdt if output_file else np.infty
        if isinstance(outputdt, delta):
            outputdt = outputdt.total_seconds()
        if isinstance(moviedt, delta):
            moviedt = moviedt.total_seconds()

        assert runtime is None or runtime >= 0, 'runtime must be positive'
        assert outputdt is None or outputdt >= 0, 'outputdt must be positive'
        assert moviedt is None or moviedt >= 0, 'moviedt must be positive'

        # Set particle.time defaults based on sign of dt, if not set at ParticleSet construction
        for p in self:
            if np.isnan(p.time):
                mintime, maxtime = self.fieldset.gridset.dimrange('time_full')
                p.time = mintime if dt >= 0 else maxtime

        # Derive _starttime and endtime from arguments or fieldset defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')
        _starttime = min([p.time for p in self]) if dt >= 0 else max([p.time for p in self])
        if self.repeatdt is not None and self.repeat_starttime is None:
            self.repeat_starttime = _starttime
        if runtime is not None:
            endtime = _starttime + runtime * np.sign(dt)
        elif endtime is None:
            mintime, maxtime = self.fieldset.gridset.dimrange('time_full')
            endtime = maxtime if dt >= 0 else mintime

        if abs(endtime-_starttime) < 1e-5 or dt == 0 or runtime == 0:
            dt = 0
            runtime = 0
            endtime = _starttime
            logger.warning_once("dt or runtime are zero, or endtime is equal to Particle.time. "
                                "The kernels will be executed once, without incrementing time")

        # Initialise particle timestepping
        for p in self:
            p.dt = dt

        # First write output_file, because particles could have been added
        if output_file:
            output_file.write(self, _starttime)
        if moviedt:
            self.show(field=movie_background_field, show_time=_starttime, animation=True)

        if moviedt is None:
            moviedt = np.infty
        time = _starttime
        if self.repeatdt:
            next_prelease = self.repeat_starttime + (abs(time - self.repeat_starttime) // self.repeatdt + 1) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.infty if dt > 0 else - np.infty
        next_output = time + outputdt if dt > 0 else time - outputdt
        next_movie = time + moviedt if dt > 0 else time - moviedt
        next_input = self.fieldset.computeTimeChunk(time, np.sign(dt))

        tol = 1e-12
        if verbose_progress is None:
            walltime_start = time_module.time()
        if verbose_progress:
            try:
                pbar = progressbar.ProgressBar(max_value=abs(endtime - _starttime)).start()
            except:  # for old versions of progressbar
                pbar = progressbar.ProgressBar(maxvalue=abs(endtime - _starttime)).start()
        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:
            if verbose_progress is None and time_module.time() - walltime_start > 10:
                # Showing progressbar if runtime > 10 seconds
                pbar = progressbar.ProgressBar(max_value=abs(endtime - _starttime)).start()
                verbose_progress = True
            if dt > 0:
                time = min(next_prelease, next_input, next_output, next_movie, endtime)
            else:
                time = max(next_prelease, next_input, next_output, next_movie, endtime)
            self.kernel.execute(self, endtime=time, dt=dt, recovery=recovery, output_file=output_file)
            if abs(time-next_prelease) < tol:
                pset_new = ParticleSet(fieldset=self.fieldset, time=time, lon=self.repeatlon,
                                       lat=self.repeatlat, depth=self.repeatdepth,
                                       pclass=self.repeatpclass, lonlatdepth_dtype=self.lonlatdepth_dtype,
                                       **self.repeatkwargs)
                for p in pset_new:
                    p.dt = dt
                self.add(pset_new)
                next_prelease += self.repeatdt * np.sign(dt)
            if abs(time-next_output) < tol:
                if output_file:
                    output_file.write(self, time)
                next_output += outputdt * np.sign(dt)
            if abs(time-next_movie) < tol:
                self.show(field=movie_background_field, show_time=time, animation=True)
                next_movie += moviedt * np.sign(dt)
            next_input = self.fieldset.computeTimeChunk(time, dt)
            if dt == 0:
                break
            if verbose_progress:
                pbar.update(abs(time - _starttime))

        if output_file:
            output_file.write(self, time)
        if verbose_progress:
            pbar.finish()

    def show(self, with_particles=True, show_time=None, field=None, domain=None, projection=None,
             land=True, vmin=None, vmax=None, savefile=None, animation=False, **kwargs):
        """Method to 'show' a Parcels ParticleSet

        :param with_particles: Boolean whether to show particles
        :param show_time: Time at which to show the ParticleSet
        :param field: Field to plot under particles (either None, a Field object, or 'vector')
        :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
        :param projection: type of cartopy projection to use (default PlateCarree)
        :param land: Boolean whether to show land. This is ignored for flat meshes
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        :param animation: Boolean whether result is a single plot, or an animation
        """
        from parcels.plotting import plotparticles
        plotparticles(particles=self, with_particles=with_particles, show_time=show_time, field=field, domain=domain,
                      projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=savefile, animation=animation, **kwargs)

    def density(self, field=None, particle_val=None, relative=False, area_scale=False):
        """Method to calculate the density of particles in a ParticleSet from their locations,
        through a 2D histogram.

        :param field: Optional :mod:`parcels.field.Field` object to calculate the histogram
                      on. Default is `fieldset.U`
        :param particle_val: Optional numpy-array of values to weigh each particle with,
                             or string name of particle variable to use weigh particles with.
                             Default is None, resulting in a value of 1 for each particle
        :param relative: Boolean to control whether the density is scaled by the total
                         weight of all particles. Default is False
        :param area_scale: Boolean to control whether the density is scaled by the area
                           (in m^2) of each grid cell. Default is False
        """

        field = field if field else self.fieldset.U
        if isinstance(particle_val, str):
            particle_val = [getattr(p, particle_val) for p in self.particles]
        else:
            particle_val = particle_val if particle_val else np.ones(len(self.particles))
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)

        for pi, p in enumerate(self.particles):
            try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                    raise('error')
                xi = p.xi[field.igrid]
                yi = p.yi[field.igrid]
            except:
                _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
            density[yi, xi] += particle_val[pi]

        if relative:
            density /= np.sum(particle_val)

        if area_scale:
            density /= field.cell_areas()

        return density

    def Kernel(self, pyfunc, c_include=""):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet"""
        return Kernel(self.fieldset, self.ptype, pyfunc=pyfunc, c_include=c_include)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFile(*args, particleset=self, **kwargs)
