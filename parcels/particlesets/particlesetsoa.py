from ctypes import Structure, POINTER  # noqa
import time as time_module
from datetime import date
from datetime import datetime
from datetime import timedelta as delta

import sys
import numpy as np
import xarray as xr
from operator import attrgetter  # noqa

from parcels.compiler import GNUCompiler
from parcels.field import Field  # noqa
from parcels.grid import GridCode
from parcels.kernel import Kernel
from parcels.kernels.advection import AdvectionRK4
from parcels.particle import JITParticle
from parcels.particlefile import ParticleFile
from .baseparticleset import BaseParticleSet
from .soa import ParticleCollectionSOA
from parcels.tools.converters import _get_cftime_calendars
from parcels.tools.statuscodes import OperationCode
from parcels.tools.loggers import logger
try:
    from mpi4py import MPI
except:
    MPI = None
if MPI:
    try:
        from sklearn.cluster import KMeans  # noqa
    except:
        raise EnvironmentError('sklearn needs to be available if MPI is installed. '
                               'See http://oceanparcels.org/#parallel_install for more information')

__all__ = ['ParticleSet']


def _to_write_particles(pd, time):
    """We don't want to write a particle that is not started yet.
    Particle will be written if particle.time is between time-dt/2 and time+dt/2
    """
    return (np.less_equal(time - np.abs(pd['dt']/2), pd['time'], where=np.isfinite(pd['time']))
            & np.greater(time + np.abs(pd['dt']/2), pd['time'], where=np.isfinite(pd['time']))
            & (np.isfinite(pd['id']))
            & (np.isfinite(pd['time'])))


def convert_to_array(var):
    # Convert lists and single integers/floats to one-dimensional numpy arrays
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.int32)):
        return np.array([var])
    else:
        return np.array(var)


def convert_to_reltime(time):
    if isinstance(time, np.datetime64) or (hasattr(time, 'calendar') and time.calendar in _get_cftime_calendars()):
        return True
    return False


class ParticleSetSOA(BaseParticleSet):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets.

    :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
           While fieldset=None is supported, this will throw a warning as it breaks most Parcels functionality
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
    :param partitions: List of cores on which to distribute the particles for MPI runs. Default: None, in which case particles
           are distributed automatically on the processors
    Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    # ==== already user-exposed ==== #
    def __init__(self, fieldset=None, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        super(ParticleSetSOA, self).__init__()
        self.fieldset = fieldset
        if self.fieldset is None:
            logger.warning_once("No FieldSet provided in ParticleSet generation. "
                                "This breaks most Parcels functionality")
        else:
            self.fieldset.check_complete()
        partitions = kwargs.pop('partitions', None)

        lon = np.empty(shape=0) if lon is None else convert_to_array(lon)
        lat = np.empty(shape=0) if lat is None else convert_to_array(lat)

        if isinstance(pid_orig, (type(None), type(False))):
            pid_orig = np.arange(lon.size)
        # pid = pid_orig + pclass.lastID

        if depth is None:
            mindepth = self.fieldset.gridset.dimrange('depth')[0] if self.fieldset is not None else 0
            depth = np.ones(lon.size) * mindepth
        else:
            depth = convert_to_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        time = convert_to_array(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time

        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        self.time_origin = fieldset.time_origin if self.fieldset is not None else 0
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = np.array([self.time_origin.reltime(t) if convert_to_reltime(t) else t for t in time])
        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        if lonlatdepth_dtype is None:
            if fieldset is not None:
                lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
            else:
                lonlatdepth_dtype = np.float32
        assert lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'

        # if partitions is not None and partitions is not False:
        #     partitions = convert_to_array(partitions)

        for kwvar in kwargs:
            kwargs[kwvar] = convert_to_array(kwargs[kwvar])
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwvar)

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

        ngrids = fieldset.gridset.size if fieldset is not None else 1
        self._collection = ParticleCollectionSOA(pclass, lon, lat, depth, time, lonlatdepth_dtype, partitions, pid_orig, ngrids, **kwargs)

        # offset = np.max(pid) if len(pid) > 0 else -1
        # if MPI:
        #     mpi_comm = MPI.COMM_WORLD
        #     mpi_rank = mpi_comm.Get_rank()
        #     mpi_size = mpi_comm.Get_size()

        #     if lon.size < mpi_size and mpi_size > 1:
        #         raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

        #     if mpi_size > 1:
        #         if partitions is not False:
        #             if partitions is None:
        #                 if mpi_rank == 0:
        #                     coords = np.vstack((lon, lat)).transpose()
        #                     kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
        #                     partitions = kmeans.labels_
        #                 else:
        #                     partitions = None
        #                 partitions = mpi_comm.bcast(partitions, root=0)
        #             elif np.max(partitions) >= mpi_size:
        #                 raise RuntimeError('Particle partitions must vary between 0 and the number of mpi procs')
        #             lon = lon[partitions == mpi_rank]
        #             lat = lat[partitions == mpi_rank]
        #             time = time[partitions == mpi_rank]
        #             depth = depth[partitions == mpi_rank]
        #             pid = pid[partitions == mpi_rank]
        #             for kwvar in kwargs:
        #                 kwargs[kwvar] = kwargs[kwvar][partitions == mpi_rank]
        #         offset = MPI.COMM_WORLD.allreduce(offset, op=MPI.MAX)

        # pclass.setLastID(offset+1)
        # pclass.set_lonlatdepth_dtype(self.lonlatdepth_dtype)
        # self.ptype = pclass.getPType()

        if self.repeatdt:
            # self.repeatpid = pid - pclass.lastID  # was computed with pid+pclass.lastID, thus pid=pid_init=pd_orig
            self.repeatpid = pid_orig
            self.partitions = self.collection.pu_indicators()

        self.kernel = None

        # store particle data as an array per variable (structure of arrays approach)
        # self.particle_data = {}
        # initialised = set()
        # for v in self.ptype.variables:
        #     if v.name in ['xi', 'yi', 'zi', 'ti']:
        #         ngrid = fieldset.gridset.size if fieldset is not None else 1
        #         self.particle_data[v.name] = np.empty((len(lon), ngrid), dtype=v.dtype)
        #     else:
        #         self.particle_data[v.name] = np.empty(len(lon), dtype=v.dtype)

        # if lon is not None and lat is not None:
        #     # Initialise from lists of lon/lat coordinates
        #     assert self.size == len(lon) and self.size == len(lat), (
        #         'Size of ParticleSet does not match length of lon and lat.')

        #     # mimic the variables that get initialised in the constructor
        #     self.particle_data['lat'][:] = lat
        #     self.particle_data['lon'][:] = lon
        #     self.particle_data['depth'][:] = depth
        #     self.particle_data['time'][:] = time
        #     self.particle_data['id'][:] = pid
        #     self.particle_data['fileid'][:] = -1

        #     # special case for exceptions which can only be handled from scipy
        #     self.particle_data['exception'] = np.empty(self.size, dtype=object)

        #     initialised |= {'lat', 'lon', 'depth', 'time', 'id'}

        #     # any fields that were provided on the command line
        #     for kwvar, kwval in kwargs.items():
        #         if not hasattr(pclass, kwvar):
        #             raise RuntimeError('Particle class does not have Variable %s' % kwvar)
        #         self.particle_data[kwvar][:] = kwval
        #         initialised.add(kwvar)

        #     # initialise the rest to their default values
        #     for v in self.ptype.variables:
        #         if v.name in initialised:
        #             continue

        #         if isinstance(v.initial, Field):
        #             for i in range(self.size):
        #                 if np.isnan(time[i]):
        #                     raise RuntimeError('Cannot initialise a Variable with a Field if no time provided. '
        #                                        'Add a "time=" to ParticleSet construction')
        #                 v.initial.fieldset.computeTimeChunk(time[i], 0)
        #                 self.particle_data[v.name][i] = v.initial[
        #                     time[i], depth[i], lat[i], lon[i]
        #                 ]
        #                 logger.warning_once("Particle initialisation from field can be very slow as it is computed in scipy mode.")
        #         elif isinstance(v.initial, attrgetter):
        #             self.particle_data[v.name][:] = v.initial(self)
        #         else:
        #             self.particle_data[v.name][:] = v.initial

        #         initialised.add(v.name)
        # else:
        #     raise ValueError("Latitude and longitude required for generating ParticleSet")

    def __getattr__(self, name):
        # Comment CK: this either a member function of the accessor or the collection - not the PSet itself
        if 'particle_data' in self.__dict__ and name in self.__dict__['particle_data']:
            return self.__dict__['particle_data'][name]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            return False

    # ==== already user-exposed ==== #
    def __getitem__(self, index):
        # Comment CK: that what we have the iterator or accessor over the collection for -> definitely not a top-level PSet function
        # Comment RB: The collection should provide this function indeed. Until we made a (more) definitive decision on how we want
        #             this to be interfaced, forward this to the collection.
        return self._collection[index]

    def cstruct(self):
        """
        'cstruct' returns the ctypes mapping of the combined collections cstruct and the fieldset cstruct.
        This depends on the specific structure in question.
        """
        cstruct = self._collection.cstruct()
        return cstruct

    @property
    def ctypes_struct(self):
        return self.cstruct()

    @classmethod
    def monte_carlo_sample(cls, start_field, size, mode='monte_carlo'):
        """
        Converts a starting field into a monte-carlo sample of lons and lats.

        :param start_field: :mod:`parcels.fieldset.Field` object for initialising particles stochastically (horizontally)  according to the presented density field.

        returns list(lon), list(lat)
        """
        if mode == 'monte_carlo':
            data = start_field.data if isinstance(start_field.data, np.ndarray) else np.array(start_field.data)
            if start_field.interp_method == 'cgrid_tracer':
                p_interior = np.squeeze(data[0, 1:, 1:])
            else:  # if A-grid
                d = data
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
            lon, lat = []
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
            return list(lon), list(lat)
        else:
            raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

    # ==== already user-exposed ==== #
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
        lon, lat = cls.monte_carlo_sample(start_field, mode)

        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time,
                   lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt)

    # ==== already user-exposed ==== #
    @classmethod
    def from_particlefile(cls, fieldset, pclass, filename, restart=True, restarttime=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from a netcdf ParticleFile.
        This creates a new ParticleSet based on locations of all particles written
        in a netcdf ParticleFile at a certain time. Particle IDs are preserved if restart=True

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param filename: Name of the particlefile from which to read initial conditions
        :param restart: Boolean to signal if pset is used for a restart (default is True).
               In that case, Particle IDs are preserved.
        :param restarttime: time at which the Particles will be restarted. Default is the last time written.
               Alternatively, restarttime could be a time value (including np.datetime64) or
               a callable function such as np.nanmin. The last is useful when running with dt < 0.
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        if repeatdt is not None:
            logger.warning('Note that the `repeatdt` argument is not retained from %s, and that '
                           'setting a new repeatdt will start particles from the _new_ particle '
                           'locations.' % filename)

        pfile = xr.open_dataset(str(filename), decode_cf=True)
        pfile_vars = [v for v in pfile.data_vars]

        vars = {}
        to_write = {}
        for v in pclass.getPType().variables:
            if v.name in pfile_vars:
                vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
            elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt', 'depth', 'id', 'fileid', 'state'] \
                    and v.to_write:
                raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
            to_write[v.name] = v.to_write
        vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
        vars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

        if isinstance(vars['time'][0, 0], np.timedelta64):
            vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

        if restarttime is None:
            restarttime = np.nanmax(vars['time'])
        elif callable(restarttime):
            restarttime = restarttime(vars['time'])
        else:
            restarttime = restarttime

        inds = np.where(vars['time'] == restarttime)
        for v in vars:
            if to_write[v] is True:
                vars[v] = vars[v][inds]
            elif to_write[v] == 'once':
                vars[v] = vars[v][inds[0]]
            if v not in ['lon', 'lat', 'depth', 'time', 'id']:
                kwargs[v] = vars[v]

        if restart:
            pclass.setLastID(0)  # reset to zero offset
        else:
            vars['id'] = None

        return cls(fieldset=fieldset, pclass=pclass, lon=vars['lon'], lat=vars['lat'],
                   depth=vars['depth'], time=vars['time'], pid_orig=vars['id'],
                   lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, **kwargs)

    def to_dict(self, pfile, time, deleted_only=False):
        """
        Convert all Particle data from one time step to a python dictionary.
        :param pfile: ParticleFile object requesting the conversion
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt,
         and one for all variables to be written once
        """
        data_dict = {}
        data_dict_once = {}

        time = time.total_seconds() if isinstance(time, delta) else time

        pd = self.collection._data

        if pfile.lasttime_written != time and \
           (pfile.write_ondelete is False or deleted_only is not False):
            if pd['id'].size == 0:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)
            else:
                if deleted_only is not False:
                    to_write = deleted_only
                else:
                    to_write = _to_write_particles(pd, time)
                if np.any(to_write) > 0:
                    for var in pfile.var_names:
                        data_dict[var] = pd[var][to_write]
                    pfile.maxid_written = max(pfile.maxid_written, np.max(data_dict['id']))

                pset_errs = (to_write & (pd['state'] != OperationCode.Delete)
                             & np.less(1e-3, np.abs(time - pd['time']), where=np.isfinite(pd['time'])))
                if np.count_nonzero(pset_errs) > 0:
                    logger.warning_once(
                        'time argument in pfile.write() is {}, but particles have time {}'.format(time, pd['time'][pset_errs]))

                if time not in pfile.time_written:
                    pfile.time_written.append(time)

                if len(pfile.var_names_once) > 0:
                    first_write = (_to_write_particles(pd, time)
                                   & np.isin(pd['id'], pfile.written_once, invert=True))
                    if np.any(first_write):
                        data_dict_once['id'] = pd['id'][first_write]
                        for var in pfile.var_names_once:
                            data_dict_once[var] = pd[var][first_write]
                        pfile.written_once.extend(pd['id'][first_write])

            if deleted_only is False:
                pfile.lasttime_written = time

        return data_dict, data_dict_once

    # ==== already user-exposed ==== #
    @property
    def size(self):
        # ==== to change at some point - len and size are different things ==== #
        return len(self._collection)

    # ==== already user-exposed ==== #
    def __repr__(self):
        return "\n".join([str(p) for p in self])

    # ==== already user-exposed ==== #
    def __len__(self):
        return len(self._collection)

    # ==== already user-exposed ==== #
    def __sizeof__(self):
        return sys.getsizeof(self._collection)

    # ==== already user-exposed ==== #
    def __iadd__(self, particles):
        self.add(particles)
        return self

    # ==== already user-exposed ==== #
    def add(self, particles):
        """Method to add particles to the ParticleSet"""
        # Method forward to new implementation
        # Note that this is implemented as an incremental add!
        self._collection += particles
        return self

    # ==== to be removed later ==== #
    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`"""
        # Method forward to new implementation
        self._collection.remove_multi_by_indices(indices)

    # ==== to be removed later ==== #
    def remove_booleanvector(self, indices):
        """Method to remove particles from the ParticleSet, based on an array of booleans"""
        # Method forward
        self.remove_indices(np.where(indices))

    # ==== already user-exposed ==== #
    def execute(self, pyfunc=AdvectionRK4, endtime=None, runtime=None, dt=1.,
                moviedt=None, recovery=None, output_file=None, movie_background_field=None,
                verbose_progress=None, postIterationCallbacks=None, callbackdt=None):
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
        :param postIterationCallbacks: (Optional) Array of functions that are to be called after each iteration (post-process, non-Kernel)
        :param callbackdt: (Optional, in conjecture with 'postIterationCallbacks) timestep inverval to (latestly) interrupt the running kernel and invoke post-iteration callbacks from 'postIterationCallbacks'
        """
        # ==== TODO: when all the restructuring is done, it should be possible to move this to the base class ==== #

        # check if pyfunc has changed since last compile. If so, recompile
        if self.kernel is None or (self.kernel.pyfunc is not pyfunc and self.kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, Kernel):
                self.kernel = pyfunc
            else:
                self.kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self.collection.ptype.uses_jit:
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
        if isinstance(callbackdt, delta):
            callbackdt = callbackdt.total_seconds()

        assert runtime is None or runtime >= 0, 'runtime must be positive'
        assert outputdt is None or outputdt >= 0, 'outputdt must be positive'
        assert moviedt is None or moviedt >= 0, 'moviedt must be positive'

        mintime, maxtime = self.fieldset.gridset.dimrange('time_full') if self.fieldset is not None else (0, 1)
        if np.any(np.isnan(self.collection._data['time'])):
            self.collection._data['time'][np.isnan(self.collection._data['time'])] = mintime if dt >= 0 else maxtime

        # Derive _starttime and endtime from arguments or fieldset defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')
        _starttime = self.collection._data['time'].min() if dt >= 0 else self.collection._data['time'].max()
        if self.repeatdt is not None and self.repeat_starttime is None:
            self.repeat_starttime = _starttime
        if runtime is not None:
            endtime = _starttime + runtime * np.sign(dt)
        elif endtime is None:
            mintime, maxtime = self.fieldset.gridset.dimrange('time_full')
            endtime = maxtime if dt >= 0 else mintime

        execute_once = False
        if abs(endtime-_starttime) < 1e-5 or dt == 0 or runtime == 0:
            dt = 0
            runtime = 0
            endtime = _starttime
            logger.warning_once("dt or runtime are zero, or endtime is equal to Particle.time. "
                                "The kernels will be executed once, without incrementing time")
            execute_once = True

        self.collection._data['dt'][:] = dt

        # First write output_file, because particles could have been added
        if output_file:
            output_file.write(self, _starttime)
        if moviedt:
            self.show(field=movie_background_field, show_time=_starttime, animation=True)

        if moviedt is None:
            moviedt = np.infty
        if callbackdt is None:
            interupt_dts = [np.infty, moviedt, outputdt]
            if self.repeatdt is not None:
                interupt_dts.append(self.repeatdt)
            callbackdt = np.min(np.array(interupt_dts))
        time = _starttime
        if self.repeatdt:
            next_prelease = self.repeat_starttime + (abs(time - self.repeat_starttime) // self.repeatdt + 1) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.infty if dt > 0 else - np.infty
        next_output = time + outputdt if dt > 0 else time - outputdt
        next_movie = time + moviedt if dt > 0 else time - moviedt
        next_callback = time + callbackdt if dt > 0 else time - callbackdt
        next_input = self.fieldset.computeTimeChunk(time, np.sign(dt)) if self.fieldset is not None else np.inf

        tol = 1e-12
        if verbose_progress is None:
            walltime_start = time_module.time()
        if verbose_progress:
            pbar = self.__create_progressbar(_starttime, endtime)
        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:
            if verbose_progress is None and time_module.time() - walltime_start > 10:
                # Showing progressbar if runtime > 10 seconds
                if output_file:
                    logger.info('Temporary output files are stored in %s.' % output_file.tempwritedir_base)
                    logger.info('You can use "parcels_convert_npydir_to_netcdf %s" to convert these '
                                'to a NetCDF file during the run.' % output_file.tempwritedir_base)
                pbar = self.__create_progressbar(_starttime, endtime)
                verbose_progress = True
            if dt > 0:
                time = min(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            else:
                time = max(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            self.kernel.execute(self, endtime=time, dt=dt, recovery=recovery, output_file=output_file,
                                execute_once=execute_once)
            if abs(time-next_prelease) < tol:
                pset_new = ParticleSet(fieldset=self.fieldset, time=time, lon=self.repeatlon,
                                       lat=self.repeatlat, depth=self.repeatdepth,
                                       pclass=self.repeatpclass, lonlatdepth_dtype=self.lonlatdepth_dtype,
                                       partitions=False, pid_orig=self.repeatpid, **self.repeatkwargs)
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
            # ==== insert post-process here to also allow for memory clean-up via external func ==== #
            if abs(time-next_callback) < tol:
                if postIterationCallbacks is not None:
                    for extFunc in postIterationCallbacks:
                        extFunc()
                next_callback += callbackdt * np.sign(dt)
            if time != endtime:
                next_input = self.fieldset.computeTimeChunk(time, dt)
            if dt == 0:
                break
            if verbose_progress:
                pbar.update(abs(time - _starttime))

        if output_file:
            output_file.write(self, time)
        if verbose_progress:
            pbar.finish()

    # ==== already user-exposed ==== #
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

    # ==== already user-exposed ==== #
    def density(self, field_name=None, particle_val=None, relative=False, area_scale=False):
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

        field_name = field_name if field_name else "U"
        field = getattr(self.fieldset, field_name)

        f_str = """
def search_kernel(particle, fieldset, time):
    x = fieldset.{}[time, particle.depth, particle.lat, particle.lon]
        """.format(field_name)

        k = Kernel(
            self.fieldset,
            self.ptype,
            funcname="search_kernel",
            funcvars=["particle", "fieldset", "time", "x"],
            funccode=f_str,
        )
        self.execute(pyfunc=k, runtime=0)

        if isinstance(particle_val, str):
            particle_val = self.collection._data[particle_val]
        else:
            particle_val = particle_val if particle_val else np.ones(self.size)
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)

        for i, p in enumerate(self):
            try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                    raise('error')
                xi = p.xi[field.igrid]
                yi = p.yi[field.igrid]
            except:
                _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
            density[yi, xi] += particle_val[i]

        if relative:
            density /= np.sum(particle_val)

        if area_scale:
            density /= field.cell_areas()

        return density

    # ==== already user-exposed ==== #
    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet
        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        """
        return Kernel(self.fieldset, self.collection.ptype, pyfunc=pyfunc, c_include=c_include,
                      delete_cfiles=delete_cfiles)

    # ==== already user-exposed ==== #
    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFile(*args, particleset=self, **kwargs)

    def set_variable_write_status(self, var, write_status):
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param status: Write status of the variable (True, False or 'once')
        """
        # Method forward (for now)
        # Method forward (shall stay) - CK
        self._collection.set_variable_write_status(var, write_status)


ParticleSet = ParticleSetSOA
