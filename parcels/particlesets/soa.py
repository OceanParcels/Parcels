import numpy as np

from .collections import ParticleCollection
from .iterators import BaseParticleAccessor
from .iterators import BaseParticleCollectionIterator
from parcels.particle import ScipyParticle, JITParticle
from parcels.field import Field
from parcels.tools.loggers import logger
from operator import attrgetter

try:
    from mpi4py import MPI
except:
    MPI = None
if MPI:
    try:
        from sklearn.cluster import KMeans
    except:
        raise EnvironmentError('sklearn needs to be available if MPI is installed. '
                               'See http://oceanparcels.org/#parallel_install for more information')

"""
Author: Dr. Christian Kehl
github relation: #913 (particleset_class_hierarchy)
purpose: defines all the specific functions for a ParticleCollection, ParticleAccessor, ParticleSet etc. that relates
         to a structure-of-array (SoA) data arrangement.
"""

def convert_to_flat_array(var):
    # Convert lists and single integers/floats to one-dimensional numpy arrays
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.int32)):
        return np.array([var])
    else:
        return np.array(var)


class ParticleCollectionSOA(ParticleCollection):

    def __init__(self, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, lonlatdepth_dtype=None, pid_orig=None, ngrid=1, **kwargs):
        """
        :param ngrid: number of grids in the fieldset of the overarching ParticleSet - required for initialising the
        field references of the ctypes-link of particles that are allocated
        """

        super(ParticleCollection, self).__init__()
        partitions = kwargs.pop('partitions', None)

        # lon = np.empty(shape=0) if lon is None else convert_to_flat_array(lon)  # input reformatting - particleset-task
        # lat = np.empty(shape=0) if lat is None else convert_to_flat_array(lat)  # input reformatting - particleset-task
        if isinstance(pid_orig, (type(None), type(False))):
            pid_orig = np.arange(lon.size)
        pid = pid_orig + pclass.lastID

        # -- We expect the overarching particle set to take care of the depth-is-none-so-calc-from-field case -- #
        # if depth is None:
        #     mindepth = self.fieldset.gridset.dimrange('depth')[0] if self.fieldset is not None else 0
        #     depth = np.ones(lon.size) * mindepth
        # else:
        #     depth = convert_to_array(depth)
        assert depth is not None, "particle's initial depth is None - incompatible with the collection. Invalid state."
        # depth = convert_to_flat_array(depth)  # input reformatting - particleset-task
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        # time = convert_to_flat_array(time)  # input reformatting - particleset-task
        # time = np.repeat(time, lon.size) if time.size == 1 else time  # input reformatting - particleset-task

        # -- Time field correction to be done in overarching particle set -- #
        # def _convert_to_reltime(time):
        #     if isinstance(time, np.datetime64) or (hasattr(time, 'calendar') and time.calendar in _get_cftime_calendars()):
        #         return True
        #     return False

        # if time.size > 0 and type(time[0]) in [datetime, date]:
        #     time = np.array([np.datetime64(t) for t in time])
        # self.time_origin = fieldset.time_origin if self.fieldset is not None else 0
        # if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
        #     raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        # time = np.array([self.time_origin.reltime(t) if _convert_to_reltime(t) else t for t in time])

        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        if partitions is not None and partitions is not False:
            self._pu_indicators = convert_to_flat_array(partitions)

        for kwvar in kwargs:
            # kwargs[kwvar] = convert_to_flat_array(kwargs[kwvar])  # input reformatting - particleset-task
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwvar)

        offset = np.max(pid) if len(pid) > 0 else -1
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()

            if lon.size < mpi_size and mpi_size > 1:
                raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

            if mpi_size > 1:
                if partitions is not False:
                    if self._pu_indicators is None:
                        if mpi_rank == 0:
                            coords = np.vstack((lon, lat)).transpose()
                            kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
                            self._pu_indicators = kmeans.labels_
                        else:
                            self._pu_indicators = None
                        self._pu_indicators = mpi_comm.bcast(self._pu_indicators, root=0)
                    elif np.max(self._pu_indicators) >= mpi_size:
                        raise RuntimeError('Particle partitions must vary between 0 and the number of mpi procs')
                    lon = lon[self._pu_indicators == mpi_rank]
                    lat = lat[self._pu_indicators == mpi_rank]
                    time = time[self._pu_indicators == mpi_rank]
                    depth = depth[self._pu_indicators == mpi_rank]
                    pid = pid[self._pu_indicators == mpi_rank]
                    for kwvar in kwargs:
                        kwargs[kwvar] = kwargs[kwvar][self._pu_indicators == mpi_rank]
                offset = MPI.COMM_WORLD.allreduce(offset, op=MPI.MAX)

        # -- Repeat-dt structure (and thus possibly also flat-array conversion) to be done by the particle set -- #
        # -- Makes sense that the overarching particle set reformats the input to what is necessary.           -- #
        # self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        # if self.repeatdt:
        #     if self.repeatdt <= 0:
        #         raise('Repeatdt should be > 0')
        #     if time[0] and not np.allclose(time, time[0]):
        #         raise ('All Particle.time should be the same when repeatdt is not None')
        #     self.repeat_starttime = time[0]
        #     self.repeatlon = lon
        #     self.repeatlat = lat
        #     self.repeatpid = pid - pclass.lastID
        #     self.repeatdepth = depth
        #     self.repeatpclass = pclass
        #     self.partitions = partitions
        #     self.repeatkwargs = kwargs
        pclass.setLastID(offset+1)

        if lonlatdepth_dtype is None:
            self._lonlatdepth_dtype = np.float32
        else:
            self._lonlatdepth_dtype = lonlatdepth_dtype
        assert self._lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'
        pclass.set_lonlatdepth_dtype(self._lonlatdepth_dtype)
        self._pclass = pclass

        self._ptype = pclass.getPType()
        # -- Kernel is a particle set-class thingy -- #
        # self.kernel = None

        # store particle data as an array per variable (structure of arrays approach)
        self._data = {}
        initialised = set()

        for v in self.ptype.variables:
            if v.name in ['xi', 'yi', 'zi', 'ti']:
                self._data[v.name] = np.empty((len(lon), ngrid), dtype=v.dtype)
            else:
                self._data[v.name] = np.empty(len(lon), dtype=v.dtype)

        self._ncount = self._data['lon'].shape[0]

        if lon is not None and lat is not None:
            # Initialise from lists of lon/lat coordinates
            assert self.ncount == len(lon) and self.ncount == len(lat), (
                'Size of ParticleSet does not match length of lon and lat.')

            # mimic the variables that get initialised in the constructor
            self._data['lat'][:] = lat
            self._data['lon'][:] = lon
            self._data['depth'][:] = depth
            self._data['time'][:] = time
            self._data['id'][:] = pid
            self._data['fileid'][:] = -1

            # special case for exceptions which can only be handled from scipy
            self._data['exception'] = np.empty(self.ncount, dtype=object)

            initialised |= {'lat', 'lon', 'depth', 'time', 'id'}

            # any fields that were provided on the command line
            for kwvar, kwval in kwargs.items():
                if not hasattr(pclass, kwvar):
                    raise RuntimeError('Particle class does not have Variable %s' % kwvar)
                self._data[kwvar][:] = kwval
                initialised.add(kwvar)

            # initialise the rest to their default values
            for v in self.ptype.variables:
                if v.name in initialised:
                    continue

                if isinstance(v.initial, Field):
                    for i in range(self.ncount):
                        if np.isnan(time[i]):
                            raise RuntimeError('Cannot initialise a Variable with a Field if no time provided. '
                                               'Add a "time=" to ParticleSet construction')
                        v.initial.fieldset.computeTimeChunk(time[i], 0)
                        self._data[v.name][i] = v.initial[
                            time[i], depth[i], lat[i], lon[i]
                        ]
                        logger.warning_once("Particle initialisation from field can be very slow as it is computed in scipy mode.")
                elif isinstance(v.initial, attrgetter):
                    self._data[v.name][:] = v.initial(self)
                else:
                    self._data[v.name][:] = v.initial

                initialised.add(v.name)
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")

    def __iter__(self):
        """Returns an Iterator that allows for forward iteration over the
        elements in the ParticleCollection (e.g. `for p in pset:`).
        """
        return ParticleCollectionIteratorSOA(self)

    def __reversed__(self):
        """Returns an Iterator that allows for backwards iteration over
        the elements in the ParticleCollection (e.g.
        `for p in reversed(pset):`).
        """
        return ParticleCollectionIteratorSOA(self, True)


class ParticleAccessorSOA(BaseParticleAccessor):
    def __init__(self, pcoll):
        super().__init__(pcoll)

    def __getattr__(self, name):
        return self.pcoll.particle_data[name][self._index]

    def __setattr__(self, name, value):
        if name in ['pcoll', '_index']:
            object.__setattr__(self, name, value)
        else:
            # avoid recursion
            self.pcoll.particle_data[name][self._index] = value

    def __repr__(self):
        time_string = 'not_yet_set' if self.time is None or np.isnan(self.time) else "{:f}".format(self.time)
        str = "P[%d](lon=%f, lat=%f, depth=%f, " % (self.id, self.lon, self.lat, self.depth)
        for var in self.pcoll.ptype.variables:
            if var.to_write is not False and var.name not in ['id', 'lon', 'lat', 'depth', 'time']:
                str += "%s=%f, " % (var.name, getattr(self, var.name))
        return str + "time=%s)" % time_string


class ParticleCollectionIteratorSOA(BaseParticleCollectionIterator):
    def __init__(self, pcoll, reverse=False, subset=None):
        # super().__init__(pcoll)  # Do not actually need this

        if subset is not None:
            if type(subset[0]) not in [int, np.int32]:
                raise TypeError("Iteration over a subset of particles in the"
                                " particleset requires a list or numpy array"
                                " of indices (of type int or np.int32).")
            if reverse:
                self._indices = subset.reverse()
            else:
                self._indices = subset
            self.max_len = len(subset)
        else:
            self.max_len = len(pcoll)
            if reverse:
                self._indices = range(self.max_len - 1, -1, -1)
            else:
                self._indices = range(self.max_len)

        self._reverse = reverse
        self._index = 0
        self.p = pcoll.data_accessor()
        self._head = pcoll.data_accessor()
        self._head.set_index(0)
        self._tail = pcoll.data_accessor()
        self._tail.set_index(self.max_len - 1)

    def __next__(self):
        if self._index < self.max_len:
            self.p.set_index(self._indices[self._index])
            result = self.p
            self._index += 1
            return result

        # End of Iteration
        raise StopIteration

    @property
    def current(self):
        return self.p

    def __repr__(self):
        dir_str = 'Backward' if self._reverse else 'Forward'
        str = f"{dir_str} iteration at index {self._index} of {self.max_len}."
        return str
