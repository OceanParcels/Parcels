from datetime import timedelta as delta
from operator import attrgetter  # NOQA

from ctypes import c_void_p

import numpy as np

from parcels.collection.collections import ParticleCollection
from parcels.collection.iterators import BaseParticleAccessor
from parcels.collection.iterators import BaseParticleCollectionIterator
from parcels.collection.iterators import BaseParticleCollectionIterable
from parcels.particle import ScipyParticle, JITParticle  # noqa
from parcels.field import Field
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import OperationCode
from scipy.spatial import distance

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

__all__ = ['ParticleCollectionAOS', 'ParticleCollectionIterableAOS', 'ParticleCollectionIteratorAOS']


def _to_write_particles(pd, time):
    """We don't want to write a particle that is not started yet.
    Particle will be written if particle.time is between time-dt/2 and time+dt (/2)
    """
    return [i for i, p in enumerate(pd) if (((time - np.abs(p.dt/2) <= p.time < time + np.abs(p.dt))
                                             or (np.isnan(p.dt) and np.equal(time, p.time)))
                                            and np.isfinite(p.id))]


def _is_particle_started_yet(particle, time):
    """We don't want to write a particle that is not started yet.
    Particle will be written if:
      * particle.time is equal to time argument of pfile.write()
      * particle.time is before time (in case particle was deleted between previous export and current one)
    """
    return (particle.dt*particle.time <= particle.dt*time or np.isclose(particle.time, time))


def _convert_to_flat_array(var):
    """Convert lists and single integers/floats to one-dimensional numpy arrays

    :param var: list or numeric to convert to a one-dimensional numpy array
    """
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.int32)):
        return np.array([var])
    else:
        return np.array(var)


class ParticleCollectionAOS(ParticleCollection):
    _data_c = None

    def __init__(self, pclass, lon, lat, depth, time, lonlatdepth_dtype, pid_orig, partitions=None, ngrid=1, **kwargs):
        """
        :arg idgen: an instance of an ID generator used to obtain unique IDs - mandatory for a node-based collection
        :arg c_lib_register: an instance of a process-consistent LibraryRegisterC object - mandatory for a node-based collection
        :arg pclass: the Particle class of the objects stored within the nodes
        :arg lon: a non-None list or array of longitudes
        :arg lat: a non-None list or array of latitudes
        :arg depth: a non-None list or array of depths
        :arg times: a non-None list- or array of time-values
        :arg lonlatdepth_dtype: the datatype (dtype) of coordinate-values (apart from time - time is fixed to 64-bit float)
        :arg pid_orig: None or a vector or list of 64-bit (signed or unsigned) integer IDs, used for repeating particle addition
        :arg paritions: None, or a list of indicators to which the particles shall be attached to
        :arg ngrid: number of grids in the fieldset of the overarching ParticleSet - required for initialising the
        field references of the ctypes-link of particles that are allocated
        """

        super(ParticleCollection, self).__init__()
        self._ngrid = ngrid

        assert pid_orig is not None, "particle IDs are None - incompatible with the collection. Invalid state."
        pid = pid_orig + pclass.lastID

        assert depth is not None, "particle's initial depth is None - incompatible with the collection. Invalid state."
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth do not all have the same lenghts')

        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) do not have the same lengths.')

        if partitions is not None and partitions is not False:
            self._pu_indicators = _convert_to_flat_array(partitions)

        for kwvar in kwargs:
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) do nott have the same lengths.' % kwvar)

        offset = np.max(pid) if (pid is not None) and len(pid) > 0 else -1
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

        pclass.setLastID(offset+1)

        if lonlatdepth_dtype is None:
            self._lonlatdepth_dtype = np.float32
        else:
            self._lonlatdepth_dtype = lonlatdepth_dtype
        assert self._lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'
        pclass.set_lonlatdepth_dtype(self._lonlatdepth_dtype)
        self._pclass = pclass

        self._ptype = self._pclass.getPType()
        self._kwarg_keys = list(kwargs.keys())
        self._data = np.empty(lon.shape[0], dtype=pclass)
        initialised = set()

        if self._ptype.uses_jit:
            # Allocate underlying data for C-allocated particles
            self._data_c = np.empty(lon.shape[0], dtype=self._ptype.dtype)

        self._ncount = self._data.shape[0]

        if lon is not None and lat is not None:
            # Initialise from lists of lon/lat coordinates
            assert self.ncount == len(lon) and self.ncount == len(lat), (
                'Size of ParticleSet does not match length of lon and lat.')

            for i in range(self.ncount):
                self._data[i] = pclass(lon[i], lat[i], pid[i], ngrids=ngrid, depth=depth[i], cptr=self.cptr(i), time=time[i])
                # Set other Variables if provided
                for kwvar in kwargs:
                    if isinstance(kwvar, Field):
                        continue
                    if not hasattr(self._data[i], kwvar):
                        raise RuntimeError('Particle class does not have Variable %s' % kwvar)
                    setattr(self._data[i], kwvar, kwargs[kwvar][i])
                    if kwvar not in initialised:
                        initialised.add(kwvar)

            initialised |= {'lat', 'lon', 'depth', 'time', 'id'}

            for v in self._ptype.variables:
                if v.name in initialised:
                    continue

                if isinstance(v.initial, Field):
                    for i in range(self.ncount):
                        init_time = time[i] if time is not None and len(time) > 0 and np.count_nonzero([tval is not None for tval in time]) == len(time) else 0
                        init_field = v.initial
                        init_field.fieldset.computeTimeChunk(init_time, 0)
                        if (time[i] is None) or (np.isnan(time[i])):
                            raise RuntimeError('Cannot initialise a Variable with a Field if no time provided (time-type: {} values: {}). Add a "time=" to ParticleSet construction'.format(type(time), time))
                        setattr(self._data[i], v.name, init_field[init_time, depth[i], lat[i], lon[i]])
                        logger.warning_once("Particle initialisation from field can be very slow as it is computed in scipy mode.")

                if v not in initialised:
                    initialised.add(v)
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")
        self._iterator = None
        self._riterator = None

    def __del__(self):
        """
        Collection - Destructor
        """
        super().__del__()

    def iterator(self):
        """
        :returns ParticleCollectionIterator, used for a 'for'-loop, in a forward-manner
        """
        self._iterator = ParticleCollectionIteratorAOS(self)
        return self._iterator

    def __iter__(self):
        """Returns an Iterator that allows for forward iteration over the
        elements in the ParticleCollection (e.g. `for p in pset:`).
        """
        return self.iterator()

    def reverse_iterator(self):
        """
        :returns ParticleCollectionIterator, used for a 'for'-loop, in a backward-manner
        """
        self._riterator = ParticleCollectionIteratorAOS(self, True)
        return self._riterator

    def __reversed__(self):
        """Returns an Iterator that allows for backwards iteration over
        the elements in the ParticleCollection (e.g.
        `for p in reversed(pset):`).
        """
        return self.reverse_iterator()

    def __getitem__(self, index):
        """
        Access a particle in this collection using the fastest access
        method for this collection - by its index.

        :arg index: int or np.int32 index of a particle in this collection
        """
        return self.get_single_by_index(index)

    def __getattr__(self, name):
        """
        Access a single property of all particles.
        CAUTION: this function is not(!) in-place and is quite slow

        :arg name: name of the property
        """
        pdtype = None
        for var in self._ptype.variables:
            if name == var.name:
                pdtype = var.dtype
        if pdtype is None:
            return None
        result = np.zeros(self._ncount, dtype=pdtype)
        for index in range(self._ncount):
            if hasattr(self._data[index], name):
                result[index] = getattr(self._data[index], name)
        return result

    @property
    def data_c(self):
        return self._data_c

    @property
    def particle_data(self):
        """
        'particle_data' is a reference to the actual barebone-storage of the particle data, and thus depends directly on the
        specific collection in question. This property is just available for convenience and backward-compatibility, and
        this returns the same as 'data'.

        Transitional/compatibility function.
        """
        return self._data_c

    # @property
    # def kernel_class(self):
    #     return self._kclass

    # @kernel_class.setter
    # def kernel_class(self, value):
    #     self._kclass = value

    def cptr(self, index):
        if self._data_c is not None:
            return self._data_c[index]
        return None

    def empty(self):
        """
        :returns if the collections is empty or not
        """
        return len(self._data) <= 0

    def get_index_by_ID(self, id):
        """
        Provides a simple function to search / get the index for a particle of the requested ID.
        Returns the particle's index.

        :arg id: search Particle-ID
        :returns index of the Particle with the ID to be looked up
        """
        super().get_single_by_ID(id)
        data_ids = np.array([p.id for p in self._data])
        index = np.where(data_ids == id)[0][0]
        return index

    def get_indices_by_IDs(self, ids):
        """
        Looking up the indices of the nodes with the IDs that are provided. This expects the nodes to be present. IDs
        for which no nodes are present in the collected are disregarded.

        :arg ids: a vector-list or numpy.ndarray of (64-bit signed or unsigned integer) IDs
        :returns: a vector-list of indices per requested ID
        """
        data_ids = np.array([p.id for p in self._data])
        AinB = np.in1d(ids, data_ids)
        if False in AinB:
            logger.warning("Did not locate all requested IDs.")
        indices = np.nonzero(AinB)[0]
        return indices

    def get_single_by_index(self, index):
        """
        This function gets a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure.
        In cases where a get-by-index would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-ID.

        :arg index: index of the object to be retrieved
        :returns Particle object (SciPy- or JIT) at the indexed location
        """
        super().get_single_by_index(index)

        return self._data[index]

    def get_single_by_object(self, particle_obj):
        """
        This function gets a (particle) object from the collection based on its actual object. For collections that
        are random-access and based on indices (e.g. unordered list, vectors, arrays and dense matrices), this function
        would involve a parsing of the whole list and translation of the object into an index in the collection - which
        results in a significant performance malus.
        In cases where a get-by-object would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-index or get-by-ID.

        In this specific implementation, we cannot look for the object
        directly, so we will look for one of its properties (the ID) that
        has the nice property of being stored in an ordered list (if the
        collection is sorted).

        :arg particle_obj: a template object of a Particle (SciPy- or JIT) with reference values to be searched for
        :returns (first) Particle-object (SciPy- or JIT) of the requested particle data
        """
        super().get_single_by_object(particle_obj)

        return self.get_single_by_ID(particle_obj.id)

    def get_single_by_ID(self, id):
        """
        This function gets a (particle) object from the collection based on the object's ID. For some collections,
        this operation may involve a parsing of the whole list and translation of the object's ID into an index  or an
        object reference in the collection - which results in a significant performance malus.
        In cases where a get-by-ID would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-index.

        This function uses binary search if we know the ID list to be sorted, and linear search otherwise. We assume
        IDs are unique.

        :arg id: search Particle-ID
        :return (first) Particle-object (SciPy- or JIT) attached to ID
        """
        super().get_single_by_ID(id)

        ids = np.array([p.id for p in self._data])
        index = np.where(ids == id)[0][0]
        return self.get_single_by_index(index)

    def get_same(self, same_class):
        """
        This function gets particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection.

        :arg same_class: a ParticleCollectionAOS object with a subsample of Particles in this collection
        :returns list of Particle-objects (SciPy- or JIT) of the requested subset-collection
        """
        super().get_same(same_class)
        results = []
        for item in same_class:
            index = np.where(self._data == item)[0]  # this will require implementing an equals(...) function to check between Particle and BaseParticleAccessor
            if index.size != 0:
                index = index[0]
                results.append(self._data[index])
        if len(results) == 0:
            results = None
        return results

    def get_collection(self, pcollection):
        """
        This function gets particles from this collection that are themselves stored in a ParticleCollection, which
        is differently structured than this one. That means the other-collection has to be re-formatted first in an
        intermediary format.

        :arg pcollection: a ParticleCollection object (i.e. derived from BaseParticleCollection) with a subsample of Particles in this collection
        :returns list of Particle-objects (SciPy- or JIT) of the requested subset-collection
        """
        super().get_collection(pcollection)
        if self._ncount <= 0:
            return None
        ngrids = len(getattr(self._data[0], 'xi'))
        results = []
        for item in pcollection:
            pdata_item = self._pclass(lon=item.lon, lat=item.lat, pid=item.pid, ngrids=ngrids, depth=item.depth, time=item.time)
            index = np.where(self._data == pdata_item)[0]  # this will require implementing and equals(...) function check between Particle and BaseParticleAccessor
            if index.size != 0:
                index = index[0]
                results.append(self._data[index])
        if len(results) == 0:
            results = None
        return results

    def get_multi_by_PyCollection_Particles(self, pycollectionp):
        """
        This function gets particles from this collection, which are themselves in common Python collections, such as
        lists, dicts and numpy structures. We can either directly get the referred Particle instances (for internally-
        ordered collections, e.g. ordered lists, sets, trees) or we may need to parse each instance for its index (for
        random-access structures), which results in a considerable performance malus.

        For collections where get-by-object incurs a performance malus, it is advisable to multi-get particles
        by indices or IDs.

        :arg pycollectionp: a Python-internal collection object (e.g. a tuple or list), filled with reference particles (SciPy- or JIT)
        :returns a vector-list of the requested particles
        """
        super().get_multi_by_PyCollection_Particles(pycollectionp)
        np_collection_p = np.array(pycollectionp, dtype=self._pclass)
        indices = np.in1d(np_collection_p, self._data).nonzero()[0]
        result = self._data[indices]
        if result.shape[0] <= 0:
            result = None
        return result

    def get_multi_by_indices(self, indices):
        """
        This function gets particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a get-via-object-reference strategy.

        :param indices: requested indices
        :returns vector-list of Particle objects
        """
        super().get_multi_by_indices(indices)
        if type(indices) is dict:
            indices = list(indices.values())
        return self._data[indices]

    def get_multi_by_IDs(self, ids):
        """
        This function gets particles from this collection based on their IDs. For collections where this removal
        strategy would require a collection transformation or by-ID parsing, it is advisable to rather apply a get-
        by-objects or get-by-indices scheme.

        Note that this implementation assumes that IDs of particles are strictly increasing with increasing index. So
        a particle with a larger index will always have a larger ID as well. The assumption often holds for this
        data structure as new particles always get a larger ID than any existing particle (IDs are not recycled)
        and their data are appended at the end of the list (largest index). This allows for the use of binary search
        in the look-up. The collection maintains a `sorted` flag to indicate whether this assumption holds.

        :arg ids: requested IDs of particles
        :returns vector-list of Particle objects
        """
        super().get_multi_by_IDs(ids)
        if type(ids) is dict:
            ids = list(ids.values())

        if len(ids) == 0:
            return None

        data_ids = np.array([p.id for p in self._data])
        indices = np.in1d(ids, data_ids)
        items_found = indices
        indices = np.nonzero(indices)[0]
        if False in items_found:
            logger.warn("Failed to located a requested Particle")
        if len(indices) <= 0:
            indices = None
        return self.get_multi_by_indices(indices)

    def merge_collection(self, pcollection):
        """
        Merges another, differently structured ParticleCollection into this collection. This is done by, for example,
        appending/adding the items of the other collection to this collection.

        this is the former "add(pcollection)" function.
        :arg pcollection: second ParticleCollection object to be merged into this collection
        :returns vector-list of indices of all merged particles
        """
        # ==== first approach - still need to incorporate the MPI re-centering ==== #
        super().merge_collection(pcollection)
        ngrids = 0
        if self._ncount > 0:
            ngrids = self._ngrid
        elif len(pcollection) > 0:
            pd0 = pcollection[0]
            ngrids = len(pd0['xi'])
        pd_cdata = None
        if self._ptype.uses_jit:
            pd_cdata = np.array(len(pcollection), dtype=self._ptype.dtype)
        results = []
        for item_index, item in enumerate(pcollection):
            pdata_item = self._pclass(lon=item.lon, lat=item.lat, pid=item.pid, ngrids=ngrids, depth=item.depth, time=item.time, cptr=pd_cdata[item_index])
            results.append(pdata_item)
        self._data = np.concatenate([self._data, np.array(results, dtype=self._pclass)])
        if self._ptype.uses_jit:
            self._data_c = np.concatenate([self._data_c, pd_cdata])
            for p, pdata in zip(self._data, self._data_c):
                p._cptr = pdata
        self._ncount = self._data.shape[0]
        return results

    def merge_same(self, same_class):
        """
        Merges another, equi-structured ParticleCollection into this collection. This is done by concatenating
        both collections. The fact that they are of the same ParticleCollection's derivative simplifies
        parsing and concatenation.

        this is the former "add(same_class)" function.
        :arg same_class: second ParticleCollectionAOS object to be merged into this collection
        :returns vector-list of indices of all merged particles
        """
        super().merge_same(same_class)
        if same_class.ncount <= 0:
            return

        if self._ncount <= 0:
            self._data = same_class._data
            if same_class.ptype.uses_jit and self._ptype.uses_jit:
                self._data_c = same_class._data_c
            self._ncount = same_class.ncount
            return

        self._data = np.concatenate([self._data, same_class.data])
        if self._ptype.uses_jit:
            self._data_c = np.concatenate([self._data_c, same_class.data_c])
            for p, pdata in zip(self._data, self._data_c):
                p._cptr = pdata
        self._ncount = self._data.shape[0]

    def add_multiple(self, data_array):
        """
        Add multiple particles from an array-like structure (i.e. list or tuple or np.ndarray)
        to the collection.
        :arg data_array: one of the following:
            i) a list or tuples containing multple Particle instances
            ii) a Numpy.ndarray of dtype = Particle dtype
            iii) a dict of Numpy.ndarray of shape, each of which with N = # particles
        :returns vector-list of indices of all added particles
        """
        super().add_multiple(data_array)
        results = []
        if len(data_array) <= 0:
            return results
        if isinstance(data_array, list) or isinstance(data_array, tuple):
            logger.info("SoA -> data_array - type: {}; values: {}".format(type(data_array[0]), data_array))
            for item in data_array:
                results.append(self.add_single(item))
        elif isinstance(data_array, np.ndarray) and (data_array.dtype == self._ptype):
            for i in range(data_array.shape[0]):
                pdata = data_array[i]
                results.append(self.add_single(pdata))
        elif isinstance(data_array, dict) and isinstance(data_array['lon'], np.ndarray):
            pu_ids = None
            pu_indices = None
            n_pu_data = 0
            if MPI and MPI.COMM_WORLD.Get_size() > 1:
                mpi_comm = MPI.COMM_WORLD
                mpi_size = mpi_comm.Get_size()
                mpi_rank = mpi_comm.Get_rank()
                spdata = np.array([data_array['lon'], data_array['lat']]).transpose([1, 0])
                min_pu = None
                if mpi_rank == 0:
                    dists = distance.cdist(spdata, self._pu_centers)
                    min_pu = np.argmax(dists, axis=1)
                    self._pu_indicators = np.concatenate((self._pu_indicators, min_pu), axis=0)
                min_pu = mpi_comm.bcast(min_pu, root=0)
                self._pu_indicators = mpi_comm.bcast(self._pu_indicators, root=0)
                pu_indices = np.nonzero(min_pu == mpi_rank)[0]
                ids = np.arange(ScipyParticle.lastID, stop=ScipyParticle.lastID+data_array['lon'].shape[0]) if 'id' not in data_array.keys() else data_array['id']
                mpi_comm.Bcast(ids, root=0)
                pu_ids = ids
                new_lastID = 0
                if mpi_rank == 0:
                    new_lastID = ScipyParticle.lastID+data_array['lon'].shape[0]-1
                new_lastID = mpi_comm.bcast(new_lastID, root=0)
                self._pclass.setLastID(new_lastID)
                pu_center = np.array(np.mean(spdata, axis=0), dtype=self._lonlatdepth_dtype)
                n_pu_data = pu_indices.shape[0]
                pu_ncenters = None
                if mpi_rank == 0:
                    pu_ncenters = np.empty([mpi_size, pu_center.shape[0]], dtype=self._latlondepth_dtype)
                mpi_comm.Gather(pu_center, pu_ncenters, root=0)
                pu_ndata = mpi_comm.gather(n_pu_data, root=0)
                if mpi_rank == 0:
                    for i in range(self._pu_centers.shape[0]):
                        ax = float(pu_ndata[i]) / float(len(np.nonzero(self._pu_indicators == i)[0]))
                        self._pu_centers[i, :] += ax*pu_ncenters[i, :]
                mpi_comm.Bcast(self._pu_centers, root=0)
            else:
                pu_ids = np.arange(ScipyParticle.lastID, stop=ScipyParticle.lastID+data_array['lon'].shape[0]) if 'id' not in data_array.keys() else data_array['id']
                new_lastID = ScipyParticle.lastID+data_array['lon'].shape[0]-1
                self._pclass.setLastID(new_lastID)
                pu_indices = np.arange(start=0, stop=data_array['lon'].shape[0])
                n_pu_data = pu_indices.shape[0]
            if n_pu_data <= 0:
                results = []
            else:
                for i in pu_indices:
                    pid = pu_ids[i]
                    pdata = self._pclass(lon=data_array['lon'][i], lat=data_array['lat'][i], pid=pid, ngrids=self._ngrid)
                    if 'depth' in data_array.keys():
                        pdata.depth = data_array['depth'][i]
                    if 'time' in data_array.keys():
                        pdata.time = data_array['time'][i]
                    if 'dt' in data_array.keys():
                        pdata.dt = data_array['dt'][i]
                    for key in self._kwarg_keys:
                        if key in data_array.keys():
                            setattr(pdata, key, data_array[key][i])
                    results.append(self.add_single(pdata, pu_checked=True))
        self._ncount = len(self._data)
        return results

    def add_single(self, particle_obj, pu_checked=False):
        """
        Adding a single Particle to the collection - either as a 'Particle; object in parcels itself, or
        via its ParticleAccessor.
        :returns index of added particle
        """
        # ==== first approach - still need to incorporate the MPI re-centering ==== #
        super().add_single(particle_obj)
        assert isinstance(particle_obj, ScipyParticle)
        # Comment: by current workflow, pset modification is only done on the front node, thus
        # the distance determination and assigment is also done on the front node
        _add_to_pu = True
        if MPI and MPI.COMM_WORLD.Get_size() > 1 and not pu_checked:
            if self._pu_centers is not None and isinstance(self._pu_centers, np.ndarray):
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                mpi_size = mpi_comm.Get_size()
                min_dist = np.finfo(self._lonlatdepth_dtype).max
                min_pu = 0
                spdata = None
                if mpi_size > 1 and mpi_rank == 0:
                    ppos = particle_obj
                    spdata = np.array([ppos.lat, ppos.lon], dtype=self._lonlatdepth_dtype)
                    n_clusters = self._pu_centers.shape[0]
                    for i in range(n_clusters):
                        diff = self._pu_centers[i, :] - spdata
                        dist = np.dot(diff, diff)
                        if dist < min_dist:
                            min_dist = dist
                            min_pu = i
                    self._pu_indicators = np.concatenate((self._pu_indicators, min_pu), axis=0)
                min_pu = mpi_comm.bcast(min_pu, root=0)
                self._pu_indicators = mpi_comm.bcast(self._pu_indicators, root=0)
                pu_id = particle_obj.id
                if particle_obj.id >= np.iinfo(np.uint64).max:
                    pu_id = ScipyParticle.lastID
                    new_lastID = 0
                    if mpi_rank == 0:
                        new_lastID = ScipyParticle.lastID + 1
                    new_lastID = mpi_comm.bcast(new_lastID, root=0)
                    self._pclass.setLastID(new_lastID)
                if mpi_rank == 0:
                    ax = 1.0 / float(len(np.nonzero(self._pu_indicators == min_pu)[0]))
                    self._pu_centers[min_pu, :] += ax * spdata
                mpi_comm.Bcast(self._pu_centers, root=0)

                if mpi_rank == min_pu:
                    _add_to_pu = True
                else:
                    _add_to_pu = False
            else:
                pu_id = particle_obj.id
                if particle_obj.id >= np.iinfo(np.uint64).max:
                    pu_id = ScipyParticle.lastID
                    self._pclass.setLastID(ScipyParticle.lastID + 1)
        else:
            pu_id = particle_obj.id
            if particle_obj.id >= np.iinfo(np.uint64).max:
                pu_id = ScipyParticle.lastID
                self._pclass.setLastID(ScipyParticle.lastID + 1)
        if _add_to_pu:
            particle_obj.id = pu_id
            self._data = np.concatenate((self._data, [particle_obj, ]), axis=0)
            index = self._data.shape[0]-1
            if self._ptype.uses_jit and isinstance(particle_obj, JITParticle):
                self._data_c = np.concatenate((self._data_c, [particle_obj.get_cptr(), ]))
            if index >= 0:
                self._ncount = self._data.shape[0]
                return index
        self._ncount = len(self._data)
        return None

    def split_by_index(self, indices):
        """
        This function splits this collection into two disect equi-structured collections using the indices as subset.
        The reason for it can, for example, be that the set exceeds a pre-defined maximum number of elements, which for
        performance reasons mandates a split.

        The function shall return the newly created or extended Particle collection, i.e. either the collection that
        results from a collection split or this very collection, containing the newly-split particles.

        :arg indices: requested indices to be split off this collection
        :returns new ParticleCollectionAOS with the split-off particles
        """
        super().split_by_index(indices)
        assert len(self._data) > 0
        result = ParticleCollectionAOS(self._idgen, self._c_lib_register, self._pclass, lon=np.empty(shape=0), lat=np.empty(shape=0), depth=np.empty(shape=0), time=np.empty(shape=0), pid_orig=None, lonlatdepth_dtype=self._lonlatdepth_dtype, ngrid=self._ngrid)
        idx = sorted(indices, reverse=True)
        tmp = []
        for index in idx:  # pop-based process needs to start from the back of the queue
            pdata = self.pop_single_by_index(index=index)
            tmp.append(pdata)
        for pdata in reversed(tmp):  # add particles in correct order again
            result.add_single(pdata)
        return result

    def split_by_id(self, ids):
        """
        This function splits this collection into two disect equi-structured collections using the indices as subset.
        The reason for it can, for example, be that the set exceeds a pre-defined maximum number of elements, which for
        performance reasons mandates a split.

        The function shall return the newly created or extended Particle collection, i.e. either the collection that
        results from a collection split or this very collection, containing the newly-split particles.

        :arg IDs: requested IDs to be split off this collection
        :returns new ParticleCollectionAOS with the split-off particles
        """
        super().split_by_id(ids)
        assert len(self._data) > 0
        result = ParticleCollectionAOS(self._idgen, self._c_lib_register, self._pclass, lon=np.empty(shape=0), lat=np.empty(shape=0), depth=np.empty(shape=0), time=np.empty(shape=0), pid_orig=None, lonlatdepth_dtype=self._lonlatdepth_dtype, ngrid=self._ngrid)
        for id in ids:
            pdata = self.pop_single_by_ID(id)
            result.add_single(pdata)
        return result

    def __iadd__(self, same_class):
        """
        Performs an incremental addition of the equi-structured ParticleCollections, such to allow

        a += b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place addition of (an) element(s).

        :arg same_class: second ParticleCollectionAOS object to be merged into this collection
        :returns vector-list of indices of all merged particles
        """
        self.merge_same(same_class)
        return self

    def insert(self, obj, index=None):
        """
        This function allows to 'insert' a Particle (as object or via its accessor) into this collection. This method
        needs to be specified to each collection individually. Some collections (e.g. unordered list) allow to define
        the index where the object is to be inserted. Some collections can optionally insert an object at a specific
        position - at a significant speed- and memory malus cost (e.g. vectors, arrays, dense matrices). Some
        collections that manage a specified indexing order internally (e.g. ordered lists, sets, trees), and thus
        have no use for an 'index' parameter. For those collections with an internally-enforced order, the function
        mapping equates to:

        insert(obj) -> add_single(obj)

        For AoS, insert with 'index==None', the function equates to 'add'. If 'index' is specified, split the array,
        insert the item and splice the arrays.
        :arg obj: Particle object to insert
        """
        if index is None:
            self.add_single(obj)
        else:
            assert isinstance(obj, ScipyParticle)
            top_array = self._data[0:index-1]
            bottom_array = self._data[index:]
            splice_array = np.concatenate([top_array, obj])
            self._data = np.concatenate([splice_array, bottom_array])
            if self._ptype.uses_jit and isinstance(obj, JITParticle):
                top_array = self._data_c[0:index-1]
                bottom_array = self._data_c[index:]
                splice_array = np.concatenate((top_array, [obj.get_cptr(), ]))
                self._data_c = np.concatenate([splice_array, bottom_array])
            self._ncount = self._data.shape[0]

    def push(self, particle_obj):
        """
        This function pushes a Particle (as object or via its accessor) to the end of a collection ('end' definition
        depends on the specific collection itself). For collections with an inherent indexing order (e.g. ordered lists,
        sets, trees), the function just includes the object at its pre-defined position (i.e. not necessarily at the
        end). For the collections, the function mapping equates to:

        int32 push(particle_obj) -> add_single(particle_obj); return -1;

        This function further returns the index, at which position the Particle has been inserted. By definition,
        the index is positive, thus: a return of '-1' indicates push failure, NOT the last position in the collection.
        Furthermore, collections that do not work on an index-preserving manner also return '-1'.

        :arg particle_obj: Particle object to push
        :returns index, i.e. position of the new element
        """
        return_index = self._ncount
        self.add_single(particle_obj)
        return return_index

    def append(self, particle_obj):
        """
        This function appends a Particle (as object or via its accessor) to the end of a collection ('end' definition
        depends on the specific collection itself). For collections with an inherent indexing order (e.g. ordered lists,
        sets, trees), the function just includes the object at its pre-defined position (i.e. not necessarily at the
        end). For the collections, the function mapping equates to:

        append(particle_obj) -> add_single(particle_obj)

        The function - in contrast to 'push' - does not return the index of the inserted object.
        :arg particle_obj: Particle object to append
        """
        self.add_single(particle_obj)

    def __delitem__(self, key):
        """
        This is the high-performance method to delete a specific object from this collection.
        As the most-performant way depends on the specific collection in question, the function is abstract.

        Highlight for the specific implementation:
        The 'key' parameter should still be evaluated for being a single or a multi-entry delete, and needs to check
        that it received the correct type of 'indexing' argument (i.e. index, id or iterator).

        This should actually delete the item instead of just marking the particle as 'to be deleted'.
        :arg key: index to be removed
        """
        self.remove_single_by_index(key)

    def delete_by_index(self, index):
        """
        This method deletes a particle from the  the collection based on its index. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        :arg index: index of the Particle to be set to the deleted state
        """
        super().delete_by_index(index)
        p = self.get_single_by_index(index)
        p.state = OperationCode.Delete

    def delete_by_ID(self, id):
        """
        This method deletes a particle from the  the collection based on its ID. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        :arg id: ID of the Particle to be set to the deleted state
        """
        super().delete_by_ID(id)
        index = self.get_index_by_ID(id)
        self.delete_by_index(index)

    def remove_single_by_index(self, index):
        """
        This function removes a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure,
        and then perform the removal.
        In cases where a removal-by-index would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-object or remove-by-ID.
        :arg index: index of the Particle to be removed from the collection
        """
        super().remove_single_by_index(index)
        result = self.get_single_by_index(index)
        self._data = np.delete(self._data, index)
        if self.ptype.uses_jit:
            self._data_c = np.delete(self._data_c, index)
            # Update C-pointer on particles
            for p, pdata in zip(self._data, self._data_c):
                p._cptr = pdata
        self._ncount -= 1
        return result

    def remove_single_by_object(self, particle_obj):
        """
        This function removes a (particle) object from the collection based on its actual object. For collections that
        are random-access and based on indices (e.g. unordered list, vectors, arrays and dense matrices), this function
        would involves a parsing of the whole list and translation of the object into an index in the collection to
        perform the removal - which results in a significant performance malus.
        In cases where a removal-by-object would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-index or remove-by-ID.
        :arg particle_obj: Particle object that is to be removed from the collection
        """
        super().remove_single_by_object(particle_obj)

        # We cannot look for the object directly, so we will look for one of
        # its properties that has the nice property of being stored in an
        # ordered list
        return self.remove_single_by_ID(particle_obj.id)

    def remove_single_by_ID(self, id):
        """
        This function removes a (particle) object from the collection based on the object's ID. For some collections,
        this operation may involve a parsing of the whole list and translation of the object's ID into an index  or an
        object reference in the collection in order to perform the removal - which results in a significant performance
        malus.
        In cases where a removal-by-ID would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-object or remove-by-index.
        :arg id: Particle ID of the object to be removed from the collection
        """
        super().remove_single_by_ID(id)
        index = self.get_index_by_ID(id)
        return self.remove_single_by_index(index)

    def remove_same(self, same_class):
        """
        This function removes particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection. As the structures of both collections are the same, a more efficient M-in-N
        removal can be applied without an in-between reformatting.

        :arg same_class: a ParticleCollectionAOS object, containing Nodes that are to be removed from this collection
        """
        super().remove_same(same_class)
        indices = []
        indices = np.in1d(same_class.data, self._data)
        indices = None if len(indices) == 0 else np.nonzero(indices)[0]

        self._data = np.delete(self._data, indices)
        if self.ptype.uses_jit:
            self._data_c = np.delete(self._data_c, indices)
            for p, pdata in zip(self._data, self._data_c):
                p._cptr = pdata
        self._ncount = self._data.shape[0]

    def remove_collection(self, pcollection):
        """
        This function removes particles from this collection that are themselves stored in a ParticleCollection, which
        is differently structured than this one. That means the removal first requires the removal-collection to be re-
        formatted in an intermediary format, before executing the removal.
        That said, this method should still be at least as efficient as a removal via common Python collections (i.e.
        lists, dicts, numpy's nD arrays & dense arrays). Despite this, due to the reformatting, in some cases it may
        be more efficient to remove items then rather by IDs oder indices.

        :arg pcollection: a BaseParticleCollection object, containing Particle objects that are to be removed from this collection
        """
        super().remove_collection(pcollection)

        ids = [p.id for p in pcollection]
        data_ids = [p.id for p in self._data]
        indices = np.in1d(ids, data_ids)
        indices = None if len(indices) == 0 else np.nonzero(indices)[0]

        self._data = np.delete(self._data, indices)
        if self.ptype.uses_jit:
            self._data_c = np.delete(self._data_c, indices)
            for p, pdata in zip(self._data, self._data_c):
                p._cptr = pdata
        self._ncount = self._data.shape[0]

    def remove_multi_by_PyCollection_Particles(self, pycollectionp):
        """
        This function removes particles from this collection, which are themselves in common Python collections, such as
        lists, dicts and numpy structures. In order to perform the removal, we can either directly remove the referred
        Particle instances (for internally-ordered collections, e.g. ordered lists, sets, trees) or we may need to parse
        each instance for its index (for random-access structures), which results in a considerable performance malus.

        For collections where removal-by-object incurs a performance malus, it is advisable to multi-remove particles
        by indices.

        :arg pycollectionp: a Python-based collection (i.e. a tuple or list), containing Particle objects that are to
                            be removed from this collection.
        """
        super().remove_multi_by_PyCollection_Particles(pycollectionp)
        npcollectionp = np.array(pycollectionp, dtype=self._pclass)
        npindices = np.in1d(npcollectionp, self._data)
        indices = None if len(npindices) == 0 else np.nonzero(npindices)[0]
        if indices is None:
            return

        self._data = np.delete(self._data, indices)
        if self.ptype.uses_jit:
            self._data_c = np.delete(self._data_c, indices)
            for p, pdata in zip(self._data, self._data_c):
                p._cptr = pdata
        self._ncount = self._data.shape[0]

    def remove_multi_by_indices(self, indices):
        """
        This function removes particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a removal-via-object-reference strategy.

        :arg indices: a list or np.ndarray of indices that are to be removed from this collection.
        """
        super().remove_multi_by_indices(indices)
        if type(indices) is dict:
            indices = list(indices.values())

        self._data = np.delete(self._data, indices)
        if self.ptype.uses_jit:
            self._data_c = np.delete(self._data_c, indices)
            for p, pdata in zip(self._data, self._data_c):
                p._cptr = pdata
        self._ncount = self._data.shape[0]

    def remove_multi_by_IDs(self, ids):
        """
        This function removes particles from this collection based on their IDs. For collections where this removal
        strategy would require a collection transformation or by-ID parsing, it is advisable to rather apply a removal-
        by-objects or removal-by-indices scheme.

        :arg ids: a list or numpy.ndarray of (signed- or unsigned) 64-bit integer IDs, the items of which are to be
                  removed from this collection
        """
        super().remove_multi_by_IDs(ids)
        if type(ids) is dict:
            ids = list(ids.values())
        if len(ids) == 0:
            return

        # Use binary search if the collection is sorted, linear search otherwise
        data_ids = [p.id for p in self._data]
        indices = np.in1d(ids, data_ids)
        indices = None if len(indices) == 0 else np.nonzero(indices)[0]
        self.remove_multi_by_indices(indices)

    def __isub__(self, other):
        """
        This method performs an incremental removal of the equi-structured ParticleCollections, such to allow

        a -= b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place removal of (an) element(s).

        :arg other: a single Particle or a collection of objects or keys that are to be removed.
        """
        if other is None:
            return
        if type(other) is type(self):
            self.remove_same(other)
        elif isinstance(other, ScipyParticle):
            self.remove_single_by_object(other)
        else:
            raise TypeError("Trying to do an incremental removal of an element of type %s, which is not supported." % type(other))
        return self

    def pop_single_by_index(self, index):
        """
        Searches for Particle at index 'index', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If index is None, return last item (-1);
        If index < 0: return from 'end' of collection.
        If index is out of bounds, throws and OutOfRangeException.
        If Particle cannot be retrieved, returns None.

        :arg index: index of the Node to be popped (i.e. retrieved and removed) from this collections
        :returns last Node (if index == -1), indexed Node (if 0 < index < len(collection)) or None (if no Node can be retrieved)
        """
        super().pop_single_by_index(index)
        return self.remove_single_by_index(index)

    def pop_single_by_ID(self, id):
        """
        Searches for Particle with ID 'id', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If Particle cannot be retrieved (e.g. because the ID is not available), returns None.

        :arg id: 64-bit (signed or unsigned) integer ID of the Particle object to be popped (i.e. retrieved and removed) from this collections
        :returns identified Particle (if ID is related or an object contained in this collection) or None (if no Particle can be retrieved)
        """
        super().pop_single_by_ID(id)
        return self.remove_single_by_ID(id)

    def pop_multi_by_indices(self, indices):
        """
        Searches for Particles with the indices registered in 'indices', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If indices is None -> Particle cannot be retrieved -> Assert-Error and return None
        If index is None, return last item (-1);
        If index < 0: return from 'end' of collection.
        If index in 'indices' is out of bounds, throws and OutOfRangeException.
        If Particles cannot be retrieved, returns None.

        :arg index: a list or numpy.ndarray of indices of Nodes to be popped (i.e. retrieved and removed) from this collections
        :returns a list of retrieved Nodes
        """
        super().pop_multi_by_indices(indices)
        results = self._data[indices]
        self.remove_multi_by_indices(indices)
        return results

    def pop_multi_by_IDs(self, ids):
        """
        Searches for Particles with the IDs registered in 'ids', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If Particles cannot be retrieved (e.g. because the IDs are not available), returns None.

        :arg id: 64-bit (signed or unsigned) integer ID of the Node to be popped (i.e. retrieved and removed) from this collections
        :returns identified Node (if ID is related or an object contained in this collection) or None (if no Node can be retrieved)
        """
        super().pop_multi_by_IDs(ids)
        indices = self.get_indices_by_IDs(ids)
        return self.pop_multi_by_indices(indices)

    def _clear_deleted_(self):
        """
        This (protected) function physically removes particles from the collection whose status is set to 'DELETE'.
        It is the logical finalisation method of physically deleting particles that have been marked for deletion and
        that have not otherwise been recovered.
        This methods in heavily dependent on the actual collection type and should be implemented very specific
        to the actual data structure, to remove objects 'the fastest way possible'.
        """
        data_states = [p.state for p in self._data]
        indices = np.where(data_states == OperationCode)
        indices = None if len(indices) == 0 else indices[0]
        indices = None if indices.size == 0 else indices
        if indices is None:
            return
        if indices.size == 1:
            indices = indices[0]
            self.remove_single_by_index(indices)
        else:
            self.remove_multi_by_indices(indices)

    def merge(self, other=None):
        """
        This function merge two strictly equally-structured ParticleCollections into one. This can be, for example,
        quite handy to merge two particle subsets that - due to continuous removal - become too small to be effective.

        TODO - RETHINK IF THAT IS STILL THE WAY TO GO:
        On the other hand, this function can also internally merge individual particles that are tagged by status as
        being 'merged' (see the particle status for information on that).

        In order to distinguish both use cases, we can evaluate the 'same_class' parameter. In cases where this is
        'None', the merge operation semantically refers to an internal merge of individual particles - otherwise,
        it performs a 2-collection merge.

        Comment: the function can be simplified later by pre-evaluating the function parameter and then reference
        the individual, specific functions for internal- or external merge.

        The function shall return the merged ParticleCollection.
        """
        super().merge(other)

    def split(self, keys=None):
        """
        This function splits this collection into two disect equi-structured collections. The reason for it can, for
        example, be that the set exceeds a pre-defined maximum number of elements, which for performance reasons
        mandates a split.

        TODO - RETHINK IF THAT IS STILL THE WAY TO GO:
        On the other hand, this function can also internally split individual particles that are tagged byt status as
        to be 'split' (see the particle status for information on that).

        In order to distinguish both use cases, we can evaluate the 'indices' parameter. In cases where this is
        'None', the split operation semantically refers to an internal split of individual particles - otherwise,
        it performs a collection-split.

        Comment: the function can be simplified later by pre-evaluating the function parameter and then reference
        the individual, specific functions for element- or collection split.

        The function shall return the newly created or extended Particle collection, i.e. either the collection that
        results from a collection split or this very collection, containing the newly-split particles.
        """
        return super().split(keys)

    def __sizeof__(self):
        """
        This function returns the size in actual bytes required in memory to hold the collection. Ideally and simply,
        the size is computed as follows:

        sizeof(self) = len(self) * sizeof(pclass)
        :returns size of this collection in bytes; initiated by calling sys.getsizeof(object)
        """
        return self._data.nbytes+self._data_c.nbytes

    def clear(self):
        """
        This function physically removes all elements of the collection, yielding an empty collection as result of the
        operation.
        """
        if self._data is not None:
            del self._data
            self._data = None
        if self._data_c is not None:
            del self._data_c
            self._data_c = None
        self._ncount = 0

    def cstruct(self):
        """
        'cstruct' returns the ctypes mapping of the particle data. This depends on the specific structure in question.
        """
        cstruct = self._data_c.ctypes.data_as(c_void_p)
        return cstruct

    def toDictionary(self, pfile, time, deleted_only=False):
        """
        Convert all Particle data from one time step to a python dictionary.
        :param pfile: ParticleFile object requesting the conversion
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt,
         and one for all variables to be written once

        This function depends on the specific collection in question and thus needs to be specified in specific
        derivative classes.
        """
        data_dict = {}
        data_dict_once = {}

        time = time.total_seconds() if isinstance(time, delta) else time

        indices_to_write = []
        if pfile.lasttime_written != time and \
           (pfile.write_ondelete is False or deleted_only is not False):
            # if self._data.shape[0] == 0:
            if self._ncount == 0:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)
            else:
                if deleted_only is not False:
                    if type(deleted_only) not in [list, np.ndarray] and deleted_only in [True, 1]:
                        data_states = [p.state for p in self._data]
                        indices_to_write = np.where(np.isin(data_states, [OperationCode.Delete]))[0]
                    elif type(deleted_only) in [list, np.ndarray] and len(deleted_only) > 0:
                        if type(deleted_only[0]) in [int, np.int32, np.uint32]:
                            indices_to_write = deleted_only
                        elif isinstance(deleted_only[0], ScipyParticle):
                            indices_to_write = [i for i, p in self._data if p in deleted_only]
                else:
                    indices_to_write = _to_write_particles(self._data, time)
                if len(indices_to_write) > 0:
                    for var in pfile.var_names:
                        if 'id' in var:
                            data_dict[var] = np.array([np.int64(getattr(p, var)) for p in self._data[indices_to_write]])
                        else:
                            data_dict[var] = np.array([getattr(p, var) for p in self._data[indices_to_write]])

                pset_errs = [p for p in self._data[indices_to_write] if p.state != OperationCode.Delete and abs(time-p.time) > 1e-3 and np.isfinite(p.time)]
                for p in pset_errs:
                    logger.warning_once('time argument in pfile.write() is %g, but a particle has time % g.' % (time, p.time))

                if len(pfile.var_names_once) > 0:
                    # _to_write_particles(self._data, time)
                    first_write = [p for p in self._data if _is_particle_started_yet(p, time) and (np.int64(p.id) not in pfile.written_once)]
                    if np.any(first_write):
                        data_dict_once['id'] = np.array([p.id for p in first_write]).astype(dtype=np.int64)
                        for var in pfile.var_names_once:
                            data_dict_once[var] = np.array([getattr(p, var) for p in first_write])
                        pfile.written_once.extend(np.array(data_dict_once['id']).astype(dtype=np.int64).tolist())

            if deleted_only is False:
                pfile.lasttime_written = time

        return data_dict, data_dict_once

    def toArray(self):
        """
        This function converts (or: transforms; reformats; translates) this collection into an array-like structure
        (e.g. Python list or numpy nD array) that can be addressed by index. In the common case of 'no ID recovery',
        the global ID and the index match exactly.

        While this function may be very convenient for may users, it is STRONGLY DISADVISED to use the function too
        often, and the performance- and memory overhead malus may exceed any speed-up one could get from optimised
        data structures - in fact, for large collections with an implicit-order structure (i.e. ordered lists, sets,
        trees, etc.), this may be 'the most constly' function in any kind of simulation.

        It can be - though - useful at the final stage of a simulation to dump the results to disk.
        """
        return self._data.tolist()

    def set_variable_write_status(self, var, write_status):
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param status: Write status of the variable (True, False or 'once')
        """
        var_changed = False
        for v in self._ptype.variables:
            if v.name == var and hasattr(v, 'to_write'):
                v.to_write = write_status
                var_changed = True
        if var_changed:
            for p in self._data:
                attrib = getattr(p, var)
                if hasattr(attrib, 'to_write'):
                    attrib.to_write = write_status
                setattr(p, var, attrib)
        if not var_changed:
            raise SyntaxError('Could not change the write status of %s, because it is not a Variable name' % var)


class ParticleAccessorAOS(BaseParticleAccessor):
    """Wrapper that provides access to particle data in the collection,
    as if interacting with the particle itself.

    :param pcoll: ParticleCollection that the represented particle
                  belongs to.
    :param index: The index at which the data for the represented
                  particle is stored in the corresponding data arrays
                  of the ParticleCollecion.
    """
    _index = 0
    _next_dt = None

    def __init__(self, pcoll, index):
        """Initializes the ParticleAccessor to provide access to one
        specific particle.
        """
        super(ParticleAccessorAOS, self).__init__(pcoll)
        self._index = index
        self._next_dt = None

    def __getattr__(self, name):
        """Get the value of an attribute of the particle.

        :param name: Name of the requested particle attribute.
        :return: The value of the particle attribute in the underlying
                 collection data array.
        """
        if name in BaseParticleAccessor.__dict__.keys():
            result = super(ParticleAccessorAOS, self).__getattr__(name)
        elif name in type(self).__dict__.keys():
            result = object.__getattribute__(self, name)
        else:
            result = getattr(self._pcoll.data[self._index], name)
        return result

    def __setattr__(self, name, value):
        """Set the value of an attribute of the particle.

        :param name: Name of the particle attribute.
        :param value: Value that will be assigned to the particle
                      attribute in the underlying collection data array.
        """
        if name in BaseParticleAccessor.__dict__.keys():
            super(ParticleAccessorAOS, self).__setattr__(name, value)
        elif name in type(self).__dict__.keys():
            object.__setattr__(self, name, value)
        else:
            setattr(self._pcoll.data[self._index], name, value)

    def getPType(self):
        return self._pcoll.data[self._index].getPType()

    def update_next_dt(self, next_dt=None):
        if next_dt is None:
            if self._next_dt is not None:
                self._pcoll._data[self._index].dt = self._next_dt
                self._next_dt = None
        else:
            self._next_dt = next_dt

    def __repr__(self):
        return repr(self._pcoll.data[self._index])


class ParticleCollectionIterableAOS(BaseParticleCollectionIterable):

    def __init__(self, pcoll, reverse=False, subset=None):
        super(ParticleCollectionIterableAOS, self).__init__(pcoll, reverse, subset)

    def __iter__(self):
        return ParticleCollectionIteratorAOS(pcoll=self._pcoll_immutable, reverse=self._reverse, subset=self._subset)


class ParticleCollectionIteratorAOS(BaseParticleCollectionIterator):
    """Iterator for looping over the particles in the ParticleCollection.

    :param pcoll: ParticleCollection that stores the particles.
    :param reverse: Flag to indicate reverse iteration (i.e. starting at
                    the largest index, instead of the smallest).
    :param subset: Subset of indices to iterate over, this allows the
                   creation of an iterator that represents part of the
                   collection.
    """

    def __init__(self, pcoll, reverse=False, subset=None):

        if subset is not None:
            if len(subset) > 0 and type(subset[0]) not in [int, np.int32, np.intp]:
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
        self._pcoll = pcoll
        self._prev_index = -1
        self._index = 0
        self._head = None
        self._tail = None
        if len(self._indices) > 0:
            self._head = ParticleAccessorAOS(pcoll, self._indices[0])
            self._tail = ParticleAccessorAOS(pcoll, self._indices[self.max_len - 1])
        self.p = None

    def __next__(self):
        """Returns a ParticleAccessor for the next particle in the
        ParticleSet.
        """
        if self._index < self.max_len:
            self._prev_index = self._index
            self._index += 1
            return ParticleAccessorAOS(self._pcoll, self._indices[self._prev_index])
        raise StopIteration

    @property
    def current(self):
        if (self.max_len > self._prev_index) and (self._prev_index > -1):
            return ParticleAccessorAOS(self._pcoll, self._indices[self._prev_index])
        raise IndexError

    def __repr__(self):
        dir_str = 'Backward' if self._reverse else 'Forward'
        return "%s iteration at index %s of %s." % (dir_str, self._index, self.max_len)
