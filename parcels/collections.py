from abc import ABC
from ctypes import POINTER, Structure
from operator import attrgetter

import numpy as np

from parcels.tools.converters import convert_to_flat_array
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import StatusCode

try:
    from mpi4py import MPI
except:
    MPI = None
if MPI:
    try:
        from sklearn.cluster import KMeans
    except:
        KMeans = None


class ParticleCollection(ABC):

    def __init__(self, pclass, lon, lat, depth, time, lonlatdepth_dtype, pid_orig, partitions=None, ngrid=1, **kwargs):
        """
        Parameters
        ----------
        ngrid :
            number of grids in the fieldset of the overarching ParticleSet - required for initialising the
            field references of the ctypes-link of particles that are allocated
        """
        self._ncount = -1
        self._pu_indicators = None
        self._offset = 0
        self._pclass = None
        self._ptype = None
        self._latlondepth_dtype = np.float32
        self._data = None

        assert pid_orig is not None, "particle IDs are None - incompatible with the collection. Invalid state."
        pid = pid_orig + pclass.lastID

        self._sorted = np.all(np.diff(pid) >= 0)

        assert depth is not None, "particle's initial depth is None - incompatible with the collection. Invalid state."
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        # If partitions is false, the partitions are already initialised
        if partitions is not None and partitions is not False:
            self._pu_indicators = convert_to_flat_array(partitions)

        for kwvar in kwargs:
            assert lon.size == kwargs[kwvar].size, (
                f"{kwvar} and positions (lon, lat, depth) don't have the same lengths.")

        offset = np.max(pid) if (pid is not None) and len(pid) > 0 else -1
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()

            if lon.size < mpi_size and mpi_size > 1:
                raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

            if mpi_size > 1:
                if partitions is not False:
                    if (self._pu_indicators is None) or (len(self._pu_indicators) != len(lon)):
                        if mpi_rank == 0:
                            coords = np.vstack((lon, lat)).transpose()
                            if KMeans:
                                kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
                                self._pu_indicators = kmeans.labels_
                            else:  # assigning random labels if no KMeans (see https://github.com/OceanParcels/parcels/issues/1261)
                                logger.warning_once('sklearn needs to be available if MPI is installed. '
                                                    'See https://docs.oceanparcels.org/en/latest/installation.html#installation-for-developers for more information')
                                self._pu_indicators = np.randint(0, mpi_size, size=len(lon))
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

        self._ptype = pclass.getPType()
        self._data = {}
        initialised = set()

        self._ncount = len(lon)

        for v in self.ptype.variables:
            if v.name in ['xi', 'yi', 'zi', 'ti']:
                self._data[v.name] = np.empty((len(lon), ngrid), dtype=v.dtype)
            else:
                self._data[v.name] = np.empty(self._ncount, dtype=v.dtype)

        if lon is not None and lat is not None:
            # Initialise from lists of lon/lat coordinates
            assert self._ncount == len(lon) and self._ncount == len(lat), (
                'Size of ParticleSet does not match length of lon and lat.')

            # mimic the variables that get initialised in the constructor
            self._data['lat'][:] = lat
            self._data['lat_nextloop'][:] = lat
            self._data['lon'][:] = lon
            self._data['lon_nextloop'][:] = lon
            self._data['depth'][:] = depth
            self._data['depth_nextloop'][:] = depth
            self._data['time'][:] = time
            self._data['time_nextloop'][:] = time
            self._data['id'][:] = pid
            self._data['obs_written'][:] = 0

            # special case for exceptions which can only be handled from scipy
            self._data['exception'] = np.empty(self._ncount, dtype=object)

            initialised |= {'lat', 'lat_nextloop', 'lon', 'lon_nextloop', 'depth', 'depth_nextloop', 'time', 'time_nextloop', 'id', 'obs_written'}

            # any fields that were provided on the command line
            for kwvar, kwval in kwargs.items():
                if not hasattr(pclass, kwvar):
                    raise RuntimeError(f'Particle class does not have Variable {kwvar}')
                self._data[kwvar][:] = kwval
                initialised.add(kwvar)

            # initialise the rest to their default values
            for v in self.ptype.variables:
                if v.name in initialised:
                    continue

                if isinstance(v.initial, attrgetter):
                    self._data[v.name][:] = v.initial(self)
                else:
                    self._data[v.name][:] = v.initial

                initialised.add(v.name)
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")
        self._iterator = None
        self._riterator = None

    def __del__(self):
        """Collection - Destructor"""
        pass

    @property
    def pu_indicators(self):
        """
        The 'pu_indicator' is an [array or dictionary]-of-indicators, where each indicator entry tells per item
        (i.e. particle) in the collection to which processing unit (PU) in a parallelised setup it belongs to.
        """
        return self._pu_indicators

    @property
    def pclass(self):
        """Stores the actual class type of the particles allocated and managed in this collection."""
        return self._pclass

    @property
    def ptype(self):
        """
        'ptype' returns an instance of the particular type of class 'ParticleType' of the particle class of the particles
        in this collection.

        basically:
        pytpe -> pclass().getPType()
        """
        return self._ptype

    @property
    def lonlatdepth_dtype(self):
        """
        'lonlatdepth_dtype' stores the numeric data type that is used to represent the lon, lat and depth of a particle.
        This can be either 'float32' (default) or 'float64'
        """
        return self._lonlatdepth_dtype

    @property
    def data(self):
        """
        'data' is a reference to the actual barebone-storage of the particle data, and thus depends directly on the
        specific collection in question.
        """
        return self._data

    def __len__(self):
        """This function returns the length, in terms of 'number of elements, of a collection."""
        return self._ncount

    def iterator(self):
        self._iterator = ParticleCollectionIterator(self)
        return self._iterator

    def __iter__(self):
        """Returns an Iterator that allows for forward iteration over the
        elements in the ParticleCollection (e.g. `for p in pset:`).
        """
        return self.iterator()

    def __getitem__(self, index):
        """
        Access a particle in this collection using the fastest access
        method for this collection - by its index.

        Parameters
        ----------
        index : int
            Index of the particle to access
        """
        return self.get_single_by_index(index)

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        Parameters
        ----------
        name : str
            Name of the property to access
        """
        for v in self.ptype.variables:
            if v.name == name and name in self._data:
                return self._data[name]
        return False

    def get_single_by_index(self, index):
        """
        This function gets a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure.
        In cases where a get-by-index would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-ID.
        """
        assert type(index) in [int, np.int32, np.intp], f"Trying to get a particle by index, but index {index} is not a 32-bit integer - invalid operation."
        return ParticleAccessor(self, index)

    def add_same(self, same_class):
        """
        Adds another, equi-structured ParticleCollection to this collection. This is done by concatenating
        both collections. The fact that they are of the same ParticleCollection's derivative simplifies
        parsing and concatenation.
        """
        assert same_class is not None, f"Trying to add another {type(self)} to this one, but the other one is None - invalid operation."
        assert type(same_class) is type(self)

        if same_class._ncount == 0:
            return

        if self._ncount == 0:
            self._data = same_class._data
            self._ncount = same_class._ncount
            return

        # Determine order of concatenation and update the sorted flag
        if self._sorted and same_class._sorted \
           and self._data['id'][0] > same_class._data['id'][-1]:
            for d in self._data:
                self._data[d] = np.concatenate((same_class._data[d], self._data[d]))
            self._ncount += same_class._ncount
        else:
            if not (same_class._sorted
                    and self._data['id'][-1] < same_class._data['id'][0]):
                self._sorted = False
            for d in self._data:
                self._data[d] = np.concatenate((self._data[d], same_class._data[d]))
            self._ncount += same_class._ncount

    def __iadd__(self, same_class):
        """
        Performs an incremental addition of the equi-structured ParticleCollections, such to allow

        a += b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place addition of (an) element(s).
        """
        assert same_class is not None
        assert type(same_class) is type(self), f"Trying to increment-add collection of type {type(same_class)} into collection of type {type(self)} - invalid operation."
        self.add_same(same_class)
        return self

    def remove_single_by_index(self, index):
        """
        This function removes a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure,
        and then perform the removal.
        In cases where a removal-by-index would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-object or remove-by-ID.
        """
        assert type(index) in [int, np.int32, np.intp], f"Trying to remove a particle by index, but index {index} is not a 32-bit integer - invalid operation."

        for d in self._data:
            self._data[d] = np.delete(self._data[d], index, axis=0)

        self._ncount -= 1

    def remove_multi_by_indices(self, indices):
        """
        This function removes particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a removal-via-object-reference strategy.
        """
        assert indices is not None, "Trying to remove particles by their collection indices, but the index list is None - invalid operation."
        assert type(indices) in [list, dict, np.ndarray], "Trying to remove particles by their indices, but the index container is not a valid Python-collection - invalid operation."
        if type(indices) is not dict:
            assert len(indices) == 0 or type(indices[0]) in [int, np.int32, np.intp], "Trying to remove particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        else:
            assert len(list(indices.values())) == 0 or type(list(indices.values())[0]) in [int, np.int32, np.intp], "Trying to remove particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        if type(indices) is dict:
            indices = list(indices.values())

        for d in self._data:
            self._data[d] = np.delete(self._data[d], indices, axis=0)

        self._ncount -= len(indices)

    def cstruct(self):
        """Returns the ctypes mapping of the particle data. This depends on the specific structure in question."""
        class CParticles(Structure):
            _fields_ = [(v.name, POINTER(np.ctypeslib.as_ctypes_type(v.dtype))) for v in self._ptype.variables]

        def flatten_dense_data_array(vname):
            data_flat = self._data[vname].view()
            data_flat.shape = -1
            return np.ctypeslib.as_ctypes(data_flat)

        cdata = [flatten_dense_data_array(v.name) for v in self._ptype.variables]
        cstruct = CParticles(*cdata)
        return cstruct

    def _to_write_particles(self, pd, time):
        """We don't want to write a particle that is not started yet.
        Particle will be written if particle.time is between time-dt/2 and time+dt (/2)
        """
        return np.where((np.less_equal(time - np.abs(pd['dt'] / 2), pd['time'], where=np.isfinite(pd['time']))
                        & np.greater_equal(time + np.abs(pd['dt'] / 2), pd['time'], where=np.isfinite(pd['time']))
                        | ((np.isnan(pd['dt'])) & np.equal(time, pd['time'], where=np.isfinite(pd['time']))))
                        & (np.isfinite(pd['id']))
                        & (np.isfinite(pd['time'])))[0]

    def getvardata(self, var, indices=None):
        if indices is None:
            return self._data[var]
        else:
            try:
                return self._data[var][indices]
            except:  # Can occur for zero-length ParticleSets
                return None

    def setvardata(self, var, index, val):
        self._data[var][index] = val

    def setallvardata(self, var, val):
        self._data[var][:] = val

    def set_variable_write_status(self, var, write_status):
        """Method to set the write status of a Variable

        Parameters
        ----------
        var :
            Name of the variable (string)
        status :
            Write status of the variable (True, False or 'once')
        write_status :
        """
        var_changed = False
        for v in self._ptype.variables:
            if v.name == var:
                v.to_write = write_status
                var_changed = True
        if not var_changed:
            raise SyntaxError(f'Could not change the write status of {var}, because it is not a Variable name')


class ParticleAccessor(ABC):
    """Wrapper that provides access to particle data in the collection,
    as if interacting with the particle itself.

    Parameters
    ----------
    pcoll :
        ParticleCollection that the represented particle
        belongs to.
    index :
        The index at which the data for the represented
        particle is stored in the corresponding data arrays
        of the ParticleCollecion.
    """

    _pcoll = None
    _index = 0

    def __init__(self, pcoll, index):
        """Initializes the ParticleAccessor to provide access to one
        specific particle.
        """
        self._pcoll = pcoll
        self._index = index

    def __getattr__(self, name):
        """
        Get the value of an attribute of the particle.

        Parameters
        ----------
        name : str
            Name of the requested particle attribute.

        Returns
        -------
        any
            The value of the particle attribute in the underlying collection data array.
        """
        if name in self.__dict__.keys():
            result = self.__getattribute__(name)
        elif name in type(self).__dict__.keys():
            result = object.__getattribute__(self, name)
        else:
            result = self._pcoll.data[name][self._index]
        return result

    def __setattr__(self, name, value):
        """
        Set the value of an attribute of the particle.

        Parameters
        ----------
        name : str
            Name of the particle attribute.
        value : any
            Value that will be assigned to the particle attribute in the underlying collection data array.
        """
        if name in self.__dict__.keys():
            self.__setattr__(name, value)
        elif name in type(self).__dict__.keys():
            object.__setattr__(self, name, value)
        else:
            self._pcoll.data[name][self._index] = value

    def getPType(self):
        return self._pcoll.ptype

    def __repr__(self):
        time_string = 'not_yet_set' if self.time is None or np.isnan(self.time) else f"{self.time:f}"
        str = "P[%d](lon=%f, lat=%f, depth=%f, " % (self.id, self.lon, self.lat, self.depth)
        for var in self._pcoll.ptype.variables:
            if var.name in ['lon_nextloop', 'lat_nextloop', 'depth_nextloop', 'time_nextloop']:  # TODO check if time_nextloop is needed (or can work with time-dt?)
                continue
            if var.to_write is not False and var.name not in ['id', 'lon', 'lat', 'depth', 'time']:
                str += f"{var.name}={getattr(self, var.name):f}, "
        return str + f"time={time_string})"

    def delete(self):
        """Signal the underlying particle for deletion."""
        self.state = StatusCode.Delete


class ParticleCollectionIterator(ABC):
    """Iterator for looping over the particles in the ParticleCollection.

    Parameters
    ----------
    pcoll :
        ParticleCollection that stores the particles.
    subset :
        Subset of indices to iterate over, this allows the
        creation of an iterator that represents part of the
        collection.
    """

    def __init__(self, pcoll, subset=None):

        if subset is not None:
            if len(subset) > 0 and type(subset[0]) not in [int, np.int32, np.intp]:
                raise TypeError("Iteration over a subset of particles in the"
                                " particleset requires a list or numpy array"
                                " of indices (of type int or np.int32).")
            self._indices = subset
            self.max_len = len(subset)
        else:
            self.max_len = len(pcoll)
            self._indices = range(self.max_len)

        self._pcoll = pcoll
        self._index = 0

    def __iter__(self):
        """Returns the iterator itself."""
        return self

    def __next__(self):
        """Returns a ParticleAccessor for the next particle in the
        ParticleSet.
        """
        if self._index < self.max_len:
            self.p = ParticleAccessor(self._pcoll, self._indices[self._index])
            self._index += 1
            return self.p

        raise StopIteration
