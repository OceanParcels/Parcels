import random
from abc import ABC, abstractmethod
from threading import Thread
from .message_service import mpi_execute_requested_messages as executor
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None


class BaseIdGenerator(ABC):
    """
    Abstract class defining the principle member variables and supported functions each specific ID generator needs to
    contain.
    """
    _total_ids = 0
    _used_ids = 0
    _recover_ids = False
    _map_id_totalindex = dict()
    _track_id_index = True

    def __init__(self):
        """
        BaseIdGenerator - abstract Constructor
        """
        self._total_ids = 0
        self._used_ids = 0
        self._recover_ids = False
        self._map_id_totalindex = dict()
        self._track_id_index = True

    def __del__(self):
        """
        BaseIdGenerator - abstract Destructor
        """
        self._total_ids = 0
        self._used_ids = 0
        self._track_id_index = False
        self._map_id_totalindex.clear()

    def setTimeLine(self, min_time, max_time):
        """
        abstract function - setting min-max limits to the 'time' dimension of the ID
        :arg min_time: lowest time value used during the simulation. For forward simulation, this would be 't_0'.
        :arg max_time: highest time value used during the simulation. For forward simulation, this would be the runtime or 't_N'.
        """
        pass

    def setDepthLimits(self, min_depth, max_depth):
        """
        abstract function - setting min-max limits to the 'depth' dimension of the ID
        :arg min_depth: lowest depth value during the simulation. With depth being measured from the sea surface, this value would be the sea surface itself (`min_depth=0`).
        :arg max_depth: highest depth value during the simulation. When depth is measured from the sea surface positive downward, this value would be the deepest level of the model.
        """
        pass

    def preGenerateIDs(self, high_value):
        """
        abstract function - pre-allocating a range of IDs from 0 up to :arg high_value.
        :arg high_value: (u)int64 value of the highest pre-generated ID itself.
        """
        pass

    def permuteIDs(self):
        """
        abstract function - randomizes pre-generated IDs
        """
        pass

    def close(self):
        """
        abstract function, closing the ID generator, either releasing (if enabled) or destroying (otherwise) the managed IDs
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        abstract function - returns the number of generated IDs
        """
        pass

    @property
    def total_length(self):
        """
        :returns the total number of generated IDs, incl. pre-generated IDs and excl. the ID recovery
        """
        return self._total_ids

    @property
    def usable_length(self):
        """
        :returns the usable number of generated IDs, excl. pre-generated IDs and considering the ID recovery
        """
        return self._used_ids

    @property
    def recover_ids(self):
        """
        :returns if ID recovery is used or not
        """
        return self._recover_ids

    @recover_ids.setter
    def recover_ids(self, bool_param):
        """
        Sets if IDs are recovered or not
        :arg bool_param: flag to use- or not-use ID recovery
        """
        self._recover_ids = bool_param

    def enable_ID_recovery(self):
        """
        this function enables ID recovery
        """
        self._recover_ids = True

    def disable_ID_recovery(self):
        """
        this function disables ID recovery
        """
        self._recover_ids = False

    @abstractmethod
    def getID(self, lon, lat, depth, time):
        """
        abstract function - major function to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        pass

    def nextID(self, lon, lat, depth, time):
        """
        external renamed method to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        return self.getID(lon, lat, depth, time)

    @abstractmethod
    def releaseID(self, id):
        """
        abstract function - releases an ID, either by just removing it from the ID list or by
                            recovering the ID, reusing it for the next 'getID(...)' call
        :arg id: ID to be released (and possibly recovered)
        """
        pass

    @abstractmethod
    def get_length(self):
        """
        function forward to
        :returns the number of generated IDs
        """
        return self.__len__()

    def get_total_length(self):
        """
        function forward to
        :returns the total number of generated IDs, incl. pre-generated IDs and excl. the ID recovery
        """
        return self._total_ids

    def get_usable_length(self):
        """
        function forward to
        :returns the usable number of generated IDs, excl. pre-generated IDs and considering the ID recovery
        """
        return self._used_ids

    def enable_id_index_tracking(self):
        """
        Enables the tracking of ID to index, so that the ID can reliably be associated to an index
        """
        self._track_id_index = True

    def disable_id_index_tracking(self):
        """
        Disables the tracking of ID to index
        """
        self._track_id_index = False

    def map_id_to_index(self, input_id):
        """
        Maps an ID to its index, if ID tracking is enabled
        :arg input_id: ID to be associated with an index
        :returns list index
        """
        index = None
        if self._track_id_index:
            try:
                index = self._map_id_totalindex[input_id]
            except (IndexError, ValueError, KeyError):
                index = None
        return index

    def is_tracking_id_index(self):
        """
        :returns if ID tracking is enabled or not
        """
        return self._track_id_index


class SequentialIdGenerator(BaseIdGenerator):
    """
    A principal class that generates IDs sequentially. It is possible to pre-generate IDs which can also be permutated.
    ID recovery is easily achievable.
    """
    released_ids = []
    next_id = 0

    def __init__(self):
        """
        SequentialIdGenerator - Constructor
        """
        super(SequentialIdGenerator, self).__init__()
        self.released_ids = []
        self.next_id = np.uint64(0)
        self._recover_ids = False

    def __del__(self):
        """
        SequentialIdGenerator - Destructor
        """
        if len(self.released_ids) > 0:
            del self.released_ids
        super(SequentialIdGenerator, self).__del__()

    def getID(self, lon, lat, depth, time):
        """
        Major function to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        n = len(self.released_ids)
        if n == 0:
            result = self.next_id
            self.next_id += 1
            if self._track_id_index:
                self._map_id_totalindex[result] = self._total_ids
            self._total_ids += 1
            self._used_ids += 1
            return np.uint64(result)
        else:
            result = self.released_ids.pop(n-1)
            if self._track_id_index:
                self._map_id_totalindex[result] = self._total_ids
            self._total_ids += 1
            self._used_ids += 1
            return np.uint64(result)

    def releaseID(self, id):
        """
        releases an ID, either by just removing it from the ID list or by
        recovering the ID, reusing it for the next 'getID(...)' call
        :arg id: ID to be released (and possibly recovered)
        """
        if not self._recover_ids:
            return
        self.released_ids.append(id)
        self._used_ids -= 1

    def preGenerateIDs(self, high_value):
        """
        abstract function - pre-allocating a range of IDs from 0 up to :arg high_value.
        :arg high_value: (u)int64 value of the highest pre-generated ID itself.
        """
        if len(self.released_ids) > 0:
            self.released_ids.clear()
        self.released_ids = [i for i in range(0, high_value)]
        self.next_id = high_value

    def permuteIDs(self):
        """
        abstract function - randomizes pre-generated IDs
        """
        n = len(self.released_ids)
        indices = random.randint(0, n, 2*n)
        for index in indices:
            id = self.released_ids.pop(index)
            self.released_ids.append(id)

    def __len__(self):
        """
        :returns the number of generated IDs
        """
        return self.next_id

    def get_length(self):
        """
        function forward to
        :returns the number of generated IDs
        """
        return len(self)


class SpatialIdGenerator(BaseIdGenerator):
    """
    Generates 64-bit IDs based on 3D spatial coordinates, aiming for an ID distribution so that spatially-close
    items have also close-by IDs.
    """
    _lon_bins = 360
    _lat_bins = 180
    _depth_bins = 32768
    _lonbounds = np.zeros(2, dtype=np.float32)
    _latbounds = np.zeros(2, dtype=np.float32)
    _depthbounds = np.zeros(2, dtype=np.float32)
    local_ids = None
    released_ids = {}

    def __init__(self, lon_bins=360, lat_bins=180, depth_bins=32768):
        """
        SpatialIdGenerator - Constructor

        ID generator that manages IDs in a spatial mapping scheme, so that IDs
        that are spatially close are also numerically close.

        Attention: the bins are used in a bit allocation scheme so that
        (log2(arg:lon_bins) * long2(arg::lat_bins) * log2(arg:depth_bins)) <= 32
        """
        super(SpatialIdGenerator, self).__init__()
        self._lonbounds = np.array([-180.0, 180.0], dtype=np.float32)
        self._latbounds = np.array([-90.0, 90.0], dtype=np.float32)
        self._depthbounds = np.array([0.0, 1.0], dtype=np.float32)
        self._lon_bins = lon_bins
        self._lat_bins = lat_bins
        self._depth_bins = depth_bins
        self.local_ids = np.zeros((self._lon_bins, self._lat_bins, self._depth_bins), dtype=np.uint32)
        self.released_ids = {}  # 32-bit spatio-temporal index => []
        self._recover_ids = False

    def __del__(self):
        """
        SpatialIdGenerator - Destructor
        """
        if self.local_ids is not None:
            del self.local_ids
        if len(self.released_ids) > 0:
            del self.released_ids
        super(SpatialIdGenerator, self).__del__()

    def setLonLimits(self, min_lon=-180.0, max_lon=180.0):
        """
        Setting min-max limits to the 'longitude' dimension of the ID
        :arg min_lon: lowest longitude value used during the simulation. For spherical maps, this would be -180.
        :arg max_lon: highest longitude value used during the simulation. For spherical maps, this would be +180.
        """
        self._lonbounds = np.array([min_lon, max_lon], dtype=np.float32)

    def setLatLimits(self, min_lat=-90.0, max_lat=90.0):
        """
        Setting min-max limits to the 'latitude' dimension of the ID
        :arg min_lat: lowest latitude value used during the simulation. For spherical maps, this would be -90.
        :arg max_lat: highest latitude value used during the simulation. For spherical maps, this would be +90.
        """
        self._latbounds = np.array([min_lat, max_lat], dtype=np.float32)

    def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
        """
        abstract function - setting min-max limits to the 'depth' dimension of the ID
        :arg min_depth: lowest depth value during the simulation. With depth being measured from the sea surface, this value would be the sea surface itself (`min_depth=0`).
        :arg max_depth: highest depth value during the simulation. With depth being measured from the sea surface positive downward, this value would be the deepest level of the model.
        """
        self._depthbounds = np.array([min_depth, max_depth], dtype=np.float32)

    def getID(self, lon, lat, depth, time=None):
        """
        abstract function - major function to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        idlon = lon  # avoid original 'lon' changes from change-by-ref artefacts
        idlat = lat  # avoid original 'lat' changes from change-by-ref artefacts
        iddepth = depth  # avoid original 'depth' changes from change-by-ref artefacts
        if idlon < self._lonbounds[0]:
            vsgn = np.sign(idlon)
            idlon = np.fmod(np.fabs(idlon), np.fabs(self._lonbounds[0])) * vsgn
        if idlon > self._lonbounds[1]:
            vsgn = np.sign(idlon)
            idlon = np.fmod(np.fabs(idlon), np.fabs(self._lonbounds[1])) * vsgn
        if idlat < self._latbounds[0]:
            vsgn = np.sign(idlat)
            idlat = np.fmod(np.fabs(idlat), np.fabs(self._latbounds[0])) * vsgn
        if idlat > self._latbounds[1]:
            vsgn = np.sign(idlat)
            idlat = np.fmod(np.fabs(idlat), np.fabs(self._latbounds[1])) * vsgn
        if iddepth is None:
            iddepth = self._depthbounds[0]
        if iddepth < self._depthbounds[0] or depth > self._depthbounds[1]:
            vsgn = np.sign(depth)
            iddepth = np.fmod(np.fabs(iddepth), np.fabs(max(self._depthbounds))) * vsgn if min(self._depthbounds) > 0 else max(self._depthbounds) - (np.fmod(np.fabs(iddepth), max(np.fabs(self._depthbounds))) * vsgn)
        lon_discrete = (idlon - self._lonbounds[0]) / (self._lonbounds[1] - self._lonbounds[0])
        lon_discrete = np.int32((self._lon_bins-1) * lon_discrete)
        lat_discrete = (idlat - self._latbounds[0]) / (self._latbounds[1] - self._latbounds[0])
        lat_discrete = np.int32((self._lat_bins-1) * lat_discrete)
        depth_discrete = (iddepth - self._depthbounds[0])/(self._depthbounds[1]-self._depthbounds[0])
        depth_discrete = np.int32((self._depth_bins-1) * depth_discrete)
        lon_index = np.uint32(np.int32(lon_discrete))
        lat_index = np.uint32(np.int32(lat_discrete))
        depth_index = np.uint32(np.int32(depth_discrete))
        id = self._get_next_id(lon_index, lat_index, depth_index, None)
        return id

    def nextID(self, lon, lat, depth, time):
        """
        external renamed method to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        return self.getID(lon, lat, depth, time)

    def releaseID(self, id):
        """
        releases an ID, either by just removing it from the ID list or by
        recovering the ID, reusing it for the next 'getID(...)' call
        :arg id: ID to be released (and possibly recovered)
        """
        full_bits = np.uint32(4294967295)
        nil_bits = np.int32(0)
        spatiotemporal_id = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(full_bits), 32), np.int64(nil_bits)), np.int64(id))
        spatiotemporal_id = np.uint32(np.right_shift(spatiotemporal_id, 32))
        local_id = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(nil_bits), 32), np.int64(full_bits)), np.int64(id))
        local_id = np.uint32(local_id)
        self._release_id(spatiotemporal_id, local_id)

    def __len__(self):
        """
        :returns the number of generated IDs
        """
        return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])

    def get_length(self):
        """
        function forward to
        :returns the number of generated IDs
        """
        return self.__len__()

    def _get_next_id(self, lon_index, lat_index, depth_index, time_index=None):
        """
        private function - generates a conclusive 64-bit ID from clamped lon-lat-depth-time indices
        """
        local_index = -1
        lon_shift = 32-int(np.ceil(np.log2(self._lon_bins)))
        lat_shift = lon_shift-int(np.ceil(np.log2(self._lat_bins)))
        id = np.left_shift(lon_index, lon_shift) + np.left_shift(lat_index, lat_shift) + depth_index
        if len(self.released_ids) > 0 and (id in self.released_ids.keys()) and len(self.released_ids[id]) > 0:
            local_index = np.uint32(self.released_ids[id].pop())
            if len(self.released_ids[id]) <= 0:
                del self.released_ids[id]
        else:
            local_index = self.local_ids[lon_index, lat_index, depth_index]
            self.local_ids[lon_index, lat_index, depth_index] += 1
        id = np.int64(id)
        id = np.bitwise_or(np.left_shift(id, 32), np.int64(local_index))
        id = np.uint64(id)
        if self._track_id_index:
            self._map_id_totalindex[id] = self._total_ids
        self._total_ids += 1
        self._used_ids += 1
        return id

    def _release_id(self, spatiotemporal_id, local_id):
        """
        private function - releases a 64-bit ID by adding the 32-bit local ID to the spatial lon-lat-depth-time ID map
        :arg spatiotemporal_id: 32-bit spatial ID segment
        :arg local_id: 32-bit sequential ID segment
        """
        if not self._recover_ids:
            return
        if spatiotemporal_id not in self.released_ids.keys():
            self.released_ids[spatiotemporal_id] = []
        self.released_ids[spatiotemporal_id].append(local_id)
        self._used_ids -= 1


class SpatioTemporalIdGenerator(BaseIdGenerator):
    """
    Generates 64-bit IDs based on 4D spatiotemporal coordinates, aiming for an ID distribution so that spatially-close
    items have also close-by IDs.
    """
    timebounds = np.zeros(2, dtype=np.float64)
    depthbounds = np.zeros(2, dtype=np.float32)
    local_ids = None
    released_ids = {}

    def __init__(self):
        """
        SpatioTemporalIDGenerator - Constructor

        ID generator that manages IDs in a spatial & temporal mapping scheme, so that IDs
        that are spatially and temporally close are also numerically close.

        Attention: the bins are used in a bit allocation scheme so that
        (log2(arg:lon_bins) * long2(arg::lat_bins) * log2(arg:depth_bins) * log2(arg:time_bins)) <= 32
        """
        super(SpatioTemporalIdGenerator, self).__init__()
        self._timebounds = np.array([0, 1.0], dtype=np.float64)
        self._depthbounds = np.array([0, 1.0], dtype=np.float32)
        self.local_ids = np.zeros((360, 180, 128, 256), dtype=np.uint32)
        self.released_ids = {}  # 32-bit spatio-temporal index => []
        self._recover_ids = False

    def __del__(self):
        """
        SpatioTemporalIDGenerator - Destructor
        """
        if self.local_ids is not None:
            del self.local_ids
        if len(self.released_ids) > 0:
            del self.released_ids

    def setTimeLine(self, min_time=0.0, max_time=1.0):
        """
        abstract function - setting min-max limits to the 'time' dimension of the ID
        :arg min_time: lowest time value used during the simulation. For forward simulation, this would be 't_0'.
        :arg max_time: highest time value used during the simulation. For forward simulation, this would be the runtime or 't_N'.
        """
        self._timebounds = np.array([min_time, max_time], dtype=np.float64)

    def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
        """
        abstract function - setting min-max limits to the 'depth' dimension of the ID
        :arg min_depth: lowest depth value during the simulation. With depth being measured from the sea surface, this value would be the sea surface itself (`min_depth=0`).
        :arg max_depth: highest depth value during the simulation. With depth being measured from the sea surface positive downward, this value would be the deepest level of the model.
        """
        self._depthbounds = np.array([min_depth, max_depth], dtype=np.float32)

    def getID(self, lon, lat, depth, time):
        """
        abstract function - major function to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        idlon = lon  # avoid original 'lon' changes from change-by-ref artefacts
        idlat = lat  # avoid original 'lat' changes from change-by-ref artefacts
        iddepth = depth  # avoid original 'depth' changes from change-by-ref artefacts
        idtime = time  # avoid original 'time' changes from change-by-ref artefacts
        if idlon < -180.0 or idlon > 180.0:
            vsgn = np.sign(idlon)
            idlon = np.fmod(np.fabs(idlon), 180.0) * vsgn
        if idlat < -90.0 or idlat > 90.0:
            vsgn = np.sign(idlat)
            idlat = np.fmod(np.fabs(idlat), 90.0) * vsgn
        if iddepth is None:
            iddepth = self._depthbounds[0]
        if iddepth < self._depthbounds[0] or iddepth > self._depthbounds[1]:
            vsgn = np.sign(iddepth)
            iddepth = np.fmod(np.fabs(iddepth), np.fabs(max(self._depthbounds))) * vsgn if min(self._depthbounds) > 0 else max(self._depthbounds) - (np.fmod(np.fabs(iddepth), max(np.fabs(self._depthbounds))) * vsgn)
        if idtime is None:
            idtime = self._timebounds[0]
        if idtime < self._timebounds[0] or idtime > self._timebounds[1]:
            vsgn = np.sign(idtime)
            idtime = np.fmod(np.fabs(idtime), np.fabs(max(self._timebounds))) * vsgn if min(self._timebounds) > 0 else max(self._timebounds) - (np.fmod(np.fabs(idtime), max(np.fabs(self._timebounds))) * vsgn)
        lon_discrete = np.int32(min(max(idlon, -179.9), 179.9))
        lat_discrete = np.int32(min(max(idlat, -179.9), 179.9))
        depth_discrete = (iddepth-self._depthbounds[0])/(self._depthbounds[1]-self._depthbounds[0])
        depth_discrete = np.int32(127.0 * depth_discrete)
        time_discrete = (idtime-self._timebounds[0])/(self._timebounds[1]-self._timebounds[0])
        time_discrete = np.int32(255.0 * time_discrete)
        lon_index = np.uint32(np.int32(lon_discrete)+180)
        lat_index = np.uint32(np.int32(lat_discrete)+90)
        depth_index = np.uint32(np.int32(depth_discrete))
        time_index = np.uint32(np.int32(time_discrete))
        id = self._get_next_id(lon_index, lat_index, depth_index, time_index)
        return id

    def nextID(self, lon, lat, depth, time):
        """
        external renamed method to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        return self.getID(lon, lat, depth, time)

    def releaseID(self, id):
        """
        function - releases an ID, either by just removing it from the ID list or by
                            recovering the ID, reusing it for the next 'getID(...)' call
        :arg id: ID to be released (and possibly recovered)
        """
        full_bits = np.uint32(4294967295)
        nil_bits = np.int32(0)
        spatiotemporal_id = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(full_bits), 32), np.int64(nil_bits)), np.int64(id))
        spatiotemporal_id = np.uint32(np.right_shift(spatiotemporal_id, 32))
        local_id = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(nil_bits), 32), np.int64(full_bits)), np.int64(id))
        local_id = np.uint32(local_id)
        self._release_id(spatiotemporal_id, local_id)

    def __len__(self):
        """
        :returns the number of generated IDs
        """
        return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])

    def get_length(self):
        """
        function forward to
        :returns the number of generated IDs
        """
        return self.__len__()

    def _get_next_id(self, lon_index, lat_index, depth_index, time_index):
        """
        private function - generates a conclusive 64-bit ID from clamped lon-lat-depth-time indices
        """
        local_index = -1
        id = np.left_shift(lon_index, 23) + np.left_shift(lat_index, 15) + np.left_shift(depth_index, 8) + time_index
        if len(self.released_ids) > 0 and (id in self.released_ids.keys()) and len(self.released_ids[id]) > 0:
            local_index = np.uint32(self.released_ids[id].pop())
            if len(self.released_ids[id]) <= 0:
                del self.released_ids[id]
        else:
            local_index = self.local_ids[lon_index, lat_index, depth_index, time_index]
            self.local_ids[lon_index, lat_index, depth_index, time_index] += 1
        id = np.int64(id)
        id = np.bitwise_or(np.left_shift(id, 32), np.int64(local_index))
        id = np.uint64(id)
        if self._track_id_index:
            self._map_id_totalindex[id] = self._total_ids
        self._total_ids += 1
        self._used_ids += 1
        return id

    def _release_id(self, spatiotemporal_id, local_id):
        """
        private function - releases a 64-bit ID by adding the 32-bit local ID to the spatial lon-lat-depth-time ID map
        :arg spatiotemporal_id: 32-bit spatial ID segment
        :arg local_id: 32-bit sequential ID segment
        """
        if not self._recover_ids:
            return
        if spatiotemporal_id not in self.released_ids.keys():
            self.released_ids[spatiotemporal_id] = []
        self.released_ids[spatiotemporal_id].append(local_id)
        self._used_ids -= 1


class GenerateID_Service(BaseIdGenerator):
    """
    Metaclass that manages IDs in a distributed MPI context via message passing and the MessageService in a background thread.
    ID generation- and release itself is done with one of the single-machine ID generators above.
    """
    _request_tag = 5
    _response_tag = 6

    def __init__(self, base_generator_obj):
        """
        GenerateID_Service - Constructor
        :arg base_generator_obj: the original single-machine ID generator class to be used for generation and release
        """
        super(GenerateID_Service, self).__init__()
        self._service_process = None
        self._serverrank = 0
        self._request_tag = 5
        self._response_tag = 6
        self._recover_ids = False
        self._use_subprocess = True

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()
            if mpi_size <= 1:
                self._use_subprocess = False
            else:
                self._serverrank = mpi_size-1
                if mpi_rank == self._serverrank:
                    self._service_process = Thread(target=executor, name="IdService", args=(base_generator_obj, self._request_tag, self._response_tag), daemon=True)
                    self._service_process.start()
                self._subscribe_()
        else:
            self._use_subprocess = False

        if not self._use_subprocess:
            self._service_process = base_generator_obj()

    def __del__(self):
        """
        GenerateID_Service - Destructor
        """
        self._abort_()

    def _subscribe_(self):
        """
        subscribes a client to the lead generator process
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            data_package = {}
            data_package["func_name"] = "thread_subscribe"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)

    def _abort_(self):
        """
        unsubscribes a client from ID generation and closes the generator thread
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            data_package = {}
            data_package["func_name"] = "thread_abort"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)

    def close(self):
        """
        Closing the ID generator, either releasing (if enabled) or destroying (otherwise) the managed IDs
        """
        self._abort_()

    def enable_ID_recovery(self):
        """
        this function enables ID recovery
        """
        self._recover_ids = True
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "enable_ID_recovery"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.enable_ID_recovery()

    def disable_ID_recovery(self):
        """
        this function disables ID recovery
        """
        self._recover_ids = False
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "disable_ID_recovery"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.disable_ID_recovery()

    def setTimeLine(self, min_time=0.0, max_time=1.0):
        """
        abstract function - setting min-max limits to the 'time' dimension of the ID
        :arg min_time: lowest time value used during the simulation. For forward simulation, this would be 't_0'.
        :arg max_time: highest time value used during the simulation. For forward simulation, this would be the runtime or 't_N'.
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                data_package = {}
                data_package["func_name"] = "setTimeLine"
                data_package["args"] = 2
                data_package["argv"] = [min_time, max_time]
                data_package["src_rank"] = mpi_rank
                mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.setTimeLine(min_time, max_time)

    def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
        """
        abstract function - setting min-max limits to the 'depth' dimension of the ID
        :arg min_depth: lowest depth value during the simulation. With depth being measured from the sea surface, this value would be the sea surface itself (`min_depth=0`).
        :arg max_depth: highest depth value during the simulation. With depth being measured from the sea surface positive downward, this value would be the deepest level of the model.
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                data_package = {}
                data_package["func_name"] = "setDepthLimits"
                data_package["args"] = 2
                data_package["argv"] = [min_depth, max_depth]
                data_package["src_rank"] = mpi_rank
                mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.setDepthLimits(min_depth, max_depth)

    def getID(self, lon, lat, depth, time):
        """
        abstract function - major function to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "getID"
            data_package["args"] = 4
            data_package["argv"] = [lon, lat, depth, time]
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)
            return int(data["result"])
        else:
            return self._service_process.getID(lon, lat, depth, time)

    def nextID(self, lon, lat, depth, time):
        """
        external renamed method to generate and/or retrieve the next ID
        :arg lon: longitude coordinate of the object
        :arg lat: latitude coordinate of the object
        :arg depth: depth coordinate of the object
        :arg time: time coordinate of the object
        :returns new ID
        """
        return self.getID(lon, lat, depth, time)

    def releaseID(self, id):
        """
        abstract function - releases an ID, either by just removing it from the ID list or by
                            recovering the ID, reusing it for the next 'getID(...)' call
        :arg id: ID to be released (and possibly recovered)
        """
        if not self._recover_ids:
            return
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "releaseID"
            data_package["args"] = 1
            data_package["argv"] = [id, ]
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.releaseID(id)

    def get_length(self):
        """
        function forward to
        :returns the number of generated IDs
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "get_length"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            return int(data["result"])
        else:
            return self._service_process.__len__()

    def get_total_length(self):
        """
        function forward to
        :returns the total number of generated IDs, incl. pre-generated IDs and excl. the ID recovery
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "get_total_length"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            return int(data["result"])
        else:
            return self._service_process.get_total_length()

    def get_usable_length(self):
        """
        function forward to
        :returns the usable number of generated IDs, excl. pre-generated IDs and considering the ID recovery
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "get_usable_length"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            return int(data["result"])
        else:
            return self._service_process.get_usable_length()

    def __len__(self):
        """
        :returns the number of generated IDs
        """
        return self.get_length()

    @property
    def total_length(self):
        """
        :returns the total number of generated IDs, incl. pre-generated IDs and excl. the ID recovery
        """
        return self.get_total_length()

    @property
    def usable_length(self):
        """
        :returns the usable number of generated IDs, excl. pre-generated IDs and considering the ID recovery
        """
        return self.get_usable_length()

    def preGenerateIDs(self, high_value):
        """
        abstract function - pre-allocating a range of IDs from 0 up to :arg high_value.
        :arg high_value: (u)int64 value of the highest pre-generated ID itself.
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                data_package = {}
                data_package["func_name"] = "preGenerateIDs"
                data_package["args"] = 1
                data_package["argv"] = [high_value, ]
                data_package["src_rank"] = mpi_rank
                mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.preGenerateIDs(high_value)

    def enable_id_index_tracking(self):
        """
        Enables the tracking of ID to index, so that the ID can reliably be associated to an index
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "enable_id_index_tracking"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.enable_id_index_tracking()

    def disable_id_index_tracking(self):
        """
        Disables the tracking of ID to index
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "disable_id_index_tracking"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
        else:
            self._service_process.disable_id_index_tracking()

    def map_id_to_index(self, input_id):
        """
        Maps an ID to its index, if ID tracking is enabled
        :arg input_id: ID to be associated with an index
        :returns list index
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "map_id_to_index"
            data_package["args"] = 1
            data_package["argv"] = [input_id, ]
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)
            return int(data["result"])
        else:
            return self._service_process.map_id_to_index(input_id)

    def is_tracking_id_index(self):
        """
        :returns if ID tracking is enabled or not
        """
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "is_tracking_id_index"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            return (True if (data["result"] or data["result"] > 0) else False)
        else:
            return self._service_process.is_tracking_id_index()
