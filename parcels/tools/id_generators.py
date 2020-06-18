import random   # could be python's random if parcels not active; can be parcel's random; can be numpy's random
# from numpy import random as nprandom
import numpy as np
import math

class IdGenerator:
    released_ids = []
    next_id = 0

    def __init__(self):
        self.released_ids = []
        self.next_id = np.uint64(0)

    def nextID(self):
        n = len(self.released_ids)
        if n == 0:
            result = self.next_id
            self.next_id += 1
            return np.uint64(result)
        else:
            result = self.released_ids.pop(n-1)
            return np.uint64(result)

    def releaseID(self, id):
        self.released_ids.append(id)

    def preGenerateIDs(self, high_value):
        if len(self.released_ids) > 0:
            self.released_ids.clear()
        #for i in range(0, high_value):
        #    self.released_ids.append(i)
        self.released_ids = [i for i in range(0, high_value)]
        self.next_id = high_value

    def permuteIDs(self):
        n = len(self.released_ids)
        indices = random.randint(0, n, 2*n)
        for index in indices:
            id = self.released_ids.pop(index)
            self.released_ids.append(id)

        #for iter in range(0, 2*n):
        #    index = random.randint(0, n)
        #    id = self.released_ids.pop(index)
        #    self.released_ids.append(id)

    def __len__(self):
        return self.next_id

class SpatioTemporalIdGenerator:
    """Generates 64-bit IDs"""
    timebounds  = np.zeros(2, dtype=np.float64)
    depthbounds = np.zeros(2, dtype=np.float32)
    local_ids = 0
    released_ids = {}

    def __init__(self):
        self.timebounds  = np.zeros(2, dtype=np.float64)
        self.depthbounds = np.zeros(2, dtype=np.float32)
        self.local_ids = np.zeros((360, 180, 128, 256), dtype=np.uint32)
        self.released_ids = {}  # 32-bit spatio-temporal index => []

    def setTimeLine(self, min_time=0.0, max_time=1.0):
        self.timebounds = np.array([min_time, max_time], dtype=np.float64)

    def setDepthLimits(self, min_dept=0.0, max_depth=1.0):
        self.depthbounds = np.array([min_dept, max_depth], dtype=np.float32)

    def getID(self, lon, lat, depth, time):
        if lon < -180.0 or lon > 180.0:
            vsgn = np.sign(lon)
            lon = np.fmod(np.fabs(lon), 180.0) * vsgn
        if lat < -90.0 or lat > 90.0:
            vsgn = np.sign(lat)
            lat = np.fmod(np.fabs(lat), 180.0) * vsgn
        if depth is None:
            depth = self.depthbounds[0]
        if time is None:
            time = self.timebounds[0]
        # lon_discrete = np.float32(np.int32(lon))
        lon_discrete = np.int32(lon)
        # lat_discrete = np.float32(np.int32(lat))
        lat_discrete = np.int32(lat)
        depth_discrete = (depth-self.depthbounds[0])/(self.depthbounds[1]-self.depthbounds[0])
        # depth_discrete = np.float32(np.int32(128.0*depth_discrete))
        depth_discrete = np.int32(127.0 * depth_discrete)
        time_discrete = (time-self.timebounds[0])/(self.timebounds[1]-self.timebounds[0])
        # time_discrete = np.float32(np.int32(256.0*time_discrete))
        time_discrete = np.int32(255.0 * time_discrete)
        lon_index   = np.uint32(np.int32(lon_discrete)+180)
        lat_index   = np.uint32(np.int32(lat_discrete)+90)
        depth_index = np.uint32(np.int32(depth_discrete))
        time_index  = np.uint32(np.int32(time_discrete))
        local_index = -1
        # id = np.bitwise_or(np.bitwise_or(np.bitwise_or(np.left_shift(lon_index, 23), np.left_shift(lat_index, 15)), np.left_shift(depth_index, 8)), time)
        id = np.left_shift(lon_index, 23) + np.left_shift(lat_index, 15) + np.left_shift(depth_index, 8) + time
        if len(self.released_ids)>0 and (id in self.released_ids.keys()) and len(self.released_ids[id])>0:
            # mlist = self.released_ids[id]
            local_index = np.uint32(self.released_ids[id].pop())
            if len(self.released_ids[id])<= 0:
                del self.released_ids[id]
        else:
            local_index = self.local_ids[lon_index, lat_index, depth_index, time_index]
            self.local_ids[lon_index, lat_index, depth_index, time_index] += 1
        id = np.int64(id)
        id = np.bitwise_or(np.left_shift(id, 32), np.int64(local_index))
        #id = np.left_shift(id, 32) + np.uint64(local_index)
        id = np.uint64(id)
        return id

    def nextID(self, lon, lat, depth, time):
        return self.getID(lon, lat, depth, time)

    def releaseID(self, id):
        full_bits = np.uint32(4294967295)
        nil_bits  = np.int32(0)
        spatiotemporal_id = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(full_bits), 32), np.int64(nil_bits)), np.int64(id))
        spatiotemporal_id = np.uint32(np.right_shift(spatiotemporal_id, 32))
        local_id          = np.bitwise_and(np.bitwise_or(np.left_shift(np.int64(nil_bits), 32), np.int64(full_bits)), np.int64(id))
        local_id          = np.uint32(local_id)
        if spatiotemporal_id not in self.released_ids.keys():
            self.released_ids[spatiotemporal_id] = []
        self.released_ids[spatiotemporal_id].append(local_id)

    def __len__(self):
        return np.sum(self.local_ids)+sum([len(entity) for entity in self.released_ids])

