import random   # could be python's random if parcels not active; can be parcel's random; can be numpy's random
# from numpy import random as nprandom
import numpy as np
from os import path
from os import remove
from time import sleep
import math
from parcels.tools import get_cache_dir
import pickle

try:
    from mpi4py import MPI
except:
    MPI = None

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
    local_ids = None
    released_ids = {}
    _total_ids = 0

    def __init__(self):
        self.timebounds  = np.zeros(2, dtype=np.float64)
        self.depthbounds = np.zeros(2, dtype=np.float32)
        self.local_ids = np.zeros((360, 180, 128, 256), dtype=np.uint32)
        # self.local_ids = None
        # if MPI:
        #     mpi_comm = MPI.COMM_WORLD
        #     mpi_rank = mpi_comm.Get_rank()
        #     if mpi_rank == 0:
        #         self.local_ids = np.zeros((360, 180, 128, 256), dtype=np.uint32)
        # else:
        #     self.local_ids = np.zeros((360, 180, 128, 256), dtype=np.uint32)
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                access_flag_file = path.join( get_cache_dir(), 'id_access' )
                occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
                idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
                with open(access_flag_file, 'wb') as f_access:
                    f_access.write(bytearray([True,]))
                    with open(idreleases_file, 'wb') as f_idrel:
                        pickle.dump(self.released_ids, f_idrel)
                    # self.local_ids.tofile(occupancy_file)
                    np.save(occupancy_file, self.local_ids)
                remove(access_flag_file)
        self.released_ids = {}  # 32-bit spatio-temporal index => []
        self._total_ids = 0

    def __del__(self):
        # occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
        # idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
        # if path.exists(occupancy_file):
        #     remove(occupancy_file)
        # if path.exists(idreleases_file):
        #     remove(idreleases_file)
        pass

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
            lat = np.fmod(np.fabs(lat), 90.0) * vsgn
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
        if MPI:
            id = self._distribute_next_id_by_file(lon_index, lat_index, depth_index, time_index)
        else:
            id = self._get_next_id(lon_index, lat_index, depth_index, time_index)
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
        if MPI:
            self._gather_released_ids_by_file(spatiotemporal_id, local_id)
        else:
            self._release_id(spatiotemporal_id, local_id)

    def __len__(self):
        if MPI:
            # return self._length_()
            return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])
        else:
            return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])

    @property
    def total_length(self):
        if MPI:
            # return self._total_length_()
            return self._total_ids
        else:
            return self._total_ids

    def _distribute_next_id(self, lon_index, lat_index, depth_index, time_index):
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        snd_requested_id = np.zeros((mpi_size, 1), dtype=np.byte)
        rcv_requested_id = np.zeros((mpi_size, 1), dtype=np.byte)
        snd_requested_add = np.zeros((mpi_size, 4), dtype=np.uint32)
        rcv_requested_add = np.zeros((mpi_size, 4), dtype=np.uint32)
        return_id = np.zeros(mpi_size, dtype=np.uint64)
        return_id.fill(np.iinfo(np.uint64).max)
        snd_requested_id[mpi_rank] = 1
        snd_requested_add[mpi_rank, :] = np.array([lon_index, lat_index, depth_index, time_index], dtype=np.uint32)
        mpi_comm.Reduce(snd_requested_id, rcv_requested_id, op=MPI.MAX, root=0)
        # rcv_requested_id = mpi_comm.reduce(snd_requested_id, op=MPI.MAX, root=0)
        mpi_comm.Reduce(snd_requested_add, rcv_requested_add, op=MPI.MAX, root=0)
        if mpi_rank == 0:
            for i in range(mpi_size):
                if rcv_requested_id[i] > 0:
                    return_id[i] = self._get_next_id(rcv_requested_add[i, 0], rcv_requested_add[i, 1], rcv_requested_add[i, 2], rcv_requested_add[i, 3])
        # mpi_comm.Bcast(return_id, root=0)
        return_id = mpi_comm.bcast(return_id, root=0)
        return return_id[mpi_rank]

    def _distribute_next_id_by_file(self, lon_index, lat_index, depth_index, time_index):
        # mpi_comm = MPI.COMM_WORLD
        # mpi_rank = mpi_comm.Get_rank()
        # mpi_size = mpi_comm.Get_size()

        return_id = None
        access_flag_file = path.join( get_cache_dir(), 'id_access' )
        occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
        idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
        while path.exists(access_flag_file):
            sleep(0.1)
        with open(access_flag_file, 'wb') as f_access:
            f_access.write(bytearray([True,]))
            # self.local_ids = np.fromfile(occupancy_file, dtype=np.uint32)
            self.local_ids = np.load(occupancy_file)
            with open(idreleases_file, 'rb') as f_idrel:
                self.released_ids = pickle.load( f_idrel )
            return_id = self._get_next_id(lon_index, lat_index, depth_index, time_index)
            with open(idreleases_file, 'wb') as f_idrel:
                pickle.dump(self.released_ids, f_idrel)
            # self.local_ids.tofile(occupancy_file)
            np.save(occupancy_file, self.local_ids)
        remove(access_flag_file)

        return return_id

    def _gather_released_ids(self, spatiotemporal_id, local_id):
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        snd_release_id = np.zeros(mpi_size, dtype=np.byte)
        rcv_release_id = np.zeros(mpi_size, dtype=np.byte)
        snd_release_add = np.zeros((mpi_size, 2), dtype=np.uint32)
        rcv_release_add = np.zeros((mpi_size, 2), dtype=np.uint32)
        snd_release_id[mpi_rank] = 1
        snd_release_add[mpi_rank, :] = np.array([spatiotemporal_id, local_id], dtype=np.uint32)
        mpi_comm.Reduce(snd_release_id, rcv_release_id, op=MPI.MAX, root=0)
        mpi_comm.Reduce(snd_release_add, rcv_release_add, op=MPI.MAX, root=0)
        if mpi_rank == 0:
            for i in range(mpi_size):
                if rcv_release_id[i] > 0:
                    self._release_id(rcv_release_add[i, 0], rcv_release_add[i, 1])

    def _gather_released_ids_by_file(self, spatiotemporal_id, local_id):

        return_id = None
        access_flag_file = path.join( get_cache_dir(), 'id_access' )
        occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
        idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
        while path.exists(access_flag_file):
            sleep(0.1)
        with open(access_flag_file, 'wb') as f_access:
            f_access.write(bytearray([True,]))
            # self.local_ids = np.fromfile(occupancy_file, dtype=np.uint32)
            self.local_ids = np.load(occupancy_file)
            with open(idreleases_file, 'rb') as f_idrel:
                self.released_ids = pickle.load( f_idrel )
            self._release_id(spatiotemporal_id, local_id)
            with open(idreleases_file, 'wb') as f_idrel:
                pickle.dump(self.released_ids, f_idrel)
            # self.local_ids.tofile(occupancy_file)
            np.save(occupancy_file, self.local_ids)
        remove(access_flag_file)

        return return_id

    def _get_next_id(self, lon_index, lat_index, depth_index, time_index):
        local_index = -1
        # id = np.bitwise_or(np.bitwise_or(np.bitwise_or(np.left_shift(lon_index, 23), np.left_shift(lat_index, 15)), np.left_shift(depth_index, 8)), time)
        id = np.left_shift(lon_index, 23) + np.left_shift(lat_index, 15) + np.left_shift(depth_index, 8) + time_index
        # id = np.left_shift(lon_index, 23) + np.left_shift(lat_index, 15) + np.left_shift(depth_index, 8) + time
        print("requtested indices: ({}, {}, {}, {})".format(lon_index, lat_index, depth_index, time_index))
        print("spatial id: {}".format(id))
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
        self._total_ids += 1
        return id

    def _release_id(self, spatiotemporal_id, local_id):
        if spatiotemporal_id not in self.released_ids.keys():
            self.released_ids[spatiotemporal_id] = []
        self.released_ids[spatiotemporal_id].append(local_id)

    def _length_(self):
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        alength = 0
        if mpi_rank == 0:
            alength = np.sum(self.local_ids.astype(dtype=np.uint64))+np.uint64(sum([len(entity) for entity in self.released_ids]))
        return mpi_comm.bcast(alength, root=0)
        #alength = np.sum(self.local_ids.astype(dtype=np.uint64))+np.uint64(sum([len(entity) for entity in self.released_ids]))


    def _total_length_(self):
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        alength = 0
        if mpi_rank == 0:
            alength = self._total_ids
        return mpi_comm.bcast(alength, root=0)

