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

        # if MPI:
        #     mpi_comm = MPI.COMM_WORLD
        #     mpi_rank = mpi_comm.Get_rank()
        #     if mpi_rank == 0:
        #         access_flag_file = path.join( get_cache_dir(), 'id_access' )
        #         occupancy_file = path.join( get_cache_dir(), 'id_occupancy.npy')
        #         idreleases_file = path.join( get_cache_dir(), 'id_releases.pkl' )
        #         with open(access_flag_file, 'wb') as f_access:
        #             f_access.write(bytearray([True,]))
        #             with open(idreleases_file, 'wb') as f_idrel:
        #                 pickle.dump(self.released_ids, f_idrel)
        #             # self.local_ids.tofile(occupancy_file)
        #             np.save(occupancy_file, self.local_ids)
        #         remove(access_flag_file)
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

    def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
        self.depthbounds = np.array([min_depth, max_depth], dtype=np.float32)

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
        self._release_id(spatiotemporal_id, local_id)

    def __len__(self):
        return np.sum(self.local_ids) + sum([len(entity) for entity in self.released_ids])

    def get_length(self):
        return self.__len__()

    @property
    def total_length(self):
        return self._total_ids

    def get_total_length(self):
        return self._total_ids

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


#from multiprocessing import Process, Pipe
#from threading import Thread
from multiprocessing import Process
# from multiprocessing.connection import Connection
from .message_service import mpi_execute_requested_messages as executor
from sys import stdout
from os import getpid
from parcels.tools import logger

class GenerateID_Service(object):
    _request_tag = 5
    _response_tag = 6

    def __init__(self, base_generator_obj):
        self._service_process = None
        self._worker_node = None
        self._service_node = None
        self._cr_sequence = '#e#'
        self._serverrank = 0
        self._use_subprocess = True

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()
            if mpi_size <= 1:
                self._use_subprocess = False
            self._serverrank = mpi_size-1
            # self._worker_node, self._service_node = Pipe()
            # service_bundle = mpi_comm.gather(self._service_node, root=0)
            if mpi_rank == self._serverrank:
                # self._service_process = Process(target=executor, name="IdService", args=(service_bundle, base_generator_obj), daemon=True)
                # self._service_process.start()
                print("Starting ID service process")
                self._service_process = Process(target=executor, name="IdService", args=(base_generator_obj, self._request_tag, self._response_tag), daemon=True)
                self._service_process.start()
                # executor(base_generator_obj, self._request_tag, self._response_tag)
            # mpi_comm.Barrier()
            logger.info("worker - MPI rank: {} pid: {}".format(mpi_rank, getpid()))
        else:
            self._use_subprocess = False

        if not self._use_subprocess:
            self._service_process = base_generator_obj()


    def __del__(self):
        if self._worker_node is not None:
            self._worker_node.close()
        if self._service_node is not None:
            self._service_node.close()
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if self._service_process is not None:
                self._service_process.join()

    def setTimeLine(self, min_time=0.0, max_time=1.0):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                data_package = {}
                data_package["func_name"] = "setTimeLine"
                data_package["args"] = 2
                data_package["argv"] = [min_time, max_time]
                data_package["src_rank"] = mpi_rank
                # msg = mpi_comm.isend(data_package, dest=self._serverrank, tag=5)
                # msg.wait()
                mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
                # msg = mpi_comm.irecv(source=self._serverrank, tag=6)
                # data = msg.wait()
                # print(data)
                # stdout.write("{}\n".format(data))

            # if mpi_rank == 0:
            #     data_package = {}
            #     data_package["func_name"] = "setTimeLine"
            #     data_package["args"] = 2
            #     data_package["argv"] = [min_time, max_time]
            #     self._worker_node.send(data_package["func_name"])
            #     self._worker_node.send(data_package["args"])
            #     self._worker_node.send(data_package["argv"])
            #     self._worker_node.send(self._cr_sequence)
        else:
            self._service_process.setTimeLine(min_time, max_time)

    def setDepthLimits(self, min_depth=0.0, max_depth=1.0):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            if mpi_rank == 0:
                data_package = {}
                data_package["func_name"] = "setDepthLimits"
                data_package["args"] = 2
                data_package["argv"] = [min_depth, max_depth]
                data_package["src_rank"] = mpi_rank
                # msg = mpi_comm.isend(data_package, dest=self._serverrank, tag=5)
                # msg.wait()
                mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
                # msg = mpi_comm.irecv(source=self._serverrank, tag=6)
                # data = msg.wait()
                # print(data)
                # stdout.write("{}\n".format(data))

            # if mpi_rank == 0:
            #     data_package = {}
            #     data_package["func_name"] = "setDepthLimits"
            #     data_package["args"] = 2
            #     data_package["argv"] = [min_depth, max_depth]
            #     self._worker_node.send(data_package["func_name"])
            #     self._worker_node.send(data_package["args"])
            #     self._worker_node.send(data_package["argv"])
            #     self._worker_node.send(self._cr_sequence)
        else:
            self._service_process.setDepthLimits(min_depth, max_depth)

    def getID(self, lon, lat, depth, time):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "getID"
            data_package["args"] = 4
            data_package["argv"] = [lon, lat, depth, time]
            data_package["src_rank"] = mpi_rank
            # msg = mpi_comm.isend(data_package, dest=self._serverrank, tag=5)
            # msg.wait()
            # logger.info("package sending: {}".format(data_package))
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            # msg = mpi_comm.irecv(source=self._serverrank, tag=6)
            # data = msg.wait()
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)
            # print(data)
            # stdout.write("{}\n".format(data))
            # logger.info("recv: {}".format(data))
            return int(data["result"])

            # data_package = {}
            # data_package["func_name"] = "getID"
            # data_package["args"] = 4
            # data_package["argv"] = [lon, lat, depth, time]
            # self._worker_node.send(data_package["func_name"])
            # self._worker_node.send(data_package["args"])
            # self._worker_node.send(data_package["argv"])
            # self._worker_node.send(self._cr_sequence)

            # self._worker_node.poll(None)
            # result = self._worker_node.recv()
            # assert isinstance(result, np.uint64)
            # assert self._worker_node.recv() == self._cr_sequence
            # return result
        else:
            return self._service_process.getID(lon, lat, depth, time)

    def nextID(self, lon, lat, depth, time):
        return self.getID(lon, lat, depth, time)

    def releaseID(self, id):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            data_package = {}
            data_package["func_name"] = "releaseID"
            data_package["args"] = 1
            data_package["argv"] = [id, ]
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            # self._worker_node.send(data_package["func_name"])
            # self._worker_node.send(data_package["args"])
            # self._worker_node.send(data_package["argv"])
            # self._worker_node.send(self._cr_sequence)
        else:
            self._service_process.releaseID(id)

    def __len__(self):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            # data_package = {}
            # data_package["func_name"] = "__len__"
            # data_package["args"] = 0
            # self._worker_node.send(data_package["func_name"])
            # self._worker_node.send(data_package["args"])
            # self._worker_node.send(self._cr_sequence)

            data_package = {}
            data_package["func_name"] = "get_length"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            # self._worker_node.poll(None)
            # result = self._worker_node.recv()
            # # assert isinstance(result, np.uint32)
            # assert self._worker_node.recv() == self._cr_sequence
            # logger.info("recv: {}".format(data))
            return int(data["result"])
        else:
            return self._service_process.__len__()

    @property
    def total_length(self):
        if MPI and self._use_subprocess:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()

            # data_package = {}
            # data_package["func_name"] = "get_total_length"
            # data_package["args"] = 0
            # self._worker_node.send(data_package["func_name"])
            # self._worker_node.send(data_package["args"])
            # self._worker_node.send(self._cr_sequence)

            data_package = {}
            data_package["func_name"] = "get_total_length"
            data_package["args"] = 0
            data_package["src_rank"] = mpi_rank
            mpi_comm.send(data_package, dest=self._serverrank, tag=self._request_tag)
            data = mpi_comm.recv(source=self._serverrank, tag=self._response_tag)

            # self._worker_node.poll(None)
            # result = self._worker_node.recv()
            # # assert isinstance(result, np.uint64)
            # assert self._worker_node.recv() == self._cr_sequence
            # logger.info("recv: {}".format(data))
            return int(data["result"])
        else:
            return self._service_process.total_length


