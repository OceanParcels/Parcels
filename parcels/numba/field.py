import numba as nb
import numpy as np
from numba.types import List

from numba import njit
from numba.experimental import jitclass
from numba.core.types.containers import Tuple
from numba.core.typing.asnumbatype import as_numba_type
from parcels import AdvectionRK4


STATUS_OK = 0
STATUS_OOB = 1
STATUS_CHUNK_MISSING = 2

NumbaAdvection = njit(AdvectionRK4)

# class NumbaChunk():
#     def __init__(data, ):

# @jitclass(spec=[('lat', Tuple),
#                 ('long', Tuple),
#                 ('depth', Tuple)])
# class GridSpec():
#     def __init__(self, lat, lon, depth):
#         self.lat = lat
#         self.lon = lon
#         self.depth = depth
# 
#     def __len__(self):
#         return self.lat[2]*self.lon[2]*self.depth[2]
# 
#     def particle_to_cell(self, p):
#         i_lat = int(self.lat[2]*(p.lat-self.lat[0])/self.lat[1])
#         i_lon = int(self.lon[2]*(p.lon-self.lon[0])/self.lon[1])
#         i_depth = int(self.depth[2]*(p.depth-self.depth[0])/self.depth[1])
#         i_cell = i_lat+i_lon*self.lat[2]+i_depth*self.lat[2]*self.lon[2]
#         return i_cell


# @jitclass(spec)
# class NumbaUV():
#     def __init__(self, data, grid):
#         self.data = data
#         self.grid = grid
# 
#     def __getitem__(self, param):
#         lon = param[1]
#         lat = param[2]
#         p = param[3]
#         


class Particle():
    def __init__(self, dt):
        self.lat = np.random.rand()
        self.lon = np.random.rand()
        self.depth = np.random.rand()
        self.dt = dt
        self.status = STATUS_OK
        self.request_chunk = -1

    def __str__(self):
        return f"{self.status}, ({self.lat}, {self.lon})"


spec = [
    ('dt', nb.float64),
    ('lon', nb.float64),
    ('lat', nb.float64),
    ('depth', nb.float64),
    ('status', nb.int32),
    ('request_chunk', nb.int32),
]
NumbaParticle = jitclass(Particle, spec=spec)


def create_chunks(grid, pset, func):
#     cells = np.unique([grid.particle_to_chunk(p) for p in pset])
#     cell_used = np.zeros(grid.lat.n*grid.lon.n, dtype=np.bool)
#     cell_used[cells] = True
    cell_used = np.ones(grid.lat.n*grid.lon.n, dtype=np.bool)
#     print(cells)

    data = nb.typed.List()
    
#     data = []
    i_chunk = 0
    for i_lon_chunk in range(grid.lon.nchunk):
        for i_lat_chunk in range(grid.lat.nchunk):
            if not cell_used[i_chunk]:
                data.append(np.zeros((0, 0)))
            else:
#                 print(i_chunk, grid.lat.chunk_spec(i_lat_chunk))
                lat_vals = np.linspace(*grid.lat.chunk_spec(i_lat_chunk), endpoint=False)
                lon_vals = np.linspace(*grid.lon.chunk_spec(i_lon_chunk), endpoint=False)
                lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals)
                field_vals = func(lat_grid, lon_grid)
                data.append(field_vals)
            i_chunk += 1
    return data


@jitclass(spec=[
    ("start", nb.float64),
    ("stop", nb.float64),
    ("n", nb.int32),
    ("nchunk", nb.int32),
    ("chunk_size", nb.int32),
    ("dx", nb.float64)]
)
class GridSpec1D():
    def __init__(self, spec):
        self.start = spec[0]
        self.stop = spec[1]
        self.n = spec[2]
        self.nchunk = spec[3]
        self.chunk_size = self.n//self.nchunk
        self.dx = (self.stop-self.start)/(self.n-1)

    def chunk_spec(self, ichunk):
        return self.start + self.chunk_size*self.dx*ichunk, self.start + self.chunk_size*self.dx*(ichunk+1), self.chunk_size


@jitclass(spec=[
    ("lat", as_numba_type(GridSpec1D)),
    ("lon", as_numba_type(GridSpec1D)),
])
class GridSpec():
    def __init__(self, lat, lon):
        self.lat = GridSpec1D(lat)
        self.lon = GridSpec1D(lon)
#         self.depth = GridSpec1D(depth)

    def __len__(self):
        return self.lat.n*self.lon.n

    def particle_to_cell(self, p):
        i_lat = int((p.lat-self.lat.start)/self.lat.dx)
        i_lon = int((p.lon-self.lon.start)/self.lon.dx)
#         i_depth = int((p.depth-self.depth.start)/self.depth.dx)
        i_cell = i_lat+i_lon*self.lat.n
        return i_cell

    def particle_to_chunk(self, p):
        i_lat = int((p.lat-self.lat.start)/self.lat.dx)
        i_lon = int((p.lon-self.lon.start)/self.lon.dx)
        i_chunk_lat = i_lat//self.lat.nchunk
        i_chunk_lon = i_lon//self.lon.nchunk
        return i_chunk_lat+i_chunk_lon*self.lat.nchunk

    def get_i(self, lat, lon):
        i_lat = int((lat-self.lat.start)/self.lat.dx)
        i_lon = int((lon-self.lon.start)/self.lon.dx)
#         i_depth = int((depth-self.depth.start)/self.depth.dx)
        return (i_lat, i_lon)

    def get_point_id(self, i_lat, i_lon):
        i_chunk_lat = i_lat//self.lat.chunk_size
        i_chunk_lon = i_lon//self.lon.chunk_size
        i_rel_lat = i_lat - i_chunk_lat*self.lat.chunk_size
        i_rel_lon = i_lon - i_chunk_lon*self.lon.chunk_size
        return i_chunk_lat+i_chunk_lon*self.lat.nchunk, i_rel_lat, i_rel_lon

    def distance(self, lat, lon, i_lat, i_lon):
        d_lat = lat - (self.lat.start+i_lat*self.lat.dx)
        d_lon = lon - (self.lon.start+i_lon*self.lon.dx)
#         d_depth = depth - (self.depth.start+i_depth*self.depth.dx)
        return np.sqrt(d_lat*d_lat+d_lon*d_lon)

    def interpolate(self, lat, lon, field):
        i_lat, i_lon = self.get_i(lat, lon)
        cur_sum = 0.0
        tot_weight = 0.0
        for x_lat in range(2):
            cur_lat = x_lat + i_lat
            for x_lon in range(2):
                cur_lon = x_lon + i_lon
                if cur_lat < 0 or cur_lat >= self.lat.n:
                    return STATUS_OOB, 0.0

                if cur_lon < 0 or cur_lon >= self.lon.n:
                    return STATUS_OOB, 0.0

                dist = self.distance(lat, lon, cur_lat, cur_lon)
                chunk_id, i_rel_lat, i_rel_lon = self.get_point_id(cur_lat, cur_lon)

                if field[chunk_id].size == 0:
                    return ((chunk_id << 2) | STATUS_CHUNK_MISSING), 0.0
#                 print(cur_lat, cur_lon)
#                 print(chunk_id, i_rel_lat, i_rel_lon)
#                 print()
                field_val = field[chunk_id][i_rel_lon, i_rel_lat]
#                 print(dist)
                if dist < 1e-7:
                    dist = 1e-7
                cur_sum += field_val/dist
                tot_weight += 1/dist
#                 print(field_val, dist)
#         print(cur_sum, tot_weight)
#         print()
        return STATUS_OK, cur_sum/tot_weight


@jitclass(spec=[
    ("fieldU", nb.types.ListType(nb.types.Array(nb.float64, 2, 'C'))),
    ("fieldV", nb.types.ListType(nb.types.Array(nb.float64, 2, 'C'))),
    ("grid", as_numba_type(GridSpec)),
])
class FieldUV():
    def __init__(self, fieldU, fieldV, grid):
        self.fieldU = fieldU
        self.fieldV = fieldV
        self.grid = grid

    def __getitem__(self, param):
        lat = param[2]
        lon = param[3]
        p = param[4]
        status, valU = self.grid.interpolate(lat, lon, self.fieldU)
        if status != STATUS_OK:
            p.status = status
            return valU, valU
        status, valV = self.grid.interpolate(lat, lon, self.fieldV)
        if status != STATUS_OK:
            p.status = status
            return valU, valU
        return valU, valV


@jitclass(spec=[
    ("grid", as_numba_type(GridSpec)),
    ("UV", as_numba_type(FieldUV))
])
class FieldSet():
    def __init__(self, grid, fieldU, fieldV):
        self.grid = grid
        self.UV = FieldUV(fieldU, fieldV, grid)


@njit
def particle_inner_loop(pset, fieldset, n_iteration=1000):
    for p in pset:
        for _ in range(n_iteration):
            NumbaAdvection(p, fieldset, 0)
