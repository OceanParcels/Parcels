import numba as nb
import numpy as np
from numba.experimental import jitclass
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.types import BaseTuple
from numba.core.typing.typeof import typeof
from numba.core.decorators import generated_jit
spec = [
    ('dt', nb.float64[:]),
    ('lon', nb.float64[:]),
    ('lat', nb.float64[:]),
    ('depth', nb.float64[:]),
]


class PSet():
    def __init__(self, dt, lon, lat, depth):
        self.dt = dt
        self.lon = lon
        self.lat = lat
        self.depth = depth

    def __getitem__(self, key):
        return NumbaParticle(self.dt[key], self.lon[key], self.lat[key], self.depth[key])

    def __setitem__(self, key, value):
        self.dt[key] = value.dt
        self.lon[key] = value.lon
        self.lat[key] = value.lat
        self.depth[key] = value.depth

    def __len__(self):
        return len(self.dt)

    def set_particle(self, key, particle):
        self.dt[key] = particle.dt
        self.lon[key] = particle.lon
        self.lat[key] = particle.lat
        self.depth[key] = particle.depth


NumbaPSet = jitclass(PSet, spec=spec)
spec = [
    ('dt', nb.float64),
    ('lon', nb.float64),
    ('lat', nb.float64),
    ('depth', nb.float64),
]


@jitclass(spec=spec)
class NumbaParticle():
    def __init__(self, dt, lon, lat, depth):
        self.dt = dt
        self.lon = lon
        self.lat = lat
        self.depth = depth


class PythonUV():
    def __init__(self):
        pass

    def __getitem__(self, param):
        return param[2], param[1]
#         return compute_UV(param)
#         try:
#             print(dir(typeof(param)))
#             print(param[0])
#         except:
#             print(param)
#         print(param, is_tuple(param))
#         if is_tuple(param):
#             particle = param[-1]
#         else:
#             particle = param
#         return particle.lat, particle.lon
#         return (2, 3)
#         if isinstance(param, BaseTuple):
#             print(param)
#         try:
#         if isinstance(param, tuple):
#             len(param)
#             return param[-1].lat, param[-1].lon
#         except Exception:
#             return param.lat, param.lon
#         return np.random.rand(2)


NumbaUV = jitclass(PythonUV, spec={})
# class NumbaUV():
#     def __init__(self):
#         pass
# 
#     def __getitem__(self, _):
#         return np.random.rand(2)

#         return (1.0, 2.0)


@jitclass(spec={"UV": as_numba_type(NumbaUV)})
class NumbaFieldset():
    def __init__(self):
        self.UV = NumbaUV()

#     def UV(self, _):
#         return (1.0, 2.0)

# @nb.extending.overload(len)
# def len_extension(instance):
#     print(instance)
#     if isinstance(instance, NumbaPSet):
#         print("hi")
#         return instance.__len__()


@nb.njit
def fast_execute(kernel, fieldset, particleset, n_time=1000):
#     print(n_time*particleset.__len__())
    for i in range(particleset.__len__()):
        p = particleset[i]
        for _ in range(n_time):
            kernel(p, fieldset, 0)
        particleset[i] = p


class PythonFieldset():
    def __init__(self):
        self.UV = PythonUV()


def python_execute(kernel, fieldset, particleset, n_time=1000):
    for i in range(particleset.__len__()):
        p = particleset[i]
        for _ in range(n_time):
            kernel(p, fieldset, 0)
        particleset[i] = p


# @generated_jit(nopython=True)
# def compute_UV(x):
#     if isinstance(x, BaseTuple):
#         return lambda x: x[-1].lat, x[-1].lon
#     return lambda x: x.lat, x.lon

