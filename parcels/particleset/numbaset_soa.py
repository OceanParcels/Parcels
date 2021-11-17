import numba as nb
import numpy as np


from numba.experimental import jitclass


class BaseParticle():
    def __init__(self):
        pass

class NumbaCollectionSOA():
    def __init__(self, n_particles, pclass=0):
#         self.pclass = pclass
#         self.n_particles = n_particles
        #self.n_particles = n_particles
        return
        # Spec has to be numba types.
        #for name, nb_type_name in spec.items():
            #if nb_type_name == "nb.float64":
            #    nb_type = nb.float64
            #print(name, nb_type)
            #setattr(self, name, np.empty(n_particles, nb_type))

    def __getitem__(self, i):
        pass

    def test(self):
        return getattr(self, "lat")

def create_numba_collection(n_particles, particle):
    dual_spec = {}
    for v in particle.getPType().variables:
        if v.dtype == np.float64:
            numba_type = nb.float64
        elif v.dtype == np.float32:
            numba_type = nb.float32
        elif v.dtype == np.int32:
            numba_type = nb.int32
        elif v.dtype == np.int64:
            numba_type = nb.int64
        else:
            raise ValueError(f"Cannot parse type {v.type}")
        dual_spec[v.name] = (numba_type, v.dtype)

    spec = {key: value[0][:] for key, value in dual_spec.items()}
    nc = jitclass(NumbaCollectionSOA, spec=spec)
    spec = {key: value[0] for key, value in dual_spec.items()}
    pc = jitclass(BaseParticle, spec=spec)
    pset = nc(n_particles)
    for name, sp in dual_spec.items():
        np_class = sp[1]
        setattr(pset, name, np.empty(n_particles, dtype=np_class))
    return pset
