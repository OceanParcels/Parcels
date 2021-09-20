from numba import from_dtype
from numba.experimental import jitclass
from numba.typed import List
import numpy as np


class Particle():
    def __init__(self):
        pass


def create_pset_from_ptype(ptype, n_particles, pid=None, **kwargs):
    if pid is None:
        pid = np.arange(n_particles)
    kwargs["id"] = pid
    spec = {}
    for v in ptype.variables:
        spec[v.name] = from_dtype(v.dtype)
        print(v.name, v.initial)
    NumbaParticle = jitclass(Particle, spec=spec)
    pset = List()
    for i in range(n_particles):
        p = NumbaParticle()
        for v in ptype.variables:
            if v.name in kwargs:
                setattr(p, v.name, kwargs[v.name][i])
            else:
                setattr(p, v.name, v.initial)
        pset.append(p)
    return pset


def convert_pset_to_tlist(pset):
    ptype = pset._collection.ptype
    spec = {}
    for v in ptype.variables:
        if v.name in ['xi', 'yi', 'zi', 'ti']:
            spec[v.name] = from_dtype(v.dtype)[:]
        else:
            spec[v.name] = from_dtype(v.dtype)
    NumbaParticle = jitclass(Particle, spec=spec)
    numba_pset = List()
    for i in range(len(pset)):
        p = NumbaParticle()
        for v in ptype.variables:
            setattr(p, v.name, getattr(pset.collection, v.name)[i])
        numba_pset.append(p)
    return numba_pset


def convert_tlist_to_pset(tlist, pset):
    ptype = pset._collection.ptype
    for i in range(len(pset)):
        p = tlist[i]
        for v in ptype.variables:
            v_array = getattr(pset.collection, v.name)
            v_array[i] = getattr(p, v.name)
