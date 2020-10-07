import numpy as np

from .collections import ParticleCollection
from .iterators import BaseParticleAccessor
from .iterators import BaseParticleCollectionIterator
# from parcels.particle import ScipyParticle

"""
Author: Dr. Christian Kehl
github relation: #913 (particleset_class_hierarchy)
purpose: defines all the specific functions for a ParticleCollection, ParticleAccessor, ParticleSet etc. that relates
         to a structure-of-array (SoA) data arrangement.
"""


class ParticleCollectionSOA(ParticleCollection):

    def __init__(self):
        super(ParticleCollection, self).__init__()

    def __iter__(self):
        """Returns an Iterator that allows for forward iteration over the
        elements in the ParticleCollection (e.g. `for p in pset:`).
        """
        return ParticleCollectionIteratorSOA(self)

    def __reversed__(self):
        """Returns an Iterator that allows for backwards iteration over
        the elements in the ParticleCollection (e.g.
        `for p in reversed(pset):`).
        """
        return ParticleCollectionIteratorSOA(self, True)


class ParticleAccessorSOA(BaseParticleAccessor):
    def __init__(self, pcoll):
        super().__init__(pcoll)

    def __getattr__(self, name):
        return self.pcoll.particle_data[name][self._index]

    def __setattr__(self, name, value):
        if name in ['pcoll', '_index']:
            object.__setattr__(self, name, value)
        else:
            # avoid recursion
            self.pcoll.particle_data[name][self._index] = value

    def __repr__(self):
        time_string = 'not_yet_set' if self.time is None or np.isnan(self.time) else "{:f}".format(self.time)
        str = "P[%d](lon=%f, lat=%f, depth=%f, " % (self.id, self.lon, self.lat, self.depth)
        for var in self.pcoll.ptype.variables:
            if var.to_write is not False and var.name not in ['id', 'lon', 'lat', 'depth', 'time']:
                str += "%s=%f, " % (var.name, getattr(self, var.name))
        return str + "time=%s)" % time_string


class ParticleCollectionIteratorSOA(BaseParticleCollectionIterator):
    def __init__(self, pcoll, reverse=False, subset=None):
        # super().__init__(pcoll)  # Do not actually need this

        if subset is not None:
            if type(subset[0]) not in [int, np.int32]:
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
        self._index = 0
        self.p = pcoll.data_accessor()
        self._head = pcoll.data_accessor()
        self._head.set_index(0)
        self._tail = pcoll.data_accessor()
        self._tail.set_index(self.max_len - 1)

    def __next__(self):
        if self._index < self.max_len:
            self.p.set_index(self._indices[self._index])
            result = self.p
            self._index += 1
            return result

        # End of Iteration
        raise StopIteration

    @property
    def current(self):
        return self.p

    def __repr__(self):
        dir_str = 'Backward' if self._reverse else 'Forward'
        str = f"{dir_str} iteration at index {self._index} of {self.max_len}."
        return str
