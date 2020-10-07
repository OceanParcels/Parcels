from abc import ABC
from abc import abstractmethod


class BaseParticleCollectionIterator(ABC):
    """Interface for the ParticleCollection iterator. Provides the
    ability to iterate over the particles in the ParticleCollection.
    """
    def ___init___(self):
        self._head = None
        self._tail = None
        self._current = None

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """Returns the next value from ParticleSet object's lists."""
        pass

    @property
    def head(self):
        return self._head

    @property
    def tail(self):
        return self._tail

    @property
    def current(self):
        return self._current

    @abstractmethod
    def __repr__(self):
        """Represents the current position in the iteration.
        """
        pass


class BaseParticleAccessor(ABC):
    """Interface for the ParticleAccessor. Implements a wrapper around
    particles to provide easy access."""
    def __init__(self, pcoll):
        """Initialize the ParticleAccessor object with at least a
        reference to the ParticleSet it encapsulates.
        """
        self.pcoll = pcoll

    def set_index(self, index):
        # Convert into a "proper" property?
        self._index = index

    def update_next_dt(self, next_dt=None):
        # == OBJECTION CK: Also here - make a guarded forward ... == #
        # if next_dt is None:
        #     if not np.isnan(self._next_dt):
        #         self.dt, self._next_dt = self._next_dt, np.nan
        # else:
        #     self._next_dt = next_dt
        pass

    def delete(self):
        # == OBJECTION CK: the actual operation, which is the particle's state, shall be done by the particle. So, == #
        # == this function should just forward the delete-call to the particle in question.                        == #
        # self.state = OperationCode.Delete
        pass

    def set_state(self, state):
        # Convert into a "proper" property?
        # Why is this even separate? It sets the state of the particle,
        # so should be handled by the __setattr__ function, right?
        # Seems to be coppied directly from ScipyParticle.

        # == OBJECTION CK: the actual operation, which is the particle's state, shall be done by the particle. So, == #
        # == this function should just forward the delete-call to the particle in question.                        == #

        # self.state = state
        pass

    @abstractmethod
    def __getattr__(self, name):
        """The ParticleAccessor should provide an implementation of this
        built-in function to allow accessing particle attributes in its
        corresponding ParticleSet datastructure.
        """
        pass

    @abstractmethod
    def __setattr__(self, name, value):
        """The ParticleAccessor should provide an implementation of this
        built-in function to allow setting particle attributes in its
        corresponding ParticleSet datastructure.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """The ParticleAccessor should provide an implementation of this
        built-in function that returns a string representation of the
        Particle that it currently provides access to.
        """
        pass