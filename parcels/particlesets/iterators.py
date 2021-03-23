from abc import ABC
from abc import abstractmethod
from parcels.tools.statuscodes import OperationCode, StateCode


class BaseParticleCollectionIterable(ABC):
    _pcoll_immutable = None
    _reverse = None
    _subset = None

    def __init__(self, pcoll, reverse=False, subset=None):
        self._pcoll_immutable = pcoll
        self._reverse = reverse
        self._subset = subset

    @abstractmethod
    def __iter__(self):
        pass


class BaseParticleCollectionIterator(ABC):
    """Interface for the ParticleCollection iterator. Provides the
    ability to iterate over the particles in the ParticleCollection.
    """
    def ___init___(self):
        self._head = None
        self._tail = None
        self._current = None

    @abstractmethod
    def __next__(self):
        """Returns a ParticleAccessor for the next particle in the
        ParticleSet.
        """
        pass

    @property
    def head(self):
        """Returns a ParticleAccessor for the first particle in the
        ParticleSet.
        """
        return self._head

    @property
    def tail(self):
        """Returns a ParticleAccessor for the last particle in the
        ParticleSet.
        """
        return self._tail

    @property
    def current(self):
        """Returns a ParticleAccessor for the particle that the iteration
        is currently at.
        """
        return self._current

    @abstractmethod
    def __repr__(self):
        """Represents the current position in the iteration.
        """
        pass


class BaseParticleAccessor(ABC):
    """Interface for the ParticleAccessor. Implements a wrapper around
    particles to provide easy access."""
    _pcoll = None

    def __init__(self, pcoll):
        """Initialize the ParticleAccessor object with at least a
        reference to the ParticleSet it encapsulates.
        """
        self._pcoll = pcoll

    @abstractmethod
    def update_next_dt(self, next_dt=None):
        pass

    def delete(self):
        """Signal the underlying particle for deletion."""
        self.state = OperationCode.Delete

    def set_state(self, state):
        """Syntactic sugar for changing the state of the underlying
        particle.
        """
        self.state = state

    def succeeded(self):
        self.state = StateCode.Success

    def isComputed(self):
        return self.state == StateCode.Success

    def reset_state(self):
        self.state = StateCode.Evaluate

    @abstractmethod
    def getPType(self):
        return None

    def __getattr__(self, name):
        """The ParticleAccessor should provide an implementation of this
        built-in function to allow accessing particle attributes in its
        corresponding ParticleSet datastructure.
        """
        if name in ['_pcoll', ]:
            return super(BaseParticleAccessor, self).__getattribute__(name)
        return None

    def __setattr__(self, name, value):
        """The ParticleAccessor should provide an implementation of this
        built-in function to allow setting particle attributes in its
        corresponding ParticleSet datastructure.
        """
        if name in ['_pcoll', ]:
            super(BaseParticleAccessor, self).__setattr__(name, value)

    @abstractmethod
    def __repr__(self):
        """The ParticleAccessor should provide an implementation of this
        built-in function that returns a string representation of the
        Particle that it currently provides access to.
        """
        pass
