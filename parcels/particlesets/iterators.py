from abc import ABC
from abc import abstractmethod
import warnings


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
        # Accessor objects should not be mutable. This functionaility
        # will be removed.
        warnings.warn(
            "The Accessor cannot switch to representing a different particle"
            " after its creation.",
            DeprecationWarning,
            stacklevel=2
        )
        self._index = index

    def update_next_dt(self, next_dt=None):
        # == OBJECTION CK: Also here - make a guarded forward ...  == #
        # == RESPONSE RB: The response I provided below I think is == #
        # == particularly applicable here.                         == #
        # if next_dt is None:
        #     if not np.isnan(self._next_dt):
        #         self.dt, self._next_dt = self._next_dt, np.nan
        # else:
        #     self._next_dt = next_dt
        pass

    def delete(self):
        # == OBJECTION CK: the actual operation, which is the particle's state, shall be done by the particle. So, == #
        # == this function should just forward the delete-call to the particle in question.                        == #
        # == RESPONSE RB: There might be a problem with that in some cases. As we discussed, the ParticleAccessor  == #
        # == basically acts like a shell around the particle (data), providing uniform access regardless of the    == #
        # == underlying datastructure. The SOA approach, for example, does not use a particle class on which       == #
        # == functions can be defined. So it makes sense, I think, to implement any function that requires more    == #
        # == logic than just setting a property on the Accessor level. The state-thing is not really a good        == #
        # == example, although the delete-alias may be useful over just treating it as a property.                 == #
        # self.state = OperationCode.Delete
        pass

    def set_state(self, state):
        # Convert into a "proper" property?
        # Why is this even separate? It sets the state of the particle,
        # so should be handled by the __setattr__ function, right?
        # Seems to be coppied directly from ScipyParticle.

        # == OBJECTION CK: the actual operation, which is the particle's state, shall be done by the particle. So, == #
        # == this function should just forward the delete-call to the particle in question.                        == #
        # == RESPONSE RB: There might be a problem with that in some cases. As we discussed, the ParticleAccessor  == #
        # == basically acts like a shell around the particle (data), providing uniform access regardless of the    == #
        # == underlying datastructure. The SOA approach, for example, does not use a particle class on which       == #
        # == functions can be defined. So it makes sense, I think, to implement any function that requires more    == #
        # == logic than just setting a property on the Accessor level. The state-thing is not really a good        == #
        # == example, although the delete-alias may be useful over just treating it as a property.                 == #

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