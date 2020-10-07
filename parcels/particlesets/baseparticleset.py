import numpy as np
from abc import ABC
from abc import abstractmethod
import warnings

from parcels.tools.statuscodes import OperationCode
from .collections import ParticleCollection


class NDCluster(ABC):
    """Interface."""


class BaseParticleSet(ParticleCollection, NDCluster):
    """Base ParticleSet."""
    def data_accessor(self):
        """Returns an Accessor for the particles in this ParticleSet.
        Deprecated - the accessor must not be manipulated directly.
        """
        warnings.warn(
            "This method of accessing particle data is deprecated",
            DeprecationWarning,
            stacklevel=2
        )

    @abstractmethod
    def execute(self, pyfunc, endtime, runtime, dt, moviedt, recovery,
                output_file, movie_background_field, verbose_progress,
                postIterationCallbacks, callbackdt):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.Kernel` object.
                       Kernels can be concatenated using the + operator
        :param endtime: End time for the timestepping loop.
                        It is either a datetime object or a positive double.
        :param runtime: Length of the timestepping loop. Use instead of endtime.
                        It is either a timedelta object or a positive double.
        :param dt: Timestep interval to be passed to the kernel.
                   It is either a timedelta object or a double.
                   Use a negative value for a backward-in-time simulation.
        :param moviedt:  Interval for inner sub-timestepping (leap), which dictates
                         the update frequency of animation.
                         It is either a timedelta object or a positive double.
                         None value means no animation.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param recovery: Dictionary with additional `:mod:parcels.tools.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
        :param movie_background_field: field plotted as background in the movie if moviedt is set.
                                       'vector' shows the velocity as a vector field.
        :param verbose_progress: Boolean for providing a progress bar for the kernel execution loop.
        :param postIterationCallbacks: (Optional) Array of functions that are to be called after each iteration (post-process, non-Kernel)
        :param callbackdt: (Optional, in conjecture with 'postIterationCallbacks) timestep inverval to (latestly) interrupt the running kernel and invoke post-iteration callbacks from 'postIterationCallbacks'
        """
        pass
