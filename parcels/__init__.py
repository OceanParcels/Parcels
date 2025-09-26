from ._version import version

__version__ = version

import warnings as _warnings

from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.kernel import Kernel
from parcels.kernels import *
from parcels.particle import (
    KernelParticle,  # ? remove?
    Particle,
    ParticleClass,
    Variable,
)
from parcels.particlefile import ParticleFile
from parcels.particleset import ParticleSet
from parcels.tools import *

_warnings.warn(
    "This is an alpha version of Parcels v4. The API is not stable and may change without deprecation warnings.",
    UserWarning,
    stacklevel=2,
)
