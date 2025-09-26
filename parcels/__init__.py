from ._version import version

__version__ = version

import warnings as _stdlib_warnings

from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.kernel import Kernel
from parcels.particle import (
    KernelParticle,  # ? remove?
    Particle,
    ParticleClass,
    Variable,
)
from parcels._warnings import (
    FieldSetWarning,
    FileWarning,
    KernelWarning,
    ParticleSetWarning,
)
from parcels.particlefile import ParticleFile
from parcels.particleset import ParticleSet
from parcels._logger import logger
from parcels.tools import *
from parcels._tutorial import download_example_dataset, list_example_datasets

_stdlib_warnings.warn(
    "This is an alpha version of Parcels v4. The API is not stable and may change without deprecation warnings.",
    UserWarning,
    stacklevel=2,
)
