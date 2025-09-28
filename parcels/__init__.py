from ._version import version

__version__ = version

import warnings as _stdlib_warnings

from parcels._logger import logger
from parcels._tutorial import download_example_dataset, list_example_datasets
from parcels._warnings import (
    FieldSetWarning,
    FileWarning,
    KernelWarning,
    ParticleSetWarning,
)
from parcels._core.field import Field, VectorField
from parcels._core.fieldset import FieldSet
from parcels._core.kernel import Kernel
from parcels._core.particle import (
    KernelParticle,  # ? remove?
    Particle,
    ParticleClass,
    Variable,
)
from parcels._core.particlefile import ParticleFile
from parcels._core.particleset import ParticleSet
from parcels._core.statuscodes import (
    AllParcelsErrorCodes,
    FieldOutOfBoundError,
    FieldSamplingError,
    KernelError,
    StatusCode,
    FieldInterpolationError,
    TimeExtrapolationError,
)
from parcels.utils import *

from parcels.converters import (
    Geographic,
    GeographicPolar,
    GeographicPolarSquare,
    GeographicSquare,
    UnitConverter,
)
from parcels._core.xgrid import XGrid
from parcels._core.uxgrid import UxGrid
from parcels._core.basegrid import BaseGrid

_stdlib_warnings.warn(
    "This is an alpha version of Parcels v4. The API is not stable and may change without deprecation warnings.",
    UserWarning,
    stacklevel=2,
)
