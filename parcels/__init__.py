from ._version import version

__version__ = version

import warnings

from parcels.application_kernels import *
from parcels.field import *
from parcels.fieldset import *
from parcels.interaction import *
from parcels.kernel import *
from parcels.particle import *
from parcels.particlefile import *
from parcels.particleset import *
from parcels.tools import *

warnings.warn(
    "This is an alpha version of Parcels v4. The API is not stable and may change without deprecation warnings.",
    UserWarning,
    stacklevel=2,
)
