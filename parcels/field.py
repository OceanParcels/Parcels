from typing import TYPE_CHECKING

import numpy as np

from parcels._typing import (
    VectorType,
)
from parcels.tools.statuscodes import (
    AllParcelsErrorCodes,
)

if TYPE_CHECKING:
    pass

__all__ = ["Field", "NestedField", "VectorField"]


def _isParticle(key):
    if hasattr(key, "obs_written"):
        return True
    else:
        return False


def _deal_with_errors(error, key, vector_type: VectorType):
    if _isParticle(key):
        key.state = AllParcelsErrorCodes[type(error)]
    elif _isParticle(key[-1]):
        key[-1].state = AllParcelsErrorCodes[type(error)]
    else:
        raise RuntimeError(f"{error}. Error could not be handled because particle was not part of the Field Sampling.")

    if vector_type and "3D" in vector_type:
        return (0, 0, 0)
    elif vector_type == "2D":
        return (0, 0)
    else:
        return 0


def _croco_from_z_to_sigma_scipy(fieldset, time, z, y, x, particle):
    """Calculate local sigma level of the particle, by linearly interpolating the
    scaling function that maps sigma to depth (using local ocean depth H,
    sea-surface Zeta and stretching parameters Cs_w and hc).
    See also https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.grid.html#vertical-grid-parameters
    """
    h = fieldset.H.eval(time, 0, y, x, particle=particle, applyConversion=False)
    zeta = fieldset.Zeta.eval(time, 0, y, x, particle=particle, applyConversion=False)
    sigma_levels = fieldset.U.grid.depth
    z0 = fieldset.hc * sigma_levels + (h - fieldset.hc) * fieldset.Cs_w.data[0, :, 0, 0]
    zvec = z0 + zeta * (1 + (z0 / h))
    zinds = zvec <= z
    if z >= zvec[-1]:
        zi = len(zvec) - 2
    else:
        zi = zinds.argmin() - 1 if z >= zvec[0] else 0

    return sigma_levels[zi] + (z - zvec[zi]) * (sigma_levels[zi + 1] - sigma_levels[zi]) / (zvec[zi + 1] - zvec[zi])


class Field:
    interp_method = "cgrid_velocity"
    allow_time_extrapolation = False

    def eval(self, *args, **kwargs):
        return np.random.normal()


class VectorField: ...


class NestedField(list): ...
