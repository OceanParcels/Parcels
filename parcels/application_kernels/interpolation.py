"""Collection of pre-built interpolation kernels."""

import math
from typing import Union
import numpy as np

from parcels.tools.statuscodes import StatusCode
from parcels.field import Field

__all__ = [
    "UXPiecewiseConstantFace",
    "UXPiecewiseLinearNode",
]

def UXPiecewiseConstantFace(
        field: Field,
        ti: int,
        ei: int,
        bcoords: np.ndarray,
        tau: Union[np.float32,np.float64],
        t: Union[np.float32,np.float64],
        z: Union[np.float32,np.float64],
        y: Union[np.float32,np.float64],
        x: Union[np.float32,np.float64]
    ):
    """
    Piecewise constant interpolation kernel for face registered data. 
    This interpolation method is appropriate for fields that are
    face registered, such as u,v in FESOM.
    """
    # To do : handle vertical interpolation
    zi, fi = field.unravel_index(ei)
    return field.data[ti, zi, fi]

def UXPiecewiseLinearNode(
        field: Field,
        ti: int,
        ei: int,
        bcoords: np.ndarray,
        tau: Union[np.float32,np.float64],
        t: Union[np.float32,np.float64],
        z: Union[np.float32,np.float64],
        y: Union[np.float32,np.float64],
        x: Union[np.float32,np.float64]
    ):
    """
    Piecewise linear interpolation kernel for node registered data. This
    interpolation method is appropriate for fields that are node registered
    such as the vertical velocity w in FESOM.
    """
    # To do: handle vertical interpolation
    zi, fi = field.unravel_index(ei)
    node_ids = field.data.uxgrid.face_node_connectivity[fi,:]
    return np.dot(field.data[ti, zi, node_ids],bcoords)
