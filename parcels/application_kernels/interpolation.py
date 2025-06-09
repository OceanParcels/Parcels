"""Collection of pre-built interpolation kernels."""

import numpy as np

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
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """
    Piecewise constant interpolation kernel for face registered data.
    This interpolation method is appropriate for fields that are
    face registered, such as u,v in FESOM.
    """
    # TODO joe : handle vertical interpolation
    zi, fi = field.grid.unravel_index(ei)
    return field.data[ti, zi, fi]


def UXPiecewiseLinearNode(
    field: Field,
    ti: int,
    ei: int,
    bcoords: np.ndarray,
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """
    Piecewise linear interpolation kernel for node registered data. This
    interpolation method is appropriate for fields that are node registered
    such as the vertical velocity w in FESOM.
    """
    # TODO joe : handle vertical interpolation
    zi, fi = field.grid.unravel_index(ei)
    node_ids = field.grid.uxgrid.face_node_connectivity[fi, :]
    return np.dot(field.data[ti, zi, node_ids], bcoords)
