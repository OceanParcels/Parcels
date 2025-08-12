"""Collection of pre-built interpolation kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from parcels.field import Field
from parcels.tools.statuscodes import (
    FieldOutOfBoundError,
)

if TYPE_CHECKING:
    from parcels.uxgrid import _UXGRID_AXES
    from parcels.xgrid import _XGRID_AXES

__all__ = [
    "UXPiecewiseConstantFace",
    "UXPiecewiseLinearNode",
    "XBiLinear",
    "XBiLinearPeriodic",
    "XTriLinear",
]


def XBiLinear(
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Bilinear interpolation on a regular grid."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, _ = position["Z"]

    data = field.data.data[:, zi, yi : yi + 2, xi : xi + 2]
    if tau > 0:
        data = (1 - tau) * data[ti, :, :] + tau * data[ti + 1, :, :]
    else:
        data = data[ti, :, :]

    return (
        (1 - xsi) * (1 - eta) * data[0, 0]
        + xsi * (1 - eta) * data[0, 1]
        + xsi * eta * data[1, 1]
        + (1 - xsi) * eta * data[1, 0]
    )


def XBiLinearPeriodic(
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Bilinear interpolation on a regular grid with periodic boundary conditions in horizontal directions."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, _ = position["Z"]

    if xi < 0:
        xi = 0
        xsi = (x - field.grid.lon[xi]) / (field.grid.lon[xi + 1] - field.grid.lon[xi])
    if yi < 0:
        yi = 0
        eta = (y - field.grid.lat[yi]) / (field.grid.lat[yi + 1] - field.grid.lat[yi])

    data = field.data.data[:, zi, yi : yi + 2, xi : xi + 2]
    data = (1 - tau) * data[ti, :, :] + tau * data[ti + 1, :, :]

    xsi = 0 if not np.isfinite(xsi) else xsi
    eta = 0 if not np.isfinite(eta) else eta

    if xsi > 0 and eta > 0:
        return (
            (1 - xsi) * (1 - eta) * data[0, 0]
            + xsi * (1 - eta) * data[0, 1]
            + xsi * eta * data[1, 1]
            + (1 - xsi) * eta * data[1, 0]
        )
    elif xsi > 0 and eta == 0:
        return (1 - xsi) * data[0, 0] + xsi * data[0, 1]
    elif xsi == 0 and eta > 0:
        return (1 - eta) * data[0, 0] + eta * data[1, 0]
    else:
        return data[0, 0]


def XTriLinear(
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Trilinear interpolation on a regular grid."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    if zi < 0 or xi < 0 or yi < 0:
        raise FieldOutOfBoundError

    data = field.data.data[:, zi : zi + 2, yi : yi + 2, xi : xi + 2]
    data = (1 - tau) * data[ti, :, :, :] + tau * data[ti + 1, :, :, :]
    if zeta > 0:
        data = (1 - zeta) * data[0, :, :] + zeta * data[1, :, :]
    else:
        data = data[0, :, :]

    xsi = 0 if not np.isfinite(xsi) else xsi
    eta = 0 if not np.isfinite(eta) else eta

    if xsi > 0 and eta > 0:
        return (
            (1 - xsi) * (1 - eta) * data[0, 0]
            + xsi * (1 - eta) * data[0, 1]
            + xsi * eta * data[1, 1]
            + (1 - xsi) * eta * data[1, 0]
        )
    elif xsi > 0 and eta == 0:
        return (1 - xsi) * data[0, 0] + xsi * data[0, 1]
    elif xsi == 0 and eta > 0:
        return (1 - eta) * data[0, 0] + eta * data[1, 0]
    else:
        return data[0, 0]


def UXPiecewiseConstantFace(
    field: Field,
    ti: int,
    position: dict[_UXGRID_AXES, tuple[int, float | np.ndarray]],
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
    return field.data.values[ti, position["Z"][0], position["FACE"][0]]


def UXPiecewiseLinearNode(
    field: Field,
    ti: int,
    position: dict[_UXGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """
    Piecewise linear interpolation kernel for node registered data located at vertical interface levels.
    This interpolation method is appropriate for fields that are node registered such as the vertical
    velocity W in FESOM2. Effectively, it applies barycentric interpolation in the lateral direction
    and piecewise linear interpolation in the vertical direction.
    """
    k, fi = position["Z"][0], position["FACE"][0]
    bcoords = position["FACE"][1]
    node_ids = field.grid.uxgrid.face_node_connectivity[fi, :]
    # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
    # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
    # First, do barycentric interpolation in the lateral direction for each interface level
    fzk = np.dot(field.data.values[ti, k, node_ids], bcoords)
    fzkp1 = np.dot(field.data.values[ti, k + 1, node_ids], bcoords)

    # Then, do piecewise linear interpolation in the vertical direction
    zk = field.grid.z.values[k]
    zkp1 = field.grid.z.values[k + 1]
    return (fzk * (zkp1 - z) + fzkp1 * (z - zk)) / (zkp1 - zk)  # Linear interpolation in the vertical direction
