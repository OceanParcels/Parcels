"""Collection of pre-built interpolation kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as dask
import numpy as np

if TYPE_CHECKING:
    from parcels.field import Field
    from parcels.uxgrid import _UXGRID_AXES
    from parcels.xgrid import _XGRID_AXES

__all__ = [
    "UXPiecewiseConstantFace",
    "UXPiecewiseLinearNode",
    "XLinear",
    "ZeroInterpolator",
]


def ZeroInterpolator(
    field: Field,
    ti: int,
    position: dict[str, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
) -> np.float32 | np.float64:
    """Template function used for the signature check of the lateral interpolation methods."""
    return 0.0


def XLinear(
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

    data = field.data

    lenT = 2 if np.any(tau > 0) else 1
    lenZ = 2 if np.any(zeta > 0) else 1

    # Time coordinates: 8 points at ti, then 8 points at ti+1
    if lenT == 1:
        ti = np.repeat(ti, lenZ * 4)
    else:
        ti_1 = np.clip(ti + 1, 0, data.shape[0] - 1)
        ti = np.concatenate([np.repeat(ti, lenZ * 4), np.repeat(ti_1, lenZ * 4)])

    # Depth coordinates: 4 points at zi, 4 at zi+1, repeated for both time levels
    if lenZ == 1:
        zi = np.repeat(zi, lenT * 4)
    else:
        zi_1 = np.clip(zi + 1, 0, data.shape[1] - 1)
        zi = np.tile(np.stack([zi, zi, zi, zi, zi_1, zi_1, zi_1, zi_1], axis=1).flatten(), lenT)

    # Y coordinates: [yi, yi, yi+1, yi+1] pattern repeated
    yi_1 = np.clip(yi + 1, 0, data.shape[2] - 1)
    yi = np.tile(np.repeat(np.column_stack([yi, yi_1]), 2), (lenT) * (lenZ))

    # X coordinates: [xi, xi+1] for each spatial point, repeated for time/depth
    xi_1 = np.clip(xi + 1, 0, data.shape[3] - 1)
    xi = np.repeat(np.column_stack([xi, xi_1]), 2 * (lenT) * (lenZ))

    corner_data = data.values[ti, zi, yi, xi].reshape(-1, lenT, lenZ, 4)

    if lenT == 2:
        tau = tau[:, np.newaxis, np.newaxis]
        corner_data = corner_data[:, 0, :, :] * (1 - tau) + corner_data[:, 1, :, :] * tau
    else:
        corner_data = corner_data[:, 0, :, :]

    if lenZ == 2:
        zeta = zeta[:, np.newaxis]
        corner_data = corner_data[:, 0, :] * (1 - zeta) + corner_data[:, 1, :] * zeta
    else:
        corner_data = corner_data[:, 0, :]

    value = (
        (1 - xsi) * (1 - eta) * corner_data[:, 0]
        + xsi * (1 - eta) * corner_data[:, 1]
        + (1 - xsi) * eta * corner_data[:, 2]
        + xsi * eta * corner_data[:, 3]
    )
    return value.compute() if isinstance(value, dask.Array) else value


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
