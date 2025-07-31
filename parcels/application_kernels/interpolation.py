"""Collection of pre-built interpolation kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

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

    axis_dim = field.grid.get_axis_dim_mapping(field.data.dims)
    data = field.data
    val = np.zeros_like(tau)

    xii = np.clip(np.stack([xi, xi + 1, xi, xi + 1], axis=-1).flatten(), 0, data.shape[3] - 1)
    yii = np.clip(np.stack([yi, yi, yi + 1, yi + 1], axis=-1).flatten(), 0, data.shape[2] - 1)
    xi_da = xr.DataArray(xii, dims=("points"))
    yi_da = xr.DataArray(yii, dims=("points"))

    timeslices = [ti, ti + 1] if tau.any() > 0 else [ti]
    depth_slices = [zi, zi + 1] if zeta.any() > 0 else [zi]
    for tii, tau_factor in zip(timeslices, [1 - tau, tau], strict=False):
        tti = np.clip(np.array([tii, tii, tii, tii]).flatten(), 0, data.shape[0] - 1)
        ti_da = xr.DataArray(tti, dims=("points"))
        for zii, depth_factor in zip(depth_slices, [1 - zeta, zeta], strict=False):
            zii = np.clip(np.array([zii, zii, zii, zii]).flatten(), 0, data.shape[1] - 1)
            zi_da = xr.DataArray(zii, dims=("points"))

            F = data.isel({axis_dim["X"]: xi_da, axis_dim["Y"]: yi_da, axis_dim["Z"]: zi_da, "time": ti_da})
            F = F.data.reshape(-1, 4)
            # TODO check if numpy can handle this more efficiently
            # F = data.values[tti, zii, yii, xii].reshape(-1, 4)
            interp_val = (
                (1 - xsi) * (1 - eta) * F[:, 0]
                + xsi * (1 - eta) * F[:, 1]
                + (1 - xsi) * eta * F[:, 2]
                + xsi * eta * F[:, 3]
            )

            val += interp_val * tau_factor * depth_factor

    return val


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
