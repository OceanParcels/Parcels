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

    # Get dimension sizes to clip indices
    x_size = data.sizes[axis_dim["X"]]
    y_size = data.sizes[axis_dim["Y"]]
    z_size = data.sizes[axis_dim["Z"]]
    t_size = data.sizes["time"]

    timeslices = [ti, ti + 1] if tau.any() > 0 else [ti]
    for tii, tau_factor in zip(timeslices, [1 - tau, tau], strict=False):
        for zii, depth_factor in zip([zi, zi + 1], [1 - zeta, zeta], strict=False):
            # Clip indices to prevent out-of-bounds access
            xi_clipped = np.clip(xi, 0, x_size - 1)
            yi_clipped = np.clip(yi, 0, y_size - 1)
            zii_clipped = np.clip(zii, 0, z_size - 1)
            tii_clipped = np.clip(tii, 0, t_size - 1)

            xi_da = xr.DataArray(xi_clipped, dims="points")
            yi_da = xr.DataArray(yi_clipped, dims="points")
            zinds = xr.DataArray(zii_clipped, dims="points")
            tinds = xr.DataArray(tii_clipped, dims="points")
            xi_plus1 = xr.DataArray(np.clip(xi_clipped + 1, 0, x_size - 1), dims="points")
            yi_plus1 = xr.DataArray(np.clip(yi_clipped + 1, 0, y_size - 1), dims="points")

            # TODO see if this can be done with one isel call, by combining the indices
            F00 = data.isel(
                {axis_dim["X"]: xi_da, axis_dim["Y"]: yi_da, axis_dim["Z"]: zinds, "time": tinds}
            ).values.flatten()
            F10 = data.isel(
                {axis_dim["X"]: xi_plus1, axis_dim["Y"]: yi_da, axis_dim["Z"]: zinds, "time": tinds}
            ).values.flatten()
            F01 = data.isel(
                {axis_dim["X"]: xi_da, axis_dim["Y"]: yi_plus1, axis_dim["Z"]: zinds, "time": tinds}
            ).values.flatten()
            F11 = data.isel(
                {axis_dim["X"]: xi_plus1, axis_dim["Y"]: yi_plus1, axis_dim["Z"]: zinds, "time": tinds}
            ).values.flatten()
            val += (
                ((1 - xsi) * (1 - eta) * F00 + xsi * (1 - eta) * F10 + (1 - xsi) * eta * F01 + xsi * eta * F11)
                * tau_factor
                * depth_factor
            )

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
