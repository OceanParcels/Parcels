"""Collection of pre-built interpolation kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as dask
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from parcels.field import Field, VectorField
    from parcels.uxgrid import _UXGRID_AXES
    from parcels.xgrid import _XGRID_AXES

__all__ = [
    "UXPiecewiseConstantFace",
    "UXPiecewiseLinearNode",
    "XFreeslip",
    "XLinear",
    "XNearest",
    "XPartialslip",
    "ZeroInterpolator",
    "ZeroInterpolator_Vector",
]


def ZeroInterpolator(
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
) -> np.float32 | np.float64:
    """Template function used for the signature check of the lateral interpolation methods."""
    return 0.0


def ZeroInterpolator_Vector(
    vectorfield: VectorField,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
) -> np.float32 | np.float64:
    """Template function used for the signature check of the interpolation methods for velocity fields."""
    return 0.0


def _get_corner_data_Agrid(
    data: np.ndarray | xr.DataArray,
    ti: int,
    zi: int,
    yi: int,
    xi: int,
    lenT: int,
    lenZ: int,
    npart: int,
    axis_dim: dict[str, str],
) -> np.ndarray:
    """Helper function to get the corner data for a given A-grid field and position."""
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
        zi = np.tile(np.array([zi, zi, zi, zi, zi_1, zi_1, zi_1, zi_1]).flatten(), lenT)

    # Y coordinates: [yi, yi, yi+1, yi+1] for each spatial point, repeated for time/depth
    yi_1 = np.clip(yi + 1, 0, data.shape[2] - 1)
    yi = np.tile(np.repeat(np.column_stack([yi, yi_1]), 2), (lenT) * (lenZ))

    # X coordinates: [xi, xi+1, xi, xi+1] for each spatial point, repeated for time/depth
    xi_1 = np.clip(xi + 1, 0, data.shape[3] - 1)
    xi = np.tile(np.column_stack([xi, xi_1, xi, xi_1]).flatten(), (lenT) * (lenZ))

    # Create DataArrays for indexing
    selection_dict = {
        axis_dim["X"]: xr.DataArray(xi, dims=("points")),
        axis_dim["Y"]: xr.DataArray(yi, dims=("points")),
    }
    if "Z" in axis_dim:
        selection_dict[axis_dim["Z"]] = xr.DataArray(zi, dims=("points"))
    if "time" in data.dims:
        selection_dict["time"] = xr.DataArray(ti, dims=("points"))

    return data.isel(selection_dict).data.reshape(lenT, lenZ, npart, 4)


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

    lenT = 2 if np.any(tau > 0) else 1
    lenZ = 2 if np.any(zeta > 0) else 1

    corner_data = _get_corner_data_Agrid(data, ti, zi, yi, xi, lenT, lenZ, len(xsi), axis_dim)

    if lenT == 2:
        tau = tau[np.newaxis, :, np.newaxis]
        corner_data = corner_data[0, :, :, :] * (1 - tau) + corner_data[1, :, :, :] * tau
    else:
        corner_data = corner_data[0, :, :, :]

    if lenZ == 2:
        zeta = zeta[:, np.newaxis]
        corner_data = corner_data[0, :, :] * (1 - zeta) + corner_data[1, :, :] * zeta
    else:
        corner_data = corner_data[0, :, :]

    value = (
        (1 - xsi) * (1 - eta) * corner_data[:, 0]
        + xsi * (1 - eta) * corner_data[:, 1]
        + (1 - xsi) * eta * corner_data[:, 2]
        + xsi * eta * corner_data[:, 3]
    )
    return value.compute() if isinstance(value, dask.Array) else value


def _Spatialslip(
    vectorfield: VectorField,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
    a: np.float32,
    b: np.float32,
):
    """Helper function for spatial boundary condition interpolation for velocity fields."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    axis_dim = vectorfield.U.grid.get_axis_dim_mapping(vectorfield.U.data.dims)
    lenT = 2 if np.any(tau > 0) else 1
    lenZ = 2 if np.any(zeta > 0) else 1
    npart = len(xsi)

    u = XLinear(vectorfield.U, ti, position, tau, t, z, y, x)
    v = XLinear(vectorfield.V, ti, position, tau, t, z, y, x)
    if vectorfield.W:
        w = XLinear(vectorfield.W, ti, position, tau, t, z, y, x)

    corner_dataU = _get_corner_data_Agrid(vectorfield.U.data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim)
    corner_dataV = _get_corner_data_Agrid(vectorfield.V.data, ti, zi, yi, xi, lenT, lenZ, npart, axis_dim)

    def is_land(ti: int, zi: int, yi: int, xi: int):
        uval = corner_dataU[ti, zi, :, xi + 2 * yi]
        vval = corner_dataV[ti, zi, :, xi + 2 * yi]
        return np.where(np.isclose(uval, 0.0) & np.isclose(vval, 0.0), True, False)

    f_u = np.ones_like(xsi)
    f_v = np.ones_like(eta)

    if lenZ == 1:
        f_u = np.where(is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & (eta > 0), f_u * (a + b * eta) / eta, f_u)
        f_u = np.where(is_land(0, 0, 1, 0) & is_land(0, 0, 1, 1) & (eta < 1), f_u * (1 - b * eta) / (1 - eta), f_u)
        f_v = np.where(is_land(0, 0, 0, 0) & is_land(0, 0, 1, 0) & (xsi > 0), f_v * (a + b * xsi) / xsi, f_v)
        f_v = np.where(is_land(0, 0, 0, 1) & is_land(0, 0, 1, 1) & (xsi < 1), f_v * (1 - b * xsi) / (1 - xsi), f_v)
    else:
        f_u = np.where(
            is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & is_land(0, 1, 0, 0) & is_land(0, 1, 0, 1) & (eta > 0),
            f_u * (a + b * eta) / eta,
            f_u,
        )
        f_u = np.where(
            is_land(0, 0, 1, 0) & is_land(0, 0, 1, 1) & is_land(0, 1, 1, 0) & is_land(0, 1, 1, 1) & (eta < 1),
            f_u * (1 - b * eta) / (1 - eta),
            f_u,
        )
        f_v = np.where(
            is_land(0, 0, 0, 0) & is_land(0, 0, 1, 0) & is_land(0, 1, 0, 0) & is_land(0, 1, 1, 0) & (xsi > 0),
            f_v * (a + b * xsi) / xsi,
            f_v,
        )
        f_v = np.where(
            is_land(0, 0, 0, 1) & is_land(0, 0, 1, 1) & is_land(0, 1, 0, 1) & is_land(0, 1, 1, 1) & (xsi < 1),
            f_v * (1 - b * xsi) / (1 - xsi),
            f_v,
        )
        f_u = np.where(
            is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & is_land(0, 0, 1, 0 & is_land(0, 0, 1, 1) & (zeta > 0)),
            f_u * (a + b * zeta) / zeta,
            f_u,
        )
        f_u = np.where(
            is_land(0, 1, 0, 0) & is_land(0, 1, 0, 1) & is_land(0, 1, 1, 0 & is_land(0, 1, 1, 1) & (zeta < 1)),
            f_u * (1 - b * zeta) / (1 - zeta),
            f_u,
        )
        f_v = np.where(
            is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & is_land(0, 0, 1, 0 & is_land(0, 0, 1, 1) & (zeta > 0)),
            f_v * (a + b * zeta) / zeta,
            f_v,
        )
        f_v = np.where(
            is_land(0, 1, 0, 0) & is_land(0, 1, 0, 1) & is_land(0, 1, 1, 0 & is_land(0, 1, 1, 1) & (zeta < 1)),
            f_v * (1 - b * zeta) / (1 - zeta),
            f_v,
        )

    u *= f_u
    v *= f_v
    if vectorfield.W:
        f_w = np.ones_like(zeta)
        f_w = np.where(
            is_land(0, 0, 0, 0) & is_land(0, 0, 0, 1) & is_land(0, 1, 0, 0) & is_land(0, 1, 0, 1) & (eta > 0),
            f_w * (a + b * eta) / eta,
            f_w,
        )
        f_w = np.where(
            is_land(0, 0, 1, 0) & is_land(0, 0, 1, 1) & is_land(0, 1, 1, 0) & is_land(0, 1, 1, 1) & (eta < 1),
            f_w * (a - b * eta) / (1 - eta),
            f_w,
        )
        f_w = np.where(
            is_land(0, 0, 0, 0) & is_land(0, 0, 1, 0) & is_land(0, 1, 0, 0) & is_land(0, 1, 1, 0) & (xsi > 0),
            f_w * (a + b * xsi) / xsi,
            f_w,
        )
        f_w = np.where(
            is_land(0, 0, 0, 1) & is_land(0, 0, 1, 1) & is_land(0, 1, 0, 1) & is_land(0, 1, 1, 1) & (xsi < 1),
            f_w * (a - b * xsi) / (1 - xsi),
            f_w,
        )

        w *= f_w
    else:
        w = None
    return u, v, w


def XFreeslip(
    vectorfield: VectorField,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Free-slip boundary condition interpolation for velocity fields."""
    return _Spatialslip(vectorfield, ti, position, tau, t, z, y, x, a=1.0, b=0.0)


def XPartialslip(
    vectorfield: VectorField,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Partial-slip boundary condition interpolation for velocity fields."""
    return _Spatialslip(vectorfield, ti, position, tau, t, z, y, x, a=0.5, b=0.5)


def XNearest(
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """
    Nearest-Neighbour spatial interpolation on a regular grid.
    Note that this still uses linear interpolation in time.
    """
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    axis_dim = field.grid.get_axis_dim_mapping(field.data.dims)
    data = field.data

    lenT = 2 if np.any(tau > 0) else 1

    # Spatial coordinates: left if barycentric < 0.5, otherwise right
    zi_1 = np.clip(zi + 1, 0, data.shape[1] - 1)
    zi_full = np.where(zeta < 0.5, zi, zi_1)

    yi_1 = np.clip(yi + 1, 0, data.shape[2] - 1)
    yi_full = np.where(eta < 0.5, yi, yi_1)

    xi_1 = np.clip(xi + 1, 0, data.shape[3] - 1)
    xi_full = np.where(xsi < 0.5, xi, xi_1)

    # Time coordinates: 1 point at ti, then 1 point at ti+1
    if lenT == 1:
        ti_full = ti
    else:
        ti_1 = np.clip(ti + 1, 0, data.shape[0] - 1)
        ti_full = np.concatenate([ti, ti_1])
        xi_full = np.repeat(xi_full, 2)
        yi_full = np.repeat(yi_full, 2)
        zi_full = np.repeat(zi_full, 2)

    # Create DataArrays for indexing
    selection_dict = {
        axis_dim["X"]: xr.DataArray(xi_full, dims=("points")),
        axis_dim["Y"]: xr.DataArray(yi_full, dims=("points")),
    }
    if "Z" in axis_dim:
        selection_dict[axis_dim["Z"]] = xr.DataArray(zi_full, dims=("points"))
    if "time" in data.dims:
        selection_dict["time"] = xr.DataArray(ti_full, dims=("points"))

    corner_data = data.isel(selection_dict).data.reshape(lenT, len(xsi))

    if lenT == 2:
        value = corner_data[0, :] * (1 - tau) + corner_data[1, :] * tau
    else:
        value = corner_data[0, :]

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
