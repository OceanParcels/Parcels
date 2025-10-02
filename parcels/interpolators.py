"""Collection of pre-built interpolation kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from dask import is_dask_collection

import parcels.utils.interpolation_utils as i_u

if TYPE_CHECKING:
    from parcels._core.field import Field, VectorField
    from parcels._core.uxgrid import _UXGRID_AXES
    from parcels._core.xgrid import _XGRID_AXES

__all__ = [
    "CGrid_Tracer",
    "CGrid_Velocity",
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
    applyConversion: bool,
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
    return value.compute() if is_dask_collection(value) else value


def CGrid_Velocity(
    vectorfield: VectorField,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
    applyConversion: bool,
):
    """
    Interpolation kernel for velocity fields on a C-Grid.
    Following Delandmeter and Van Sebille (2019), velocity fields should be interpolated
    only in the direction of the grid cell faces.
    """
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    U = vectorfield.U.data
    V = vectorfield.V.data
    grid = vectorfield.grid
    tdim, zdim, ydim, xdim = U.shape[0], U.shape[1], U.shape[2], U.shape[3]

    if grid.lon.ndim == 1:
        px = np.array([grid.lon[xi], grid.lon[xi + 1], grid.lon[xi + 1], grid.lon[xi]])
        py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi + 1], grid.lat[yi + 1]])
    else:
        px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
        py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])

    if grid._mesh == "spherical":
        px[0] = np.where(px[0] < x - 225, px[0] + 360, px[0])
        px[0] = np.where(px[0] > x + 225, px[0] - 360, px[0])
        px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
        px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
    c1 = i_u._geodetic_distance(
        py[0], py[1], px[0], px[1], grid._mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(0.0, xsi), py)
    )
    c2 = i_u._geodetic_distance(
        py[1], py[2], px[1], px[2], grid._mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(eta, 1.0), py)
    )
    c3 = i_u._geodetic_distance(
        py[2], py[3], px[2], px[3], grid._mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(1.0, xsi), py)
    )
    c4 = i_u._geodetic_distance(
        py[3], py[0], px[3], px[0], grid._mesh, np.einsum("ij,ji->i", i_u.phi2D_lin(eta, 0.0), py)
    )

    lenT = 2 if np.any(tau > 0) else 1

    # Create arrays of corner points for xarray.isel
    # TODO C grid may not need all xi and yi cornerpoints, so could speed up here?

    # Time coordinates: 4 points at ti, then 4 points at ti+1
    if lenT == 1:
        ti_full = np.repeat(ti, 4)
    else:
        ti_1 = np.clip(ti + 1, 0, tdim - 1)
        ti_full = np.concatenate([np.repeat(ti, 4), np.repeat(ti_1, 4)])

    # Depth coordinates: 4 points at zi, repeated for both time levels
    zi_full = np.repeat(zi, lenT * 4)

    # Y coordinates: [yi, yi, yi+1, yi+1] for each spatial point, repeated for time/depth
    yi_1 = np.clip(yi + 1, 0, ydim - 1)
    yi_full = np.tile(np.repeat(np.column_stack([yi, yi_1]), 2), (lenT))
    # # TODO check why in some cases minus needed here!!!
    # yi_minus_1 = np.clip(yi - 1, 0, ydim - 1)
    # yi = np.tile(np.repeat(np.column_stack([yi_minus_1, yi]), 2), (lenT))

    # X coordinates: [xi, xi+1, xi, xi+1] for each spatial point, repeated for time/depth
    xi_1 = np.clip(xi + 1, 0, xdim - 1)
    xi_full = np.tile(np.column_stack([xi, xi_1, xi, xi_1]).flatten(), (lenT))

    for data in [U, V]:
        axis_dim = grid.get_axis_dim_mapping(data.dims)

        # Create DataArrays for indexing
        selection_dict = {
            axis_dim["X"]: xr.DataArray(xi_full, dims=("points")),
            axis_dim["Y"]: xr.DataArray(yi_full, dims=("points")),
        }
        if "Z" in axis_dim:
            selection_dict[axis_dim["Z"]] = xr.DataArray(zi_full, dims=("points"))
        if "time" in data.dims:
            selection_dict["time"] = xr.DataArray(ti_full, dims=("points"))

        corner_data = data.isel(selection_dict).data.reshape(lenT, len(xsi), 4)

        if lenT == 2:
            tau_full = tau[:, np.newaxis]
            corner_data = corner_data[0, :, :] * (1 - tau_full) + corner_data[1, :, :] * tau_full
        else:
            corner_data = corner_data[0, :, :]
        # # See code below for v3 version
        # # if self.gridindexingtype == "nemo":
        # #     U0 = self.U.data[ti, zi, yi + 1, xi] * c4
        # #     U1 = self.U.data[ti, zi, yi + 1, xi + 1] * c2
        # #     V0 = self.V.data[ti, zi, yi, xi + 1] * c1
        # #     V1 = self.V.data[ti, zi, yi + 1, xi + 1] * c3
        # # elif self.gridindexingtype in ["mitgcm", "croco"]:
        # #     U0 = self.U.data[ti, zi, yi, xi] * c4
        # #     U1 = self.U.data[ti, zi, yi, xi + 1] * c2
        # #     V0 = self.V.data[ti, zi, yi, xi] * c1
        # #     V1 = self.V.data[ti, zi, yi + 1, xi] * c3
        # # TODO Nick can you help use xgcm to fix this implementation?

        # # CROCO and MITgcm grid indexing,
        # if data is U:
        #     U0 = corner_data[:, 0] * c4
        #     U1 = corner_data[:, 1] * c2
        # elif data is V:
        #     V0 = corner_data[:, 0] * c1
        #     V1 = corner_data[:, 2] * c3
        # # NEMO grid indexing
        if data is U:
            U0 = corner_data[:, 2] * c4
            U1 = corner_data[:, 3] * c2
        elif data is V:
            V0 = corner_data[:, 1] * c1
            V1 = corner_data[:, 3] * c3

    U = (1 - xsi) * U0 + xsi * U1
    V = (1 - eta) * V0 + eta * V1

    deg2m = 1852 * 60.0
    if applyConversion:
        meshJac = (deg2m * deg2m * np.cos(np.deg2rad(y))) if grid._mesh == "spherical" else 1
    else:
        meshJac = deg2m if grid._mesh == "spherical" else 1

    jac = i_u._compute_jacobian_determinant(py, px, eta, xsi) * meshJac

    u = (
        (-(1 - eta) * U - (1 - xsi) * V) * px[0]
        + ((1 - eta) * U - xsi * V) * px[1]
        + (eta * U + xsi * V) * px[2]
        + (-eta * U + (1 - xsi) * V) * px[3]
    ) / jac
    v = (
        (-(1 - eta) * U - (1 - xsi) * V) * py[0]
        + ((1 - eta) * U - xsi * V) * py[1]
        + (eta * U + xsi * V) * py[2]
        + (-eta * U + (1 - xsi) * V) * py[3]
    ) / jac
    if is_dask_collection(u):
        u = u.compute()
        v = v.compute()

    # check whether the grid conversion has been applied correctly
    xx = (1 - xsi) * (1 - eta) * px[0] + xsi * (1 - eta) * px[1] + xsi * eta * px[2] + (1 - xsi) * eta * px[3]
    u = np.where(np.abs((xx - x) / x) > 1e-4, np.nan, u)

    if vectorfield.W:
        data = vectorfield.W.data
        # Time coordinates: 2 points at ti, then 2 points at ti+1
        if lenT == 1:
            ti_full = np.repeat(ti, 2)
        else:
            ti_1 = np.clip(ti + 1, 0, tdim - 1)
            ti_full = np.concatenate([np.repeat(ti, 2), np.repeat(ti_1, 2)])

        # Depth coordinates: 1 points at zi, repeated for both time levels
        zi_1 = np.clip(zi + 1, 0, zdim - 1)
        zi_full = np.tile(np.array([zi, zi_1]).flatten(), lenT)

        # Y coordinates: yi+1 for each spatial point, repeated for time/depth
        yi_1 = np.clip(yi + 1, 0, ydim - 1)
        yi_full = np.tile(yi_1, (lenT) * 2)

        # X coordinates: xi+1 for each spatial point, repeated for time/depth
        xi_1 = np.clip(xi + 1, 0, xdim - 1)
        xi_full = np.tile(xi_1, (lenT) * 2)

        axis_dim = grid.get_axis_dim_mapping(data.dims)

        # Create DataArrays for indexing
        selection_dict = {
            axis_dim["X"]: xr.DataArray(xi_full, dims=("points")),
            axis_dim["Y"]: xr.DataArray(yi_full, dims=("points")),
            axis_dim["Z"]: xr.DataArray(zi_full, dims=("points")),
        }
        if "time" in data.dims:
            selection_dict["time"] = xr.DataArray(ti_full, dims=("points"))

        corner_data = data.isel(selection_dict).data.reshape(lenT, 2, len(xsi))

        if lenT == 2:
            tau_full = tau[np.newaxis, :]
            corner_data = corner_data[0, :, :] * (1 - tau_full) + corner_data[1, :, :] * tau_full
        else:
            corner_data = corner_data[0, :, :]

        w = corner_data[0, :] * (1 - zeta) + corner_data[1, :] * zeta
        if is_dask_collection(w):
            w = w.compute()
    else:
        w = np.zeros_like(u)

    return (u, v, w)


def CGrid_Tracer(
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Interpolation kernel for tracer fields on a C-Grid.

    Following Delandmeter and Van Sebille (2019), tracer fields should be interpolated
    constant over the grid cell
    """
    xi, _ = position["X"]
    yi, _ = position["Y"]
    zi, _ = position["Z"]

    axis_dim = field.grid.get_axis_dim_mapping(field.data.dims)
    data = field.data

    lenT = 2 if np.any(tau > 0) else 1

    if lenT == 2:
        ti_1 = np.clip(ti + 1, 0, data.shape[0] - 1)
        ti = np.concatenate([np.repeat(ti), np.repeat(ti_1)])
        zi_1 = np.clip(zi + 1, 0, data.shape[1] - 1)
        zi = np.concatenate([np.repeat(zi), np.repeat(zi_1)])
        yi_1 = np.clip(yi + 1, 0, data.shape[2] - 1)
        yi = np.concatenate([np.repeat(yi), np.repeat(yi_1)])
        xi_1 = np.clip(xi + 1, 0, data.shape[3] - 1)
        xi = np.concatenate([np.repeat(xi), np.repeat(xi_1)])

    # Create DataArrays for indexing
    selection_dict = {
        axis_dim["X"]: xr.DataArray(xi, dims=("points")),
        axis_dim["Y"]: xr.DataArray(yi, dims=("points")),
    }
    if "Z" in axis_dim:
        selection_dict[axis_dim["Z"]] = xr.DataArray(zi, dims=("points"))
    if "time" in field.data.dims:
        selection_dict["time"] = xr.DataArray(ti, dims=("points"))

    value = data.isel(selection_dict).data.reshape(lenT, len(xi))

    if lenT == 2:
        tau = tau[:, np.newaxis]
        value = value[0, :] * (1 - tau) + value[1, :] * tau
    else:
        value = value[0, :]

    return value.compute() if is_dask_collection(value) else value


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
    applyConversion: bool,
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
    applyConversion: bool,
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

    return value.compute() if is_dask_collection(value) else value


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
    node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values
    # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
    # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
    # First, do barycentric interpolation in the lateral direction for each interface level
    fzk = np.sum(field.data.values[ti[:, None], k[:, None], node_ids] * bcoords, axis=-1)
    fzkp1 = np.sum(field.data.values[ti[:, None], k[:, None] + 1, node_ids] * bcoords, axis=-1)

    # Then, do piecewise linear interpolation in the vertical direction
    zk = field.grid.z.values[k]
    zkp1 = field.grid.z.values[k + 1]
    return (fzk * (zkp1 - z) + fzkp1 * (z - zk)) / (zkp1 - zk)  # Linear interpolation in the vertical direction
