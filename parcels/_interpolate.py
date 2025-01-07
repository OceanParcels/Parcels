from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from parcels.tools.statuscodes import (
    _raise_field_out_of_bound_error,
    _raise_field_out_of_bound_surface_error,
)

if TYPE_CHECKING:
    from parcels.field import Field


def search_indices_vertical_z(field: Field, z):
    grid = field.grid
    z = np.float32(z)
    if grid.depth[-1] > grid.depth[0]:
        if z < grid.depth[0]:
            # Since MOM5 is indexed at cell bottom, allow z at depth[0] - dz where dz = (depth[1] - depth[0])
            if field.gridindexingtype == "mom5" and z > 2 * grid.depth[0] - grid.depth[1]:
                return (-1, z / grid.depth[0])
            else:
                _raise_field_out_of_bound_surface_error(z, 0, 0)
        elif z > grid.depth[-1]:
            # In case of CROCO, allow particles in last (uppermost) layer using depth[-1]
            if field.gridindexingtype in ["croco"] and z < 0:
                return (-2, 1)
            _raise_field_out_of_bound_error(z, 0, 0)
        depth_indices = grid.depth <= z
        if z >= grid.depth[-1]:
            zi = len(grid.depth) - 2
        else:
            zi = depth_indices.argmin() - 1 if z >= grid.depth[0] else 0
    else:
        if z > grid.depth[0]:
            _raise_field_out_of_bound_surface_error(z, 0, 0)
        elif z < grid.depth[-1]:
            _raise_field_out_of_bound_error(z, 0, 0)
        depth_indices = grid.depth >= z
        if z <= grid.depth[-1]:
            zi = len(grid.depth) - 2
        else:
            zi = depth_indices.argmin() - 1 if z <= grid.depth[0] else 0
    zeta = (z - grid.depth[zi]) / (grid.depth[zi + 1] - grid.depth[zi])
    return (zi, zeta)


def search_indices_vertical_s(
    field: Field, time: float, z: float, y: float, x: float, ti: int, yi: int, xi: int, eta: float, xsi: float
):
    grid = field.grid
    if field.interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"]:
        xsi = 1
        eta = 1
    if time < grid.time[ti]:
        ti -= 1
    if grid._z4d:
        if ti == len(grid.time) - 1:
            depth_vector = (
                (1 - xsi) * (1 - eta) * grid.depth[-1, :, yi, xi]
                + xsi * (1 - eta) * grid.depth[-1, :, yi, xi + 1]
                + xsi * eta * grid.depth[-1, :, yi + 1, xi + 1]
                + (1 - xsi) * eta * grid.depth[-1, :, yi + 1, xi]
            )
        else:
            dv2 = (
                (1 - xsi) * (1 - eta) * grid.depth[ti : ti + 2, :, yi, xi]
                + xsi * (1 - eta) * grid.depth[ti : ti + 2, :, yi, xi + 1]
                + xsi * eta * grid.depth[ti : ti + 2, :, yi + 1, xi + 1]
                + (1 - xsi) * eta * grid.depth[ti : ti + 2, :, yi + 1, xi]
            )
            tt = (time - grid.time[ti]) / (grid.time[ti + 1] - grid.time[ti])
            assert tt >= 0 and tt <= 1, "Vertical s grid is being wrongly interpolated in time"
            depth_vector = dv2[0, :] * (1 - tt) + dv2[1, :] * tt
    else:
        depth_vector = (
            (1 - xsi) * (1 - eta) * grid.depth[:, yi, xi]
            + xsi * (1 - eta) * grid.depth[:, yi, xi + 1]
            + xsi * eta * grid.depth[:, yi + 1, xi + 1]
            + (1 - xsi) * eta * grid.depth[:, yi + 1, xi]
        )
    z = np.float32(z)  # type: ignore # TODO: remove type ignore once we migrate to float64

    if depth_vector[-1] > depth_vector[0]:
        depth_indices = depth_vector <= z
        if z >= depth_vector[-1]:
            zi = len(depth_vector) - 2
        else:
            zi = depth_indices.argmin() - 1 if z >= depth_vector[0] else 0
        if z < depth_vector[zi]:
            _raise_field_out_of_bound_surface_error(z, 0, 0)
        elif z > depth_vector[zi + 1]:
            _raise_field_out_of_bound_error(z, y, x)
    else:
        depth_indices = depth_vector >= z
        if z <= depth_vector[-1]:
            zi = len(depth_vector) - 2
        else:
            zi = depth_indices.argmin() - 1 if z <= depth_vector[0] else 0
        if z > depth_vector[zi]:
            _raise_field_out_of_bound_surface_error(z, 0, 0)
        elif z < depth_vector[zi + 1]:
            _raise_field_out_of_bound_error(z, y, x)
    zeta = (z - depth_vector[zi]) / (depth_vector[zi + 1] - depth_vector[zi])
    return (zi, zeta)
