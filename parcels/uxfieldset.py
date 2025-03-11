import cftime
import numpy as np
import uxarray as ux
from uxarray.neighbors import _barycentric_coordinates

__all__ = ["UXFieldSet"]

_inside_tol = 1e-6


class UXFieldSet:
    """A FieldSet class that holds hydrodynamic data needed to execute particles
    in a UXArray.Dataset
    """

    def __init__(self, uxds: ux.UxDataset, time_origin: float | np.datetime64 | np.timedelta64 | cftime.datetime = 0):
        # Ensure that dataset provides a grid, and the u and v velocity
        # components at a minimum
        if not hasattr(uxds, "uxgrid"):
            raise ValueError("The UXArray dataset does not provide a grid")
        if not hasattr(uxds, "u"):
            raise ValueError("The UXArray dataset does not provide u velocity data")
        if not hasattr(uxds, "v"):
            raise ValueError("The UXArray dataset does not provide v velocity data")

        self.time_origin = time_origin
        self.uxds = uxds
        self._spatialhash = self.uxds.get_spatialhash()

    def _check_complete(self):
        assert self.uxds is not None, "UXFieldSet has not been loaded"
        assert self.uxds.u is not None, "UXFieldSet does not provide u velocity data"
        assert self.uxds.v is not None, "UXFieldSet does not provide v velocity data"
        assert self.uxds.uxgrid is not None, "UXFieldSet does not provide a grid"

    def _face_interp(self, field, time, z, y, x, particle=None):
        # ti, zi, fi = self.unravel_index(particle.ei) # Get the time, z, and face index of the particle
        ti = 0
        zi = 0
        fi = particle.ei
        return field[ti, zi, fi]

    def _node_interp(self, field, time, z, y, x, particle=None):
        """Performs barycentric interpolation of a field at a given location."""
        # ti, zi, fi = self.unravel_index(particle.ei) # Get the time, z, and face index of the particle
        ti = 0
        zi = 0
        fi = particle.ei
        # Check if particle is in the same face, otherwise search again.
        n_nodes = self.uxds.uxgrid.n_nodes_per_face[fi].to_numpy()
        node_ids = self.uxds.uxgrid.face_node_connectivity[fi, 0:n_nodes]
        nodes = np.column_stack(
            (
                np.deg2rad(self.uxds.uxgrid.node_lon[node_ids].to_numpy()),
                np.deg2rad(self.uxds.uxgrid.node_lat[node_ids].to_numpy()),
            )
        )

        coord = np.deg2rad([x, y])
        bcoord = _barycentric_coordinates(nodes, coord)
        return np.sum(bcoord * field[ti, zi, node_ids].flatten(), axis=0)

    def eval(self, field_names: list(str), time, z, y, x, particle=None, applyConversion=True):
        res = {}
        if particle:
            # ti, zi, fi = self.unravel_index(particle.ei) # Get the time, z, and face index of the particle
            fi = particle.ei
            # Check if particle is in the same face, otherwise search again.
            n_nodes = self.uxds.uxgrid.n_nodes_per_face[fi].to_numpy()
            node_ids = self.uxds.uxgrid.face_node_connectivity[fi, 0:n_nodes]
            nodes = np.column_stack(
                (
                    np.deg2rad(self.uxds.uxgrid.node_lon[node_ids].to_numpy()),
                    np.deg2rad(self.uxds.uxgrid.node_lat[node_ids].to_numpy()),
                )
            )

            coord = np.deg2rad([x, y])
            bcoord = _barycentric_coordinates(nodes, coord)
            err = np.abs(np.sum(bcoord * nodes[:, 0], axis=0) - coord[0])
            +np.abs(np.sum(bcoord * nodes[:, 1], axis=0) - coord[1])
            is_inside = all(lambda_i >= 0 for lambda_i in bcoord)

            # To do : Get the vertical and time indices for the particle

            if (not is_inside) or (err > _inside_tol):
                fi = self._spatialhash.query([particle.x, particle.y])  # Get the face id for the particle
                particle.ei = fi

        for f in field_names:
            field = getattr(self, f)
            face_registered = "n_face" in field.dims
            if face_registered:
                if particle:
                    r = self._face_interp(field, particle.time, particle.z, particle.y, particle.x, particle)
                else:
                    res = self._face_interp(field, particle.time, z, y, x)
            else:
                if particle:
                    r = self._node_interp(field, time, particle.z, particle.y, particle.x, particle)
                else:
                    r = self._node_interp(field, time, z, y, x)

            if applyConversion:
                res[f] = self.units.to_target(r, z, y, x)
            else:
                res[f] = r

        return res

        # if self.U.interp_method not in ["cgrid_velocity", "partialslip", "freeslip"]:
        #     u = self.U.eval(time, z, y, x, particle=particle, applyConversion=False)
        #     v = self.V.eval(time, z, y, x, particle=particle, applyConversion=False)
        #     if applyConversion:
        #         u = self.U.units.to_target(u, z, y, x)
        #         v = self.V.units.to_target(v, z, y, x)
        #     if "3D" in self.vector_type:
        #         w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
        #         if applyConversion:
        #             w = self.W.units.to_target(w, z, y, x)
        #         return (u, v, w)
        #     else:
        #         return (u, v)
        # else:
        #     interp = {
        #         "cgrid_velocity": {
        #             "2D": self.spatial_c_grid_interpolation2D,
        #             "3D": self.spatial_c_grid_interpolation3D,
        #         },
        #         "partialslip": {"2D": self.spatial_slip_interpolation, "3D": self.spatial_slip_interpolation},
        #         "freeslip": {"2D": self.spatial_slip_interpolation, "3D": self.spatial_slip_interpolation},
        #     }
        #     grid = self.U.grid
        #     ti = self.U._time_index(time)
        #     if ti < grid.tdim - 1 and time > grid.time[ti]:
        #         t0 = grid.time[ti]
        #         t1 = grid.time[ti + 1]
        #         if "3D" in self.vector_type:
        #             (u0, v0, w0) = interp[self.U.interp_method]["3D"](
        #                 ti, z, y, x, time, particle=particle, applyConversion=applyConversion
        #             )
        #             (u1, v1, w1) = interp[self.U.interp_method]["3D"](
        #                 ti + 1, z, y, x, time, particle=particle, applyConversion=applyConversion
        #             )
        #             w = w0 + (w1 - w0) * ((time - t0) / (t1 - t0))
        #         else:
        #             (u0, v0) = interp[self.U.interp_method]["2D"](
        #                 ti, z, y, x, time, particle=particle, applyConversion=applyConversion
        #             )
        #             (u1, v1) = interp[self.U.interp_method]["2D"](
        #                 ti + 1, z, y, x, time, particle=particle, applyConversion=applyConversion
        #             )
        #         u = u0 + (u1 - u0) * ((time - t0) / (t1 - t0))
        #         v = v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
        #         if "3D" in self.vector_type:
        #             return (u, v, w)
        #         else:
        #             return (u, v)
        #     else:
        #         # Skip temporal interpolation if time is outside
        #         # of the defined time range or if we have hit an
        #         # exact value in the time array.
        #         if "3D" in self.vector_type:
        #             return interp[self.U.interp_method]["3D"](
        #                 ti, z, y, x, grid.time[ti], particle=particle, applyConversion=applyConversion
        #             )
        #         else:
        #             return interp[self.U.interp_method]["2D"](
        #                 ti, z, y, x, grid.time[ti], particle=particle, applyConversion=applyConversion
        #             )
