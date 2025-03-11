import cftime
import numpy as np
import uxarray as ux
import cftime

from parcels._compat import MPI
from parcels._typing import GridIndexingType, InterpMethodOption, Mesh
from parcels.field import DeferredArray, Field, NestedField, VectorField
from parcels.grid import Grid
from parcels.gridset import GridSet
from parcels.particlefile import ParticleFile
from parcels.tools._helpers import default_repr, fieldset_repr
from parcels.tools.converters import TimeConverter, convert_xarray_time_units
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import TimeExtrapolationError
from parcels.tools.warnings import FieldSetWarning

__all__ = ["UXFieldSet"]

_inside_tol = 1e-6

# class UXVectorField:
#     def __init__(self, name: str, U: ux.UxDataArray, V: ux.UxDataArray, W: ux.UxDataArray | None = None):
#         self.name = name
#         self.U = U
#         self.V = V
#         self.W = W
#         if self.W:
#             self.vector_type = "3D"
#         else:
#             self.vector_type = "2D"

#     def __repr__(self):
#         return f"""<{type(self).__name__}>
#     name: {self.name!r}
#     U: {default_repr(self.U)}
#     V: {default_repr(self.V)}
#     W: {default_repr(self.W)}"""

#     def eval(self, time, z, y, x, particle=None, applyConversion=True):
        
class UXFieldSet:
    """A FieldSet class that holds hydrodynamic data needed to execute particles
    in a UXArray.Dataset"""
    # Change uxds to ds_list - which is a list of either uxDataset or xarray dataset
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
        self._spatialhash = self.uxds.uxgrid.get_spatial_hash()

    #def _validate_uxds(self, uxds: ux.UxDataset):
    #def _validate_xds(self, xds: xr.Dataset): 

    def _check_complete(self):
        assert self.uxds is not None, "UXFieldSet has not been loaded"
        assert self.uxds.u is not None, "UXFieldSet does not provide u velocity data"
        assert self.uxds.v is not None, "UXFieldSet does not provide v velocity data"
        assert self.uxds.uxgrid is not None, "UXFieldSet does not provide a grid"

    def _face_interp(self, field, time, z, y, x, ei):
        ti = 0
        zi = 0
        return field[ti,zi,ei]
    
    def _node_interp(self, field, time, z, y, x, ei):
        """Performs barycentric interpolation of a field at a given location."""        
        ti = 0
        zi = 0
        coords =np.deg2rad([[x, y]])
        n_nodes = self.uxds.uxgrid.n_nodes_per_face[ei].to_numpy()
        node_ids = self.uxds.uxgrid.face_node_connectivity[ei, 0:n_nodes] 
        nodes = np.column_stack(
            (
                np.deg2rad(self.uxds.uxgrid.node_lon[node_ids].to_numpy()),
                np.deg2rad(self.uxds.uxgrid.node_lat[node_ids].to_numpy()),
            )
        )
        bcoord = np.asarray(_barycentric_coordinates(nodes, coords))
        return np.sum(bcoord * field[ti,zi,node_ids].flatten(), axis=0) 

    def get_time_range(self):
        return self.uxds.time.min().to_numpy(), self.uxds.time.max().to_numpy()
    
    def _point_is_in_face(self, y, x, ei):
        "Checks if a point is inside a given face id "
        #ti, zi, fi = self.unravel_index(particle.ei) # Get the time, z, and face index of the particle
        fi = ei

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
        bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
        if ( not (bcoord >= 0).all() ) and (not (bcoord <= 1.0).all()):
            return False
        
        return True

    def eval(self, field_names, time, z, y, x, ei: int=None, applyConversion=True):
        
        res = {}

        if ei is not None:
            fi = ei
            if not self._point_is_in_face(y,x,ei):
                # If the point is not in the previously defined face, then
                # search for the face again.
                # To do : Update the search here to do nearest neighbors search, rather than spatial hash - joe@fluidnumerics.com
                print(f"Position : {x}, {y}")
                print(f"Hash indices : {self._spatialhash._hash_index2d(np.deg2rad([[x,y]]))}")
                fi, bcoords = self._spatialhash.query([[x,y]]) # Get the face id for the particle  
                fi = fi[0] 
                print(f"Face index (updated): {fi}")
                print(f"Barycentric coordinates (updated): {bcoords}")

        for f in field_names:
            field = getattr(self.uxds, f)
            face_registered = ("n_face" in field.dims)

            if face_registered:
                r = self._face_interp(field, time, z, y, x, fi)
            else:
                r = self._node_interp(field, time, z, y, x, fi)

            #if applyConversion:
            #    res[f] = self.units.to_target(r, z, y, x)
            #else:
            # To do : Add call to units.to_target to handle unit conversion : joe@fluidnumerics.com
            res[f] = r/111111.111111111
            
        return res, fi
            
def _barycentric_coordinates(nodes, point):
    """
    Compute the barycentric coordinates of a point P inside a convex polygon using area-based weights.
    So that this method generalizes to n-sided polygons, we use the Waschpress points as the generalized
    barycentric coordinates, which is only valid for convex polygons.

    Parameters
    ----------
        nodes : numpy.ndarray
            Spherical coordinates (lon,lat) of each corner node of a face
        point : numpy.ndarray
            Spherical coordinates (lon,lat) of the point
    Returns
    -------
    numpy.ndarray
        Barycentric coordinates corresponding to each vertex.

    """
    n = len(nodes)
    sum_wi = 0
    w = []

    for i in range(0, n):
        vim1 = nodes[i - 1]
        vi = nodes[i]
        vi1 = nodes[(i + 1) % n]
        a0 = _triangle_area(vim1, vi, vi1)
        a1 = _triangle_area(point, vim1, vi)
        a2 = _triangle_area(point, vi, vi1)
        sum_wi += a0 / (a1 * a2)
        w.append(a0 / (a1 * a2))

    barycentric_coords = [w_i / sum_wi for w_i in w]

    return barycentric_coords

def _triangle_area(A, B, C):
    """
    Compute the area of a triangle given by three points.
    """
    return 0.5 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))