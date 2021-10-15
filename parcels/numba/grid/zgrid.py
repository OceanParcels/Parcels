import numpy as np


class BaseZGrid():
    def search_indices_vertical_z(self, z):
        grid = self.grid
        z = np.float32(z)
        if grid.depth[-1] > grid.depth[0]:
            if z < grid.depth[0]:
                # Since MOM5 is indexed at cell bottom, allow z at depth[0] - dz where dz = (depth[1] - depth[0])
                if self.gridindexingtype == "mom5" and z > 2*grid.depth[0] - grid.depth[1]:
                    return (-1, z / grid.depth[0])
                else:
                    raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z > grid.depth[-1]:
                raise FieldOutOfBoundError(0, 0, z, field=self)
            depth_indices = grid.depth <= z
            if z >= grid.depth[-1]:
                zi = len(grid.depth) - 2
            else:
                zi = depth_indices.argmin() - 1 if z >= grid.depth[0] else 0
        else:
            if z > grid.depth[0]:
                raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z < grid.depth[-1]:
                raise FieldOutOfBoundError(0, 0, z, field=self)
            depth_indices = grid.depth >= z
            if z <= grid.depth[-1]:
                zi = len(grid.depth) - 2
            else:
                zi = depth_indices.argmin() - 1 if z <= grid.depth[0] else 0
        zeta = (z-grid.depth[zi]) / (grid.depth[zi+1]-grid.depth[zi])
        return (zi, zeta)
