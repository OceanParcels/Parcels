import numpy as np
import numba as nb


class BaseZGrid():
    def search_indices_vertical(self, _x, _y, z, _xi, _yi, _xsi, _eta, _ti,
                                _time, interp_method):
        # TODO: fix 
        gridindexingtype = "unknown"
        z = np.float32(z)
        if self.depth[-1] > self.depth[0]:
            if z < self.depth[0]:
                # Since MOM5 is indexed at cell bottom, allow z at depth[0] - dz where dz = (depth[1] - depth[0])
                if gridindexingtype == "mom5" and z > 2*self.depth[0] - self.depth[1]:
                    return (-1, z / self.depth[0])
                else:
                    self.FieldOutOfBoundSurfaceError(0, 0, z)
            elif z > self.depth[-1]:
                self.FieldOutOfBoundError(0, 0, z)
            depth_indices = self.depth <= z
            if z >= self.depth[-1]:
                zi = len(self.depth) - 2
            else:
                zi = depth_indices.argmin() - 1 if z >= self.depth[0] else 0
        else:
            if z > self.depth[0]:
                self.FieldOutOfBoundSurfaceError(0, 0, z)
            elif z < self.depth[-1]:
                self.FieldOutOfBoundError(0, 0, z)
            depth_indices = self.depth >= z
            if z <= self.depth[-1]:
                zi = len(self.depth) - 2
            else:
                zi = depth_indices.argmin() - 1 if z <= self.depth[0] else 0
        zeta = (z-self.depth[zi]) / (self.depth[zi+1]-self.depth[zi])
        return (zi, zeta)

    def get_pz(self, _xi, _yi, zi):
        return np.array([self.depth[zi], self.depth[zi+1]]).astype(nb.float64)
