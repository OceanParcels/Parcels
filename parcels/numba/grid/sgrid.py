import numpy as np
import numba as nb


class BaseSGrid():
    """Base class for S-grids"""
    def search_indices_vertical(self, x, y, z, xi, yi, xsi, eta, ti, time, interp_method):
        if interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer']:
            xsi = 1
            eta = 1
        if time < self.time[ti]:
            ti -= 1
        if self.z4d:
            if ti == len(self.time)-1:
                depth_vector = (1-xsi)*(1-eta) * self.depth[-1, :, yi, xi] + \
                    xsi*(1-eta) * self.depth[-1, :, yi, xi+1] + \
                    xsi*eta * self.depth[-1, :, yi+1, xi+1] + \
                    (1-xsi)*eta * self.depth[-1, :, yi+1, xi]
            else:
                dv2 = (1-xsi)*(1-eta) * self.depth[ti:ti+2, :, yi, xi] + \
                    xsi*(1-eta) * self.depth[ti:ti+2, :, yi, xi+1] + \
                    xsi*eta * self.depth[ti:ti+2, :, yi+1, xi+1] + \
                    (1-xsi)*eta * self.depth[ti:ti+2, :, yi+1, xi]
                tt = (time-self.time[ti]) / (self.time[ti+1]-self.time[ti])
                assert tt >= 0 and tt <= 1, 'Vertical s grid is being wrongly interpolated in time'
                depth_vector = dv2[0, :] * (1-tt) + dv2[1, :] * tt
        else:
            depth_vector = (1-xsi)*(1-eta) * self.depth[0, :, yi, xi] + \
                xsi*(1-eta) * self.depth[0, :, yi, xi+1] + \
                xsi*eta * self.depth[0, :, yi+1, xi+1] + \
                (1-xsi)*eta * self.depth[0, :, yi+1, xi]
        z = nb.float32(z)

        if depth_vector[-1] > depth_vector[0]:
            depth_indices = depth_vector <= z
            if z >= depth_vector[-1]:
                zi = len(depth_vector) - 2
            else:
                zi = depth_indices.argmin() - 1 if z >= depth_vector[0] else 0
            if z < depth_vector[zi]:

                self.FieldOutOfBoundSurfaceError(0, 0, z)
            elif z > depth_vector[zi+1]:
                self.FieldOutOfBoundError(x, y, z)
        else:
            depth_indices = depth_vector >= z
            if z <= depth_vector[-1]:
                zi = len(depth_vector) - 2
            else:
                zi = depth_indices.argmin() - 1 if z <= depth_vector[0] else 0
            if z > depth_vector[zi]:
                self.FieldOutOfBoundSurfaceError(0, 0, z)
            elif z < depth_vector[zi+1]:
                self.FieldOutOfBoundError(x, y, z)
        zeta = (z - depth_vector[zi]) / (depth_vector[zi+1]-depth_vector[zi])
        return (zi, zeta)

    def get_pz(self, xi, yi, zi):
        return np.array([
            self.depth[0, zi, yi, xi], self.depth[0, zi, yi, xi+1],
            self.depth[0, zi, yi+1, xi+1], self.depth[0, zi, yi+1, xi],
            self.depth[0, zi+1, yi, xi], self.depth[0, zi+1, yi, xi+1],
            self.depth[0, zi+1, yi+1, xi+1],
            self.depth[0, zi+1, yi+1, xi]]).astype(nb.float64)
