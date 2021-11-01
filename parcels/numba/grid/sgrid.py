import numpy as np


class BaseSGrid():
    def search_indices_vertical(self, x, y, z, xi, yi, xsi, eta, ti, time):
        grid = self.grid
        if self.interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer']:
            xsi = 1
            eta = 1
        if time < grid.time[ti]:
            ti -= 1
        if grid.z4d:
            if ti == len(grid.time)-1:
                depth_vector = (1-xsi)*(1-eta) * grid.depth[-1, :, yi, xi] + \
                    xsi*(1-eta) * grid.depth[-1, :, yi, xi+1] + \
                    xsi*eta * grid.depth[-1, :, yi+1, xi+1] + \
                    (1-xsi)*eta * grid.depth[-1, :, yi+1, xi]
            else:
                dv2 = (1-xsi)*(1-eta) * grid.depth[ti:ti+2, :, yi, xi] + \
                    xsi*(1-eta) * grid.depth[ti:ti+2, :, yi, xi+1] + \
                    xsi*eta * grid.depth[ti:ti+2, :, yi+1, xi+1] + \
                    (1-xsi)*eta * grid.depth[ti:ti+2, :, yi+1, xi]
                tt = (time-grid.time[ti]) / (grid.time[ti+1]-grid.time[ti])
                assert tt >= 0 and tt <= 1, 'Vertical s grid is being wrongly interpolated in time'
                depth_vector = dv2[0, :] * (1-tt) + dv2[1, :] * tt
        else:
            depth_vector = (1-xsi)*(1-eta) * grid.depth[:, yi, xi] + \
                xsi*(1-eta) * grid.depth[:, yi, xi+1] + \
                xsi*eta * grid.depth[:, yi+1, xi+1] + \
                (1-xsi)*eta * grid.depth[:, yi+1, xi]
        z = np.float32(z)

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
                self.FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z < depth_vector[zi+1]:
                self.FieldOutOfBoundError(x, y, z, field=self)
        zeta = (z - depth_vector[zi]) / (depth_vector[zi+1]-depth_vector[zi])
        return (zi, zeta)
