import numpy as np
from parcels.numba.grid.base import GridCode
from numba.experimental import jitclass
from numba.core.typing.asnumbatype import as_numba_type
from parcels.numba.grid.rectilinear import RectilinearZGrid
import numba as nb
from numba.core.decorators import njit


@njit
def numba_isclose(a, b):
    return np.absolute(a-b) <= 1e-8 + 1e-5*np.absolute(b)


@jitclass(spec=[
    ("grid", as_numba_type(RectilinearZGrid)),
    ("data", nb.float32[:, :, :]),
    ("interp_method", nb.types.string)
])
class NumbaField():
    def __init__(self, grid, data):
        self.grid = grid
        self.data = data
        self.interp_method = "nearest"

    def interpolator2D(self, ti, z, y, x, particle=None):
        (xsi, eta, _, xi, yi, _) = self.grid.search_indices(x, y, z)
        if self.interp_method == 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            return self.data[ti, yii, xii]
        elif self.interp_method in ['linear', 'bgrid_velocity', 'partialslip', 'freeslip']:
            val = (1-xsi)*(1-eta) * self.data[ti, yi, xi] + \
                xsi*(1-eta) * self.data[ti, yi, xi+1] + \
                xsi*eta * self.data[ti, yi+1, xi+1] + \
                (1-xsi)*eta * self.data[ti, yi+1, xi]
            return val
        elif self.interp_method == 'linear_invdist_land_tracer':
            land = numba_isclose(self.data[ti, yi:yi+2, xi:xi+2], 0.)
            nb_land = np.sum(land)
            if nb_land == 4:
                return 0
            elif nb_land > 0:
                val = 0
                w_sum = 0
                for j in range(2):
                    for i in range(2):
                        distance = pow((eta - j), 2) + pow((xsi - i), 2)
                        if numba_isclose(distance, 0):
                            if land[j][i] == 1:  # index search led us directly onto land
                                return 0
                            else:
                                return self.data[ti, yi+j, xi+i]
                        elif land[i][j] == 0:
                            val += self.data[ti, yi+j, xi+i] / distance
                            w_sum += 1 / distance
                return val / w_sum
            else:
                val = (1 - xsi) * (1 - eta) * self.data[ti, yi, xi] + \
                    xsi * (1 - eta) * self.data[ti, yi, xi + 1] + \
                    xsi * eta * self.data[ti, yi + 1, xi + 1] + \
                    (1 - xsi) * eta * self.data[ti, yi + 1, xi]
                return val
        elif self.interp_method in ['cgrid_tracer', 'bgrid_tracer']:
            return self.data[ti, yi+1, xi+1]
        elif self.interp_method == 'cgrid_velocity':
            # Todo fix "this"
            raise ValueError("This is a scalar field. cgrid_velocity interpolation method should be used for vector fields (e.g. FieldSet.UV)")
        else:
            raise ValueError("Interpolation method is not implemented for 2D grids")

