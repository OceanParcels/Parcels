import numpy as np
from numba.experimental import jitclass
from numba.core.typing.asnumbatype import as_numba_type
from parcels.numba.grid.rectilinear import RectilinearZGrid
from parcels.numba.utils import _numba_isclose
import numba as nb
from numba.core.decorators import njit
import math
from parcels.numba.grid.base import GridCode
from parcels.numba.grid.curvilinear import CurvilinearZGrid, CurvilinearSGrid


def _base_field_spec():
    return [
        ("data", nb.float32[:, :, :, :]),
        ("interp_method", nb.types.string),
        ("gridindexingtype", nb.types.string),
        ("time_periodic", nb.bool_),
        ("allow_time_extrapolation", nb.bool_),
    ]
#     if grid_type == GridCode.CurvilinearZGrid:
#         _base_spec += [("grid", as_numba_type(CurvilinearZGrid))]
#     elif grid_type == GridCode.CurvilinearSGrid:
#         _base_spec += [("grid", as_numba_type(CurvilinearSGrid))]
#     elif grid_type == GridCode.
# @jitclass(spec=[
#     ("grid", as_numba_type(RectilinearZGrid)),
#     ("data", nb.float32[:, :, :, :]),
#     ("interp_method", nb.types.string),
#     ("gridindexingtype", nb.types.string),
#     ("time_periodic", nb.bool_),
#     ("allow_time_extrapolation", nb.bool_),
# ])
#
# NumbaField


class NumbaField():
    def __init__(self, grid, data, interp_method="nearest",
                 gridindexingtype="nemo", time_periodic=True):
        self.grid = grid
        self.data = data
        self.interp_method = interp_method
        self.gridindexingtype = gridindexingtype
        self.time_periodic = time_periodic
        # TODO: add unit conversion.

    def interpolator2D(self, ti, z, y, x, particle=None):
        (xsi, eta, _, xi, yi, _) = self.grid.search_indices(x, y, z, particle=particle)
        if self.interp_method == 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            return self.data[ti, 0, yii, xii]
        elif self.interp_method in ['linear', 'bgrid_velocity', 'partialslip',
                                    'freeslip']:
            val = (1-xsi)*(1-eta) * self.data[ti, 0, yi, xi] + \
                xsi*(1-eta) * self.data[ti, 0, yi, xi+1] + \
                xsi*eta * self.data[ti, 0, yi+1, xi+1] + \
                (1-xsi)*eta * self.data[ti, 0, yi+1, xi]
            return val
        elif self.interp_method == 'linear_invdist_land_tracer':
            land = _numba_isclose(self.data[ti, 0, yi:yi+2, xi:xi+2], 0.)
            nb_land = np.sum(land)
            if nb_land == 4:
                return 0
            elif nb_land > 0:
                val = 0
                w_sum = 0
                for j in range(2):
                    for i in range(2):
                        distance = pow((eta - j), 2) + pow((xsi - i), 2)
                        if _numba_isclose(distance, 0):
                            if land[j][i] == 1:  # index search led us directly onto land
                                return 0
                            else:
                                return self.data[ti, 0, yi+j, xi+i]
                        elif land[i][j] == 0:
                            val += self.data[ti, 0, yi+j, xi+i] / distance
                            w_sum += 1 / distance
                return val / w_sum
            else:
                val = (1 - xsi) * (1 - eta) * self.data[ti, 0, yi, xi] + \
                    xsi * (1 - eta) * self.data[ti, 0, yi, xi + 1] + \
                    xsi * eta * self.data[ti, 0, yi + 1, xi + 1] + \
                    (1 - xsi) * eta * self.data[ti, 0, yi + 1, xi]
                return val
        elif self.interp_method in ['cgrid_tracer', 'bgrid_tracer']:
            return self.data[ti, 0, yi+1, xi+1]
        elif self.interp_method == 'cgrid_velocity':
            # Todo fix "this"
            raise ValueError(
                "This is a scalar field. cgrid_velocity interpolation method "
                "should be used for vector fields (e.g. FieldSet.UV)")
        else:
            raise ValueError("Interpolation method is not implemented for 2D "
                             "grids")

    def interpolator3D(self, ti, z, y, x, time, particle=None):
        (xsi, eta, zeta, xi, yi, zi) = self.grid.search_indices(
            x, y, z, ti, time, particle=particle)
        if self.interp_method == 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            zii = zi if zeta <= .5 else zi+1
            return self.data[ti, zii, yii, xii]
        elif self.interp_method == 'cgrid_velocity':
            # evaluating W velocity in c_grid
            if self.gridindexingtype == 'nemo':
                f0 = self.data[ti, zi, yi+1, xi+1]
                f1 = self.data[ti, zi+1, yi+1, xi+1]
            elif self.gridindexingtype == 'mitgcm':
                f0 = self.data[ti, zi, yi, xi]
                f1 = self.data[ti, zi+1, yi, xi]
            return (1-zeta) * f0 + zeta * f1
        elif self.interp_method == 'linear_invdist_land_tracer':
            land = _numba_isclose(self.data[ti, zi:zi+2, yi:yi+2, xi:xi+2], 0.)
            nb_land = np.sum(land)
            if nb_land == 8:
                return 0
            elif nb_land > 0:
                val = 0
                w_sum = 0
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            distance = pow((zeta - k), 2) + pow((eta - j), 2) + pow((xsi - i), 2)
                            if _numba_isclose(distance, 0):
                                if land[k][j][i] == 1:  # index search led us directly onto land
                                    return 0
                                else:
                                    return self.data[ti, zi+i, yi+j, xi+k]
                            elif land[k][j][i] == 0:
                                val += self.data[ti, zi+k, yi+j, xi+i] / distance
                                w_sum += 1 / distance
                return val / w_sum
            else:
                data = self.data[ti, zi, :, :]
                f0 = (1 - xsi) * (1 - eta) * data[yi, xi] + \
                    xsi * (1 - eta) * data[yi, xi + 1] + \
                    xsi * eta * data[yi + 1, xi + 1] + \
                    (1 - xsi) * eta * data[yi + 1, xi]
                data = self.data[ti, zi + 1, :, :]
                f1 = (1 - xsi) * (1 - eta) * data[yi, xi] + \
                    xsi * (1 - eta) * data[yi, xi + 1] + \
                    xsi * eta * data[yi + 1, xi + 1] + \
                    (1 - xsi) * eta * data[yi + 1, xi]
                return (1 - zeta) * f0 + zeta * f1
        elif self.interp_method in [
            'linear', 'bgrid_velocity', 'bgrid_w_velocity',
                'partialslip', 'freeslip']:
            if self.interp_method == 'bgrid_velocity':
                if self.gridindexingtype == 'mom5':
                    zeta = 1.
                else:
                    zeta = 0.
            elif self.interp_method == 'bgrid_w_velocity':
                eta = 1.
                xsi = 1.
            data = self.data[ti, zi, :, :]
            f0 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            if self.gridindexingtype == 'pop' and zi >= self.grid.zdim-2:
                # Since POP is indexed at cell top, allow linear interpolation
                # of W to zero in lowest cell
                return (1-zeta) * f0
            data = self.data[ti, zi+1, :, :]
            f1 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            if (self.interp_method == 'bgrid_w_velocity'
                    and self.gridindexingtype == 'mom5' and zi == -1):
                # Since MOM5 is indexed at cell bottom, allow linear
                # interpolation of W to zero in uppermost cell
                return zeta * f1
            else:
                return (1-zeta) * f0 + zeta * f1
        elif self.interp_method in ['cgrid_tracer', 'bgrid_tracer']:
            return self.data[ti, zi, yi+1, xi+1]
        else:
            # TODO: fix
            raise RuntimeError("Current interpolation method is not "
                               "implemented for 3D grids")

    def temporal_interpolate_fullfield(self, ti, time):
        """Calculate the data of a field between two snapshots,
        using linear interpolation

        :param ti: Index in time array associated with time (via :func:`time_index`)
        :param time: Time to interpolate to

        :rtype: Linearly interpolated field"""
        t0 = self.grid.time[ti]
        if time == t0:
            return self.data[ti, :]
        elif ti+1 >= len(self.grid.time):
            raise self.TimeExtrapolationError(time, field=self, msg='show_time')
        else:
            t1 = self.grid.time[ti+1]
            f0 = self.data[ti, :]
            f1 = self.data[ti+1, :]
            return f0 + (f1 - f0) * ((time - t0) / (t1 - t0))

    def spatial_interpolation(self, ti, z, y, x, time, particle=None):
        """Interpolate horizontal field values using a SciPy interpolator"""

        if self.data.shape[1] == 1:
            val = self.interpolator2D(ti, z, y, x, particle=particle)
        else:
            val = self.interpolator3D(ti, z, y, x, time, particle=particle)
        if np.isnan(val):
            # Detect Out-of-bounds sampling and raise exception
            self.FieldOutOfBoundError(x, y, z)
        else:
            return val

    def time_index(self, time):
        """Find the index in the time array associated with a given time

        Note that we normalize to either the first or the last index
        if the sampled value is outside the time value range.
        """
        if not self.time_periodic and not self.allow_time_extrapolation and (
                time < self.grid.time[0] or time > self.grid.time[-1]):
            self.TimeExtrapolationError(time)
        time_index = self.grid.time <= time
        if self.time_periodic:
            if time_index.all() or np.logical_not(time_index).all():
                periods = int(math.floor((time-self.grid.time_full[0])/(self.grid.time_full[-1]-self.grid.time_full[0])))
                self.grid.periods = periods
                time -= periods*(self.grid.time_full[-1]-self.grid.time_full[0])
                time_index = self.grid.time <= time
                ti = time_index.argmin() - 1 if time_index.any() else 0
                return (ti, periods)
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)
        if time_index.all():
            # If given time > last known field time, use
            # the last field frame without interpolation
            return (len(self.grid.time) - 1, 0)
        elif np.logical_not(time_index).all():
            # If given time < any time in the field, use
            # the first field frame without interpolation
            return (0, 0)
        else:
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)

    def TimeExtrapolationError(self, time):
        raise Exception()

    def FieldOutOfBoundError(self, x, y, z):
        raise Exception()

    def eval(self, time, z, y, x, particle=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        (ti, periods) = self.time_index(time)
        time -= periods*(self.grid.time_full[-1]-self.grid.time_full[0])
        if ti < self.grid.tdim-1 and time > self.grid.time[ti]:
            f0 = self.spatial_interpolation(ti, z, y, x, time, particle=particle)
            f1 = self.spatial_interpolation(ti + 1, z, y, x, time, particle=particle)
            t0 = self.grid.time[ti]
            t1 = self.grid.time[ti + 1]
            value = f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            # Skip temporal interpolation if time is outside
            # of the defined time range or if we have hit an
            # excat value in the time array.
            value = self.spatial_interpolation(ti, z, y, x, self.grid.time[ti], particle=particle)

        # if applyConversion:
            # return self.units.to_target(value, x, y, z)
        return value

