from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from parcels._typing import GridIndexingType, InterpMethodOption


@dataclass
class InterpolationContext2D:
    """Information provided by Parcels during 2D spatial interpolation. See Delandmeter, P. and van Sebille, E (2019) for more info.

    Attributes
    ----------
    data: np.ndarray
        field data of shape (time, y, x)
    eta: float
        y-direction interpolation coordinate in unit cube (between 0 and 1)
    xsi: float
        x-direction interpolation coordinate in unit cube (between 0 and 1)
    ti: int
        time index
    yi: int
        y index of cell containing particle
    xi: int
        x index of cell containing particle

    """

    data: np.ndarray
    eta: float
    xsi: float
    ti: int
    yi: int
    xi: int


@dataclass
class InterpolationContext3D:
    """Information provided by Parcels during 3D spatial interpolation. See Delandmeter, P. and van Sebille, E (2019) for more info.

    Attributes
    ----------
    data: np.ndarray
        field data of shape (time, z, y, x). This needs to be complete in the vertical
        direction as some interpolation methods need to know whether they are at the
        surface or bottom.
    zeta: float
        vertical interpolation coordinate in unit cube
    eta: float
        y-direction interpolation coordinate in unit cube
    xsi: float
        x-direction interpolation coordinate in unit cube
    zi: int
        z index of cell containing particle
    ti: int
        time index
    yi: int
        y index of cell containing particle
    xi: int
        x index of cell containing particle
    gridindexingtype: GridIndexingType
        grid indexing type

    """

    data: np.ndarray
    zeta: float
    eta: float
    xsi: float
    ti: int
    zi: int
    yi: int
    xi: int
    gridindexingtype: GridIndexingType  #! Needed in 2d as well??
    interp_method: InterpMethodOption  # TODO: Remove during refactoring


interpolator_registry_2d: dict[str, Callable[[InterpolationContext2D], float]] = {}
interpolator_registry_3d: dict[str, Callable[[InterpolationContext3D], float]] = {}


def register_2d_interpolator(name: str):
    def decorator(interpolator: Callable[[InterpolationContext2D], float]):
        interpolator_registry_2d[name] = interpolator
        return interpolator

    return decorator


def register_3d_interpolator(name: str):
    def decorator(interpolator: Callable[[InterpolationContext3D], float]):
        interpolator_registry_3d[name] = interpolator
        return interpolator

    return decorator


@register_2d_interpolator("nearest")
def _nearest_2d(ctx: InterpolationContext2D) -> float:
    xii = ctx.xi if ctx.xsi <= 0.5 else ctx.xi + 1
    yii = ctx.yi if ctx.eta <= 0.5 else ctx.yi + 1
    return ctx.data[ctx.ti, yii, xii]


@register_2d_interpolator("linear")
@register_2d_interpolator("bgrid_velocity")
@register_2d_interpolator("partialslip")
@register_2d_interpolator("freeslip")
def _linear_2d(ctx: InterpolationContext2D) -> float:
    xsi = ctx.xsi
    eta = ctx.eta
    data = ctx.data
    yi = ctx.yi
    xi = ctx.xi
    ti = ctx.ti
    val = (
        (1 - xsi) * (1 - eta) * data[ti, yi, xi]
        + xsi * (1 - eta) * data[ti, yi, xi + 1]
        + xsi * eta * data[ti, yi + 1, xi + 1]
        + (1 - xsi) * eta * data[ti, yi + 1, xi]
    )
    return val


@register_2d_interpolator("linear_invdist_land_tracer")
def _linear_invdist_land_tracer_2d(ctx: InterpolationContext2D) -> float:
    xsi = ctx.xsi
    eta = ctx.eta
    data = ctx.data
    yi = ctx.yi
    xi = ctx.xi
    ti = ctx.ti
    land = np.isclose(data[ti, yi : yi + 2, xi : xi + 2], 0.0)
    nb_land = np.sum(land)

    if nb_land == 4:
        return 0
    elif nb_land > 0:
        val = 0
        w_sum = 0
        for j in range(2):
            for i in range(2):
                distance = pow((eta - j), 2) + pow((xsi - i), 2)
                if np.isclose(distance, 0):
                    if land[j][i] == 1:  # index search led us directly onto land
                        return 0
                    else:
                        return data[ti, yi + j, xi + i]
                elif land[j][i] == 0:
                    val += data[ti, yi + j, xi + i] / distance
                    w_sum += 1 / distance
        return val / w_sum
    else:
        val = (
            (1 - xsi) * (1 - eta) * data[ti, yi, xi]
            + xsi * (1 - eta) * data[ti, yi, xi + 1]
            + xsi * eta * data[ti, yi + 1, xi + 1]
            + (1 - xsi) * eta * data[ti, yi + 1, xi]
        )
        return val


@register_2d_interpolator("cgrid_tracer")
@register_2d_interpolator("bgrid_tracer")
def _tracer_2d(ctx: InterpolationContext2D) -> float:
    return ctx.data[ctx.ti, ctx.yi + 1, ctx.xi + 1]


@register_3d_interpolator("nearest")
def _nearest_3d(ctx: InterpolationContext3D) -> float:
    xii = ctx.xi if ctx.xsi <= 0.5 else ctx.xi + 1
    yii = ctx.yi if ctx.eta <= 0.5 else ctx.yi + 1
    zii = ctx.zi if ctx.zeta <= 0.5 else ctx.zi + 1
    return ctx.data[ctx.ti, zii, yii, xii]


@register_3d_interpolator("cgrid_velocity")
def _cgrid_velocity_3d(ctx: InterpolationContext3D) -> float:
    # evaluating W velocity in c_grid
    if ctx.gridindexingtype == "nemo":
        f0 = ctx.data[ctx.ti, ctx.zi, ctx.yi + 1, ctx.xi + 1]
        f1 = ctx.data[ctx.ti, ctx.zi + 1, ctx.yi + 1, ctx.xi + 1]
    elif ctx.gridindexingtype in ["mitgcm", "croco"]:
        f0 = ctx.data[ctx.ti, ctx.zi, ctx.yi, ctx.xi]
        f1 = ctx.data[ctx.ti, ctx.zi + 1, ctx.yi, ctx.xi]
    return (1 - ctx.zeta) * f0 + ctx.zeta * f1


@register_3d_interpolator("linear_invdist_land_tracer")
def _linear_invdist_land_tracer_3d(ctx: InterpolationContext3D) -> float:
    land = np.isclose(ctx.data[ctx.ti, ctx.zi : ctx.zi + 2, ctx.yi : ctx.yi + 2, ctx.xi : ctx.xi + 2], 0.0)
    nb_land = np.sum(land)
    if nb_land == 8:
        return 0
    elif nb_land > 0:
        val = 0
        w_sum = 0
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    distance = pow((ctx.zeta - k), 2) + pow((ctx.eta - j), 2) + pow((ctx.xsi - i), 2)
                    if np.isclose(distance, 0):
                        if land[k][j][i] == 1:  # index search led us directly onto land
                            return 0
                        else:
                            return ctx.data[ctx.ti, ctx.zi + k, ctx.yi + j, ctx.xi + i]
                    elif land[k][j][i] == 0:
                        val += ctx.data[ctx.ti, ctx.zi + k, ctx.yi + j, ctx.xi + i] / distance
                        w_sum += 1 / distance
        return val / w_sum
    else:
        data = ctx.data[ctx.ti, ctx.zi, :, :]
        f0 = (
            (1 - ctx.xsi) * (1 - ctx.eta) * data[ctx.yi, ctx.xi]
            + ctx.xsi * (1 - ctx.eta) * data[ctx.yi, ctx.xi + 1]
            + ctx.xsi * ctx.eta * data[ctx.yi + 1, ctx.xi + 1]
            + (1 - ctx.xsi) * ctx.eta * data[ctx.yi + 1, ctx.xi]
        )
        data = ctx.data[ctx.ti, ctx.zi + 1, :, :]
        f1 = (
            (1 - ctx.xsi) * (1 - ctx.eta) * data[ctx.yi, ctx.xi]
            + ctx.xsi * (1 - ctx.eta) * data[ctx.yi, ctx.xi + 1]
            + ctx.xsi * ctx.eta * data[ctx.yi + 1, ctx.xi + 1]
            + (1 - ctx.xsi) * ctx.eta * data[ctx.yi + 1, ctx.xi]
        )
        return (1 - ctx.zeta) * f0 + ctx.zeta * f1


@register_3d_interpolator("linear")
@register_3d_interpolator("bgrid_velocity")
@register_3d_interpolator("bgrid_w_velocity")
@register_3d_interpolator("partialslip")
@register_3d_interpolator("freeslip")
def _linear_3d_old(ctx: InterpolationContext3D) -> float:
    zeta = ctx.zeta
    eta = ctx.eta
    xsi = ctx.xsi
    ti = ctx.ti
    zi = ctx.zi
    xi = ctx.xi
    yi = ctx.yi
    if ctx.interp_method == "bgrid_velocity":
        if ctx.gridindexingtype == "mom5":
            zeta = 1.0
        else:
            zeta = 0.0
    elif ctx.interp_method == "bgrid_w_velocity":
        eta = 1.0
        xsi = 1.0
    data = ctx.data[ti, zi, :, :]
    f0 = (
        (1 - xsi) * (1 - eta) * data[yi, xi]
        + xsi * (1 - eta) * data[yi, xi + 1]
        + xsi * eta * data[yi + 1, xi + 1]
        + (1 - xsi) * eta * data[yi + 1, xi]
    )
    if ctx.gridindexingtype == "pop" and zi >= ctx.grid.zdim - 2:
        # Since POP is indexed at cell top, allow linear interpolation of W to zero in lowest cell
        return (1 - zeta) * f0
    data = ctx.data[ti, zi + 1, :, :]
    f1 = (
        (1 - xsi) * (1 - eta) * data[yi, xi]
        + xsi * (1 - eta) * data[yi, xi + 1]
        + xsi * eta * data[yi + 1, xi + 1]
        + (1 - xsi) * eta * data[yi + 1, xi]
    )
    if ctx.interp_method == "bgrid_w_velocity" and ctx.gridindexingtype == "mom5" and zi == -1:
        # Since MOM5 is indexed at cell bottom, allow linear interpolation of W to zero in uppermost cell
        return zeta * f1
    else:
        return (1 - zeta) * f0 + zeta * f1


@register_3d_interpolator("bgrid_tracer")
@register_3d_interpolator("cgrid_tracer")
def _tracer_3d(ctx: InterpolationContext3D) -> float:
    return ctx.data[ctx.ti, ctx.zi, ctx.yi + 1, ctx.xi + 1]
