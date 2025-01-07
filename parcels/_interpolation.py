from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


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
        field data of shape (time, z, y, x)
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
    """

    data: np.ndarray
    zeta: float
    eta: float
    xsi: float
    ti: int
    zi: int
    yi: int
    xi: int


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
