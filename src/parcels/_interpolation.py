from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from parcels._typing import GridIndexingType
from parcels.utils._helpers import should_calculate_next_ti


@dataclass
class InterpolationContext2D:
    """Information provided by Parcels during 2D spatial interpolation. See Delandmeter and Van Sebille (2019), 10.5194/gmd-12-3571-2019 for more info.

    Attributes
    ----------
    data: np.ndarray
        field data of shape (time, y, x)
    tau: float
        time interpolation coordinate in unit length
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
    tau: float
    eta: float
    xsi: float
    ti: int
    yi: int
    xi: int


@dataclass
class InterpolationContext3D:
    """Information provided by Parcels during 3D spatial interpolation. See Delandmeter and Van Sebille (2019), 10.5194/gmd-12-3571-2019 for more info.

    Attributes
    ----------
    data: np.ndarray
        field data of shape (time, z, y, x). This needs to be complete in the vertical
        direction as some interpolation methods need to know whether they are at the
        surface or bottom.
    tau: float
        time interpolation coordinate in unit length
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
    tau: float
    zeta: float
    eta: float
    xsi: float
    ti: int
    zi: int
    yi: int
    xi: int
    gridindexingtype: GridIndexingType  # included in 3D as z-face is indexed differently with MOM5 and POP


_interpolator_registry_2d: dict[str, Callable[[InterpolationContext2D], float]] = {}
_interpolator_registry_3d: dict[str, Callable[[InterpolationContext3D], float]] = {}


def get_2d_interpolator_registry() -> Mapping[str, Callable[[InterpolationContext2D], float]]:
    # See Discussion on Python Discord for more context (function prevents re-alias of global variable)
    # _interpolator_registry_2d etc shouldn't be imported directly
    # https://discord.com/channels/267624335836053506/1329136004459794483
    return _interpolator_registry_2d


def get_3d_interpolator_registry() -> Mapping[str, Callable[[InterpolationContext3D], float]]:
    return _interpolator_registry_3d


def register_2d_interpolator(name: str):
    def decorator(interpolator: Callable[[InterpolationContext2D], float]):
        _interpolator_registry_2d[name] = interpolator
        return interpolator

    return decorator


def register_3d_interpolator(name: str):
    def decorator(interpolator: Callable[[InterpolationContext3D], float]):
        _interpolator_registry_3d[name] = interpolator
        return interpolator

    return decorator


@register_2d_interpolator("nearest")
def _nearest_2d(ctx: InterpolationContext2D) -> float:
    xii = ctx.xi if ctx.xsi <= 0.5 else ctx.xi + 1
    yii = ctx.yi if ctx.eta <= 0.5 else ctx.yi + 1
    ft0 = ctx.data[ctx.ti, yii, xii]
    if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
        return ft0
    ft1 = ctx.data[ctx.ti + 1, yii, xii]
    return (1 - ctx.tau) * ft0 + ctx.tau * ft1


def _interp_on_unit_square(*, eta: float, xsi: float, data: np.ndarray, yi: int, xi: int) -> float:
    """Interpolation on a unit square. See Delandmeter and Van Sebille (2019), 10.5194/gmd-12-3571-2019."""
    return (
        (1 - xsi) * (1 - eta) * data[yi, xi]
        + xsi * (1 - eta) * data[yi, xi + 1]
        + xsi * eta * data[yi + 1, xi + 1]
        + (1 - xsi) * eta * data[yi + 1, xi]
    )


@register_2d_interpolator("linear")
@register_2d_interpolator("bgrid_velocity")
@register_2d_interpolator("partialslip")
@register_2d_interpolator("freeslip")
def _linear_2d(ctx: InterpolationContext2D) -> float:
    ft0 = _interp_on_unit_square(eta=ctx.eta, xsi=ctx.xsi, data=ctx.data[ctx.ti, :, :], yi=ctx.yi, xi=ctx.xi)
    if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
        return ft0
    ft1 = _interp_on_unit_square(eta=ctx.eta, xsi=ctx.xsi, data=ctx.data[ctx.ti + 1, :, :], yi=ctx.yi, xi=ctx.xi)
    return (1 - ctx.tau) * ft0 + ctx.tau * ft1


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

    def _get_data_temporalinterp(*, ti, yi, xi):
        dt0 = data[ti, yi, xi]
        if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
            return dt0
        dt1 = data[ti + 1, yi, xi]
        return (1 - ctx.tau) * dt0 + ctx.tau * dt1

    if nb_land == 4:
        return 0
    elif nb_land > 0:
        val = 0
        w_sum = 0.0
        for j in range(2):
            for i in range(2):
                distance = pow((eta - j), 2) + pow((xsi - i), 2)
                if np.isclose(distance, 0):
                    if land[j][i] == 1:  # index search led us directly onto land
                        return 0
                    else:
                        return _get_data_temporalinterp(ti=ti, yi=yi + j, xi=xi + i)
                elif land[j][i] == 0:
                    val += _get_data_temporalinterp(ti=ti, yi=yi + j, xi=xi + i) / distance
                    w_sum += 1 / distance
        return val / w_sum
    else:
        return _interp_on_unit_square(eta=eta, xsi=xsi, data=data[ti, :, :], yi=yi, xi=xi)


@register_2d_interpolator("cgrid_tracer")
@register_2d_interpolator("bgrid_tracer")
def _tracer_2d(ctx: InterpolationContext2D) -> float:
    ft0 = ctx.data[ctx.ti, ctx.yi + 1, ctx.xi + 1]
    if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
        return ft0
    ft1 = ctx.data[ctx.ti + 1, ctx.yi + 1, ctx.xi + 1]
    return (1 - ctx.tau) * ft0 + ctx.tau * ft1


@register_3d_interpolator("nearest")
def _nearest_3d(ctx: InterpolationContext3D) -> float:
    xii = ctx.xi if ctx.xsi <= 0.5 else ctx.xi + 1
    yii = ctx.yi if ctx.eta <= 0.5 else ctx.yi + 1
    zii = ctx.zi if ctx.zeta <= 0.5 else ctx.zi + 1
    ft0 = ctx.data[ctx.ti, zii, yii, xii]
    if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
        return ft0
    ft1 = ctx.data[ctx.ti + 1, zii, yii, xii]
    return (1 - ctx.tau) * ft0 + ctx.tau * ft1


def _get_cgrid_depth_point(*, zeta: float, data: np.ndarray, zi: int, yi: int, xi: int) -> float:
    f0 = data[zi, yi, xi]
    f1 = data[zi + 1, yi, xi]
    return (1 - zeta) * f0 + zeta * f1


@register_3d_interpolator("cgrid_velocity")
def _cgrid_W_velocity_3d(ctx: InterpolationContext3D) -> float:
    # evaluating W velocity in c_grid
    if ctx.gridindexingtype == "nemo":
        ft0 = _get_cgrid_depth_point(
            zeta=ctx.zeta, data=ctx.data[ctx.ti, :, :, :], zi=ctx.zi, yi=ctx.yi + 1, xi=ctx.xi + 1
        )
    elif ctx.gridindexingtype in ["mitgcm", "croco"]:
        ft0 = _get_cgrid_depth_point(zeta=ctx.zeta, data=ctx.data[ctx.ti, :, :, :], zi=ctx.zi, yi=ctx.yi, xi=ctx.xi)
    if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
        return ft0

    if ctx.gridindexingtype == "nemo":
        ft1 = _get_cgrid_depth_point(
            zeta=ctx.zeta, data=ctx.data[ctx.ti + 1, :, :, :], zi=ctx.zi, yi=ctx.yi + 1, xi=ctx.xi + 1
        )
    elif ctx.gridindexingtype in ["mitgcm", "croco"]:
        ft1 = _get_cgrid_depth_point(zeta=ctx.zeta, data=ctx.data[ctx.ti + 1, :, :, :], zi=ctx.zi, yi=ctx.yi, xi=ctx.xi)
    return (1 - ctx.tau) * ft0 + ctx.tau * ft1


@register_3d_interpolator("linear_invdist_land_tracer")
def _linear_invdist_land_tracer_3d(ctx: InterpolationContext3D) -> float:
    land = np.isclose(ctx.data[ctx.ti, ctx.zi : ctx.zi + 2, ctx.yi : ctx.yi + 2, ctx.xi : ctx.xi + 2], 0.0)
    nb_land = np.sum(land)

    def _get_data_temporalinterp(*, ti, zi, yi, xi):
        dt0 = ctx.data[ti, zi, yi, xi]
        if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
            return dt0
        dt1 = data[ti + 1, zi, yi, xi]
        return (1 - ctx.tau) * dt0 + ctx.tau * dt1

    if nb_land == 8:
        return 0
    elif nb_land > 0:
        val = 0.0
        w_sum = 0.0
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    distance = pow((ctx.zeta - k), 2) + pow((ctx.eta - j), 2) + pow((ctx.xsi - i), 2)
                    if np.isclose(distance, 0):
                        if land[k][j][i] == 1:  # index search led us directly onto land
                            return 0
                        else:
                            return _get_data_temporalinterp(ti=ctx.ti, zi=ctx.zi + k, yi=ctx.yi + j, xi=ctx.xi + i)
                    elif land[k][j][i] == 0:
                        val += (
                            _get_data_temporalinterp(ti=ctx.ti, zi=ctx.zi + k, yi=ctx.yi + j, xi=ctx.xi + i) / distance
                        )
                        w_sum += 1 / distance
        return val / w_sum
    else:
        data = ctx.data[ctx.ti, ctx.zi, :, :]
        f0 = _interp_on_unit_square(eta=ctx.eta, xsi=ctx.xsi, data=data, yi=ctx.yi, xi=ctx.xi)

        data = ctx.data[ctx.ti, ctx.zi + 1, :, :]
        f1 = _interp_on_unit_square(eta=ctx.eta, xsi=ctx.xsi, data=data, yi=ctx.yi, xi=ctx.xi)

        return (1 - ctx.zeta) * f0 + ctx.zeta * f1


def _get_3d_f0_f1(*, eta: float, xsi: float, data: np.ndarray, zi: int, yi: int, xi: int) -> tuple[float, float | None]:
    data_2d = data[zi, :, :]
    f0 = _interp_on_unit_square(eta=eta, xsi=xsi, data=data_2d, yi=yi, xi=xi)
    try:
        data_2d = data[zi + 1, :, :]
    except IndexError:
        f1 = None  # POP indexing at edge of domain
    else:
        f1 = _interp_on_unit_square(eta=eta, xsi=xsi, data=data_2d, yi=yi, xi=xi)

    return f0, f1


def _z_layer_interp(
    *, zeta: float, f0: float, f1: float | None, zi: int, zdim: int, gridindexingtype: GridIndexingType
):
    if gridindexingtype == "pop" and zi >= zdim - 2:
        # Since POP is indexed at cell top, allow linear interpolation of W to zero in lowest cell
        return (1 - zeta) * f0
    assert f1 is not None, "f1 should not be None for gridindexingtype != 'pop'"
    if gridindexingtype == "mom5" and zi == -1:
        # Since MOM5 is indexed at cell bottom, allow linear interpolation of W to zero in uppermost cell
        return zeta * f1
    return (1 - zeta) * f0 + zeta * f1


@register_3d_interpolator("linear")
@register_3d_interpolator("partialslip")
@register_3d_interpolator("freeslip")
def _linear_3d(ctx: InterpolationContext3D) -> float:
    zdim = ctx.data.shape[1]
    data_3d = ctx.data[ctx.ti, :, :, :]
    fz0, fz1 = _get_3d_f0_f1(eta=ctx.eta, xsi=ctx.xsi, data=data_3d, zi=ctx.zi, yi=ctx.yi, xi=ctx.xi)
    if should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
        data_3d = ctx.data[ctx.ti + 1, :, :, :]
        fz0_t1, fz1_t1 = _get_3d_f0_f1(eta=ctx.eta, xsi=ctx.xsi, data=data_3d, zi=ctx.zi, yi=ctx.yi, xi=ctx.xi)
        fz0 = (1 - ctx.tau) * fz0 + ctx.tau * fz0_t1
        if fz1_t1 is not None and fz1 is not None:
            fz1 = (1 - ctx.tau) * fz1 + ctx.tau * fz1_t1

    return _z_layer_interp(zeta=ctx.zeta, f0=fz0, f1=fz1, zi=ctx.zi, zdim=zdim, gridindexingtype=ctx.gridindexingtype)


@register_3d_interpolator("bgrid_velocity")
def _linear_3d_bgrid_velocity(ctx: InterpolationContext3D) -> float:
    if ctx.gridindexingtype == "mom5":
        ctx.zeta = 1.0
    else:
        ctx.zeta = 0.0
    return _linear_3d(ctx)


@register_3d_interpolator("bgrid_w_velocity")
def _linear_3d_bgrid_w_velocity(ctx: InterpolationContext3D) -> float:
    ctx.eta = 1.0
    ctx.xsi = 1.0
    return _linear_3d(ctx)


@register_3d_interpolator("bgrid_tracer")
@register_3d_interpolator("cgrid_tracer")
def _tracer_3d(ctx: InterpolationContext3D) -> float:
    ft0 = ctx.data[ctx.ti, ctx.zi, ctx.yi + 1, ctx.xi + 1]
    if not should_calculate_next_ti(ctx.ti, ctx.tau, ctx.data.shape[0]):
        return ft0
    ft1 = ctx.data[ctx.ti + 1, ctx.zi, ctx.yi + 1, ctx.xi + 1]
    return (1 - ctx.tau) * ft0 + ctx.tau * ft1
