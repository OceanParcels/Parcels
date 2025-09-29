"""Handling of Errors and particle status codes"""

__all__ = [
    "AllParcelsErrorCodes",
    "FieldOutOfBoundError",
    "FieldSamplingError",
    "KernelError",
    "StatusCode",
    "TimeExtrapolationError",
    "_raise_field_interpolation_error",
    "_raise_field_out_of_bound_error",
    "_raise_field_out_of_bound_surface_error",
    "_raise_general_error",
    "_raise_grid_searching_error",
    "_raise_time_extrapolation_error",
]


class StatusCode:
    """Class defining the status codes for particles.state."""

    Success = 0
    Evaluate = 10
    Repeat = 20
    Delete = 30
    StopExecution = 40
    StopAllExecution = 41
    Error = 50
    ErrorInterpolation = 51
    ErrorGridSearching = 52
    ErrorOutOfBounds = 60
    ErrorThroughSurface = 61
    ErrorTimeExtrapolation = 70


class FieldInterpolationError(RuntimeError):
    """Utility error class to propagate NaN field interpolation."""

    pass


def _raise_field_interpolation_error(z, y, x):
    raise FieldInterpolationError(f"Field interpolation returned NaN at (depth={z}, lat={y}, lon={x})")


class FieldOutOfBoundError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling."""

    pass


def _raise_field_out_of_bound_error(z, y, x):
    raise FieldOutOfBoundError(f"Field sampled out-of-bound, at (depth={z}, lat={y}, lon={x})")


class FieldOutOfBoundSurfaceError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling at the surface."""

    pass


def _raise_field_out_of_bound_surface_error(z: float | None, y: float | None, x: float | None) -> None:
    def format_out(val):
        return "unknown" if val is None else val

    raise FieldOutOfBoundSurfaceError(
        f"Field sampled out-of-bound at the surface, at (depth={format_out(z)}, lat={format_out(y)}, lon={format_out(x)})"
    )


class FieldSamplingError(RuntimeError):
    """Utility error class to propagate field sampling errors."""

    pass


class GridSearchingError(RuntimeError):
    """Utility error class to propagate grid searching errors."""

    pass


def _raise_grid_searching_error(z, y, x):
    raise GridSearchingError(f"Grid searching failed at (depth={z}, lat={y}, lon={x})")


class GeneralError(RuntimeError):
    """Utility error class to propagate general errors."""

    pass


def _raise_general_error(z, y, x):
    raise GeneralError(f"General error occurred at (depth={z}, lat={y}, lon={x})")


class TimeExtrapolationError(RuntimeError):
    """Utility error class to propagate erroneous time extrapolation sampling."""

    def __init__(self, time, field=None):
        message = (
            f"{field.name if field else 'Field'} sampled outside time domain at time {time}."
            " Try setting allow_time_extrapolation to True."
        )
        super().__init__(message)


def _raise_time_extrapolation_error(time: float, field=None):
    raise TimeExtrapolationError(time, field)


class KernelError(RuntimeError):
    """General particles kernel error with optional custom message."""

    def __init__(self, particles, fieldset=None, msg=None):
        message = f"{particles.state}\nParticle {particles}\nTime: {particles.time}\ntimestep dt: {particles.dt}\n"
        if msg:
            message += msg
        super().__init__(message)


AllParcelsErrorCodes = {
    FieldInterpolationError: StatusCode.ErrorInterpolation,
    FieldOutOfBoundError: StatusCode.ErrorOutOfBounds,
    FieldOutOfBoundSurfaceError: StatusCode.ErrorThroughSurface,
    GridSearchingError: StatusCode.ErrorGridSearching,
    TimeExtrapolationError: StatusCode.ErrorTimeExtrapolation,
    KernelError: StatusCode.Error,
    GeneralError: StatusCode.Error,
}
