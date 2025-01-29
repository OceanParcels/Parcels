"""Handling of Errors and particle status codes"""

__all__ = [
    "AllParcelsErrorCodes",
    "FieldOutOfBoundError",
    "FieldSamplingError",
    "KernelError",
    "StatusCode",
    "TimeExtrapolationError",
    "_raise_field_out_of_bound_error",
    "_raise_field_out_of_bound_surface_error",
    "_raise_field_sampling_error",
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
    ErrorOutOfBounds = 60
    ErrorThroughSurface = 61
    ErrorTimeExtrapolation = 70


class DaskChunkingError(RuntimeError):
    """Error indicating to the user that something with setting up Dask and chunked fieldsets went wrong."""

    pass


class FieldSamplingError(RuntimeError):
    """Utility error class to propagate erroneous field sampling."""

    pass


class FieldOutOfBoundError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling."""

    pass


class FieldOutOfBoundSurfaceError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling at the surface."""

    pass


def _raise_field_sampling_error(z, y, x):
    raise FieldSamplingError(f"Field sampled at (depth={z}, lat={y}, lon={x})")


def _raise_field_out_of_bound_error(z, y, x):
    raise FieldOutOfBoundError(f"Field sampled out-of-bound, at (depth={z}, lat={y}, lon={x})")


def _raise_field_out_of_bound_surface_error(z: float | None, y: float | None, x: float | None) -> None:
    def format_out(val):
        return "unknown" if val is None else val

    raise FieldOutOfBoundSurfaceError(
        f"Field sampled out-of-bound at the surface, at (depth={format_out(z)}, lat={format_out(y)}, lon={format_out(x)})"
    )


class TimeExtrapolationError(RuntimeError):
    """Utility error class to propagate erroneous time extrapolation sampling."""

    def __init__(self, time, field=None):
        if field is not None and field.grid.time_origin and time is not None:
            time = field.grid.time_origin.fulltime(time)
        message = (
            f"{field.name if field else 'Field'} sampled outside time domain at time {time}."
            " Try setting allow_time_extrapolation to True."
        )
        super().__init__(message)


class KernelError(RuntimeError):
    """General particle kernel error with optional custom message."""

    def __init__(self, particle, fieldset=None, msg=None):
        message = (
            f"{particle.state}\n"
            f"Particle {particle}\n"
            f"Time: {_parse_particletime(particle.time, fieldset)}\n"
            f"timestep dt: {particle.dt}\n"
        )
        if msg:
            message += msg
        super().__init__(message)


def _parse_particletime(time, fieldset):
    if fieldset is not None and fieldset.time_origin:
        time = fieldset.time_origin.fulltime(time)
    return time


AllParcelsErrorCodes = {
    FieldSamplingError: StatusCode.Error,
    FieldOutOfBoundError: StatusCode.ErrorOutOfBounds,
    FieldOutOfBoundSurfaceError: StatusCode.ErrorThroughSurface,
    TimeExtrapolationError: StatusCode.ErrorTimeExtrapolation,
    KernelError: StatusCode.Error,
}
