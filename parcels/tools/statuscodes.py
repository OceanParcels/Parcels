"""Handling of Errors and particle status codes"""


__all__ = ['StatusCode', 'FieldSamplingError', 'FieldOutOfBoundError', 'TimeExtrapolationError',
           'KernelError', 'AllParcelsErrorCodes']


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

    def __init__(self, src_class_type, message):
        msg = f"[{str(src_class_type)}]: {message}"
        super().__init__(msg)


class FieldSamplingError(RuntimeError):
    """Utility error class to propagate erroneous field sampling."""

    def __init__(self, x, y, z, field=None):
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        message = f"{field.name if field else 'Field'} sampled at ({self.x}, {self.y}, {self.z})"
        super().__init__(message)


class FieldOutOfBoundError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling."""

    def __init__(self, x, y, z, field=None):
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        message = f"{field.name if field else 'Field'} sampled out-of-bound, at ({self.x}, {self.y}, {self.z})"
        super().__init__(message)


class FieldOutOfBoundSurfaceError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling at the surface."""

    def __init__(self, x, y, z, field=None):
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        message = f"{field.name if field else 'Field'} sampled out-of-bound at the surface, at ({self.x}, {self.y}, {self.z})"
        super().__init__(message)


class TimeExtrapolationError(RuntimeError):
    """Utility error class to propagate erroneous time extrapolation sampling."""

    def __init__(self, time, field=None, msg='allow_time_extrapoltion'):
        if field is not None and field.grid.time_origin and time is not None:
            time = field.grid.time_origin.fulltime(time)
        message = f"{field.name if field else 'Field'} sampled outside time domain at time {time}."
        if msg == 'allow_time_extrapoltion':
            message += " Try setting allow_time_extrapolation to True"
        elif msg == 'show_time':
            message += " Try explicitly providing a 'show_time'"
        else:
            message += msg + " Try setting allow_time_extrapolation to True"
        super().__init__(message)


class KernelError(RuntimeError):
    """General particle kernel error with optional custom message."""

    def __init__(self, particle, fieldset=None, msg=None):
        message = (f"{particle.state}\n"
                   f"Particle {particle}\n"
                   f"Time: {parse_particletime(particle.time, fieldset)}\n"
                   f"timestep dt: {particle.dt}\n")
        if msg:
            message += msg
        super().__init__(message)


def parse_particletime(time, fieldset):
    if fieldset is not None and fieldset.time_origin:
        time = fieldset.time_origin.fulltime(time)
    return time


class InterpolationError(KernelError):
    """Particle kernel error for interpolation error."""

    def __init__(self, particle, fieldset=None, lon=None, lat=None, depth=None):
        if lon and lat:
            message = f"Field interpolation error at ({lon}, {lat}, {depth})"
        else:
            message = f"Field interpolation error for particle at ({particle.lon}, {particle.lat}, {particle.depth})"
        super().__init__(particle, fieldset=fieldset, msg=message)


AllParcelsErrorCodes = {FieldSamplingError: StatusCode.Error,
                        FieldOutOfBoundError: StatusCode.ErrorOutOfBounds,
                        FieldOutOfBoundSurfaceError: StatusCode.ErrorThroughSurface,
                        TimeExtrapolationError: StatusCode.ErrorTimeExtrapolation,
                        KernelError: StatusCode.Error,
                        }
