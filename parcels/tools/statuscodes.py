"""Handling of Errors and particle status codes"""


__all__ = ['StateCode', 'OperationCode', 'ErrorCode',
           'FieldSamplingError', 'FieldOutOfBoundError', 'TimeExtrapolationError',
           'KernelError', 'AllParcelsErrors', 'AllParcelsErrorCodes']


class StateCode:
    Success = 0
    Evaluate = 1


class OperationCode:
    Repeat = 2
    Delete = 3
    StopExecution = 4


class ErrorCode:
    Error = 5
    ErrorInterpolation = 51
    ErrorOutOfBounds = 6
    ErrorThroughSurface = 61
    ErrorTimeExtrapolation = 7


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


class NotTestedError(Exception):
    """Raised when the function is not tested. Code is commented."""

    def __init__(self):
        super().__init__("Method has not been tested. Code has been commented out until unit tests are available.")


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


AllParcelsErrors = (FieldSamplingError, FieldOutOfBoundError, FieldOutOfBoundSurfaceError, TimeExtrapolationError,  # TODO see if we can use only the dictionary, not also this list
                    KernelError)


AllParcelsErrorCodes = {FieldSamplingError: ErrorCode.Error,
                        FieldOutOfBoundError: ErrorCode.ErrorOutOfBounds,
                        FieldOutOfBoundSurfaceError: ErrorCode.ErrorThroughSurface,
                        TimeExtrapolationError: ErrorCode.ErrorTimeExtrapolation,
                        KernelError: ErrorCode.Error,
                        }
