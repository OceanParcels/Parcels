"""Collection of pre-built recovery kernels"""


__all__ = ['StateCode', 'OperationCode', 'ErrorCode',
           'FieldSamplingError', 'FieldOutOfBoundError', 'TimeExtrapolationError',
           'KernelError', 'OutOfBoundsError', 'ThroughSurfaceError', 'OutOfTimeError',
           'recovery_map']


class StateCode(object):
    Success = 0
    Evaluate = 1


class OperationCode(object):
    Repeat = 2
    Delete = 3
    StopExecution = 4


class ErrorCode(object):
    Error = 5
    ErrorInterpolation = 51
    ErrorOutOfBounds = 6
    ErrorThroughSurface = 61
    ErrorTimeExtrapolation = 7


class DaskChunkingError(RuntimeError):
    """
    Error indicating to the user that something with setting up Dask and chunked fieldsets went wrong.
    """

    def __init__(self, src_class_type, message):
        msg = "[{}]: {}".format(str(src_class_type), message)
        super(DaskChunkingError, self).__init__(msg)


class FieldSamplingError(RuntimeError):
    """Utility error class to propagate erroneous field sampling in Scipy mode"""

    def __init__(self, x, y, z, field=None):
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        message = "%s sampled at (%f, %f, %f)" % (
            field.name if field else "Field", self.x, self.y, self.z
        )
        super(FieldSamplingError, self).__init__(message)


class FieldOutOfBoundError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling in Scipy mode"""

    def __init__(self, x, y, z, field=None):
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        message = "%s sampled out-of-bound, at (%f, %f, %f)" % (
            field.name if field else "Field", self.x, self.y, self.z
        )
        super(FieldOutOfBoundError, self).__init__(message)


class FieldOutOfBoundSurfaceError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling at the surface in Scipy mode.
       Note that if ErrorThroughSurface is not part of the recovery kernel, Parcels will use ErrorOutOfBounds."""

    def __init__(self, x, y, z, field=None):
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        message = "%s sampled out-of-bound at the surface, at (%f, %f, %f)" % (
            field.name if field else "Field", self.x, self.y, self.z
        )
        super(FieldOutOfBoundSurfaceError, self).__init__(message)


class TimeExtrapolationError(RuntimeError):
    """Utility error class to propagate erroneous time extrapolation sampling in Scipy mode"""

    def __init__(self, time, field=None, msg='allow_time_extrapoltion'):
        if field is not None and field.grid.time_origin and time is not None:
            time = field.grid.time_origin.fulltime(time)
        message = "%s sampled outside time domain at time %s." % (
            field.name if field else "Field", time)
        if msg == 'allow_time_extrapoltion':
            message += " Try setting allow_time_extrapolation to True"
        elif msg == 'show_time':
            message += " Try explicitly providing a 'show_time'"
        else:
            message += msg + " Try setting allow_time_extrapolation to True"
        super(TimeExtrapolationError, self).__init__(message)


class KernelError(RuntimeError):
    """General particle kernel error with optional custom message"""

    def __init__(self, particle, fieldset=None, msg=None):
        message = ("%s\nParticle %s\nTime: %s,\ttimestep dt: %f\n") % (
            particle.state, particle, parse_particletime(particle.time, fieldset),
            particle.dt
        )
        if msg:
            message += msg
        super(KernelError, self).__init__(message)


def parse_particletime(time, fieldset):
    if fieldset is not None and fieldset.time_origin:
        time = fieldset.time_origin.fulltime(time)
    return time


def recovery_kernel_error(particle, fieldset, time):
    """Default error kernel that throws exception"""
    msg = "Error: %s" % particle.exception if particle.exception else None
    raise KernelError(particle, fieldset=fieldset, msg=msg)


class InterpolationError(KernelError):
    """Particle kernel error for interpolation error"""

    def __init__(self, particle, fieldset=None, lon=None, lat=None, depth=None):
        if lon and lat:
            message = "Field interpolation error at (%f, %f, %f)" % (
                lon, lat, depth
            )
        else:
            message = "Field interpolation error for particle at (%f, %f, %f)" % (
                particle.lon, particle.lat, particle.depth
            )
        super(InterpolationError, self).__init__(particle, fieldset=fieldset, msg=message)


class OutOfBoundsError(KernelError):
    """Particle kernel error for out-of-bounds field sampling"""

    def __init__(self, particle, fieldset=None, lon=None, lat=None, depth=None):
        if lon and lat:
            message = "Field sampled at (%f, %f, %f)" % (
                lon, lat, depth
            )
        else:
            message = "Out-of-bounds sampling by particle at (%f, %f, %f)" % (
                particle.lon, particle.lat, particle.depth
            )
        super(OutOfBoundsError, self).__init__(particle, fieldset=fieldset, msg=message)


class ThroughSurfaceError(KernelError):
    """Particle kernel error for field sampling at surface"""

    def __init__(self, particle, fieldset=None, lon=None, lat=None, depth=None):
        if lon and lat:
            message = "Field sampled at (%f, %f, %f)" % (
                lon, lat, depth
            )
        else:
            message = "Through-surface sampling by particle at (%f, %f, %f)" % (
                particle.lon, particle.lat, particle.depth
            )
        super(ThroughSurfaceError, self).__init__(particle, fieldset=fieldset, msg=message)


class OutOfTimeError(KernelError):
    """Particle kernel error for time extrapolation field sampling"""

    def __init__(self, particle, fieldset):
        message = "Field sampled outside time domain at time %s." % (
            parse_particletime(particle.time, fieldset)
        )
        message += " Try setting allow_time_extrapolation to True"
        super(OutOfTimeError, self).__init__(particle, fieldset=fieldset, msg=message)


def recovery_kernel_interpolation(particle, fieldset, time):
    """Default sampling error kernel that throws InterpolationError"""
    if particle.exception is None:
        # TODO: JIT does not yet provide the context that created
        # the exception. We need to pass that info back from C.
        raise InterpolationError(particle, fieldset)
    else:
        error = particle.exception
        raise InterpolationError(particle, fieldset, error.x, error.y, error.z)


def recovery_kernel_out_of_bounds(particle, fieldset, time):
    """Default sampling error kernel that throws OutOfBoundsError"""
    if particle.exception is None:
        # TODO: JIT does not yet provide the context that created
        # the exception. We need to pass that info back from C.
        raise OutOfBoundsError(particle, fieldset)
    else:
        error = particle.exception
        raise OutOfBoundsError(particle, fieldset, error.x, error.y, error.z)


def recovery_kernel_through_surface(particle, fieldset, time):
    """Default sampling error kernel that throws OutOfBoundsError"""
    if particle.exception is None:
        # TODO: JIT does not yet provide the context that created
        # the exception. We need to pass that info back from C.
        raise ThroughSurfaceError(particle, fieldset)
    else:
        error = particle.exception
        raise ThroughSurfaceError(particle, fieldset, error.z)


def recovery_kernel_time_extrapolation(particle, fieldset, time):
    """Default sampling error kernel that throws OutOfTimeError"""
    raise OutOfTimeError(particle, fieldset)


# Default mapping of failure types (KernelOp)
# to recovery kernels.
recovery_map = {ErrorCode.Error: recovery_kernel_error,
                ErrorCode.ErrorInterpolation: recovery_kernel_interpolation,
                ErrorCode.ErrorOutOfBounds: recovery_kernel_out_of_bounds,
                ErrorCode.ErrorTimeExtrapolation: recovery_kernel_time_extrapolation,
                ErrorCode.ErrorThroughSurface: recovery_kernel_through_surface}
