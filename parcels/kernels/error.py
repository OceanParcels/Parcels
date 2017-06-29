"""Collection of pre-built recovery kernels"""
from enum import IntEnum
from datetime import timedelta


__all__ = ['ErrorCode', 'KernelError', 'OutOfBoundsError', 'recovery_map']


class ErrorCode(IntEnum):
    Success = 0
    Repeat = 1
    Delete = 2
    Error = 3
    ErrorOutOfBounds = 4
    ErrorTimeExtrapolation = 5


class KernelError(RuntimeError):
    """General particle kernel error with optional custom message"""

    def __init__(self, particle, field=None, msg=None):
        message = ("%s\nParticle %s\nTime: %s,\ttimestep dt: %f\n") % (
            particle.state, particle, parse_particletime(particle.time, field),
            particle.dt
        )
        if msg:
            message += msg
        super(KernelError, self).__init__(message)


def parse_particletime(time, field):
    if field is not None and field.time_origin != 0:
        time = field.time_origin + timedelta(seconds=time)
    return time


def recovery_kernel_error(particle, field):
    """Default error kernel that throws exception"""
    msg = "Error: %s" % particle.exception if particle.exception else None
    raise KernelError(particle, field=field, msg=msg)


class OutOfBoundsError(KernelError):
    """Particle kernel error for out-of-bounds field sampling"""

    def __init__(self, particle, field, lon=None, lat=None, depth=None):
        if lon and lat:
            message = "%s sampled at (%f, %f, %f)" % (
                field.name if field else "Field", lon, lat, depth
            )
        else:
            message = "Out-of-bounds sampling by particle at (%f, %f, %f)" % (
                particle.lon, particle.lat, particle.depth
            )
        super(OutOfBoundsError, self).__init__(particle, field=field, msg=message)


class OutOfTimeError(KernelError):
    """Particle kernel error for time extrapolation field sampling"""

    def __init__(self, particle, field):
        message = "Field sampled outside time domain at time %s." % (
            parse_particletime(particle.time, field)
        )
        message += " Try setting allow_time_extrapolation to True"
        super(OutOfTimeError, self).__init__(particle, field=field, msg=message)


def recovery_kernel_out_of_bounds(particle, field):
    """Default sampling error kernel that throws OutOfBoundsError"""
    if particle.exception is None:
        # TODO: JIT does not yet provide the context that created
        # the exception. We need to pass that info back from C.
        raise OutOfBoundsError(particle, field)
    else:
        error = particle.exception
        raise OutOfBoundsError(particle, field, error.x, error.y, error.z)


def recovery_kernel_time_extrapolation(particle, field):
    """Default sampling error kernel that throws OutOfTimeError"""
    raise OutOfTimeError(particle, field)


# Default mapping of failure types (KernelOp)
# to recovery kernels.
recovery_map = {ErrorCode.Error: recovery_kernel_error,
                ErrorCode.ErrorOutOfBounds: recovery_kernel_out_of_bounds,
                ErrorCode.ErrorTimeExtrapolation: recovery_kernel_time_extrapolation}
