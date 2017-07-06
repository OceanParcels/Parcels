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

    def __init__(self, particle, fieldset=None, msg=None):
        message = ("%s\nParticle %s\nTime: %s,\ttimestep dt: %f\n") % (
            particle.state, particle, parse_particletime(particle.time, fieldset),
            particle.dt
        )
        if msg:
            message += msg
        super(KernelError, self).__init__(message)


def parse_particletime(time, fieldset):
    if fieldset is not None and fieldset.U.time_origin != 0:
        # TODO assuming that error was thrown on U field
        time = fieldset.U.time_origin + timedelta(seconds=time)
    return time


def recovery_kernel_error(particle, fieldset, time, dt):
    """Default error kernel that throws exception"""
    msg = "Error: %s" % particle.exception if particle.exception else None
    raise KernelError(particle, fieldset=fieldset, msg=msg)


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


class OutOfTimeError(KernelError):
    """Particle kernel error for time extrapolation field sampling"""

    def __init__(self, particle, fieldset):
        message = "Field sampled outside time domain at time %s." % (
            parse_particletime(particle.time, fieldset)
        )
        message += " Try setting allow_time_extrapolation to True"
        super(OutOfTimeError, self).__init__(particle, fieldset=fieldset, msg=message)


def recovery_kernel_out_of_bounds(particle, fieldset, time, dt):
    """Default sampling error kernel that throws OutOfBoundsError"""
    if particle.exception is None:
        # TODO: JIT does not yet provide the context that created
        # the exception. We need to pass that info back from C.
        raise OutOfBoundsError(particle, fieldset)
    else:
        error = particle.exception
        raise OutOfBoundsError(particle, fieldset, error.x, error.y, error.z)


def recovery_kernel_time_extrapolation(particle, fieldset, time, dt):
    """Default sampling error kernel that throws OutOfTimeError"""
    raise OutOfTimeError(particle, fieldset)


# Default mapping of failure types (KernelOp)
# to recovery kernels.
recovery_map = {ErrorCode.Error: recovery_kernel_error,
                ErrorCode.ErrorOutOfBounds: recovery_kernel_out_of_bounds,
                ErrorCode.ErrorTimeExtrapolation: recovery_kernel_time_extrapolation}
