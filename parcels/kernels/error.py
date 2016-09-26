"""Collection of pre-built recovery kernels"""
from enum import IntEnum


__all__ = ['ErrorCode', 'recovery_map']


class ErrorCode(IntEnum):
    Success = 0
    Repeat = 1
    Delete = 2
    Fail = 3
    FailOutOfBounds = 4


def recovery_fail(particle):
    """Default failure kernel that throws exception"""
    raise RuntimeError(
        "\nKernel error during execution: %s\n"
        "Particle %s\nTime time: %f,\ttimestep size: %f"
        % (particle.state, particle, particle.time, particle.dt)
    )


# Default mapping of failure types (KernelOp)
# to recovery kernels.
recovery_map = {ErrorCode.Fail: recovery_fail}
