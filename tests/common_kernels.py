"""Shared kernels between tests."""

from parcels import StatusCode


def DoNothing(particle, fieldset, time):
    pass


def DeleteParticle(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds or particle.state == StatusCode.ErrorThroughSurface:
        particle.delete()


def MoveEast(particle, fieldset, time):
    particle_dlon += 0.1  # noqa


def MoveNorth(particle, fieldset, time):
    particle_dlat += 0.1  # noqa
