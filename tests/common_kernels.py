"""Shared kernels between tests."""


def DoNothing(particle, fieldset, time):  # pragma: no cover
    pass


def DeleteParticle(particle, fieldset, time):  # pragma: no cover
    if particle.state >= 50:  # This captures all Errors
        particle.delete()


def MoveEast(particle, fieldset, time):  # pragma: no cover
    particle_dlon += 0.1  # noqa


def MoveNorth(particle, fieldset, time):  # pragma: no cover
    particle_dlat += 0.1  # noqa
