"""Shared kernels between tests."""


def DoNothing(particle, fieldset, time):
    pass


def DeleteParticle(particle, fieldset, time):
    if particle.state >= 50:  # This captures all Errors
        particle.delete()


def MoveEast(particle, fieldset, time):
    particle_dlon += 0.1  # noqa


def MoveNorth(particle, fieldset, time):
    particle_dlat += 0.1  # noqa
