"""Shared kernels between tests."""

import numpy as np

from parcels import StatusCode


def DoNothing(particles, fieldset):  # pragma: no cover
    pass


def DeleteParticle(particles, fieldset):  # pragma: no cover
    particles.state = np.where(particles.state >= 50, StatusCode.Delete, particles.state)


def MoveEast(particles, fieldset):  # pragma: no cover
    particles.dlon += 0.1


def MoveNorth(particles, fieldset):  # pragma: no cover
    particles.dlat += 0.1
