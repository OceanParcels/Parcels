"""Collection of pre-built interaction kernels"""
import math

from parcels.tools.statuscodes import OperationCode


__all__ = ['DummyMoveNeighbour']


def DummyMoveNeighbour(particle, fieldset, time, neighbours):
    """A particle boosts the movement of its nearest neighbour, by adding
    0.1 to its lat position.
    """
    pass
