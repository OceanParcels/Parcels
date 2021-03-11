"""Collection of pre-built interaction kernels"""
import math
import numpy as np

from parcels.tools.statuscodes import OperationCode, StateCode
from parcels.interaction.geo_utils import relative_3d_distance


__all__ = ['DummyMoveNeighbour']


def DummyMoveNeighbour(particle, fieldset, time, neighbours, mutator):
    """A particle boosts the movement of its nearest neighbour, by adding
    0.1 to its lat position.
    """
    distances = []
    neighbour_ids = []
    for n in neighbours:
        distances.append(
            relative_3d_distance(particle.lat, particle.lon, particle.depth,
                                 n.lat, n.lon, n.depth))
        neighbour_ids.append(n.id)

    if len(distances):
        i_min_dist = np.argmin(distances)

        def f(p):
            p.lat += 0.1
        mutator[neighbour_ids[i_min_dist]].append(f)

    return StateCode.Success
