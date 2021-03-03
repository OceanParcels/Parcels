"""Collection of pre-built interaction kernels"""
import math
import numpy as np

from parcels.tools.statuscodes import OperationCode
from parcels.interaction.geo_utils import relative_3d_distance


__all__ = ['DummyMoveNeighbour']


def create_mutator():
    from collections import defaultdict
    return defaultdict(lambda: [])


def DummyMoveNeighbour(particle, fieldset, time, neighbours, mutator):
    """A particle boosts the movement of its nearest neighbour, by adding
    0.1 to its lat position.
    """
    true_neighbours = [n for n in neighbours if n.id != particle.id]
    distances = [
        relative_3d_distance(particle.lat, particle.long, particle.depth,
                             n.lat, n.long, n.depth)
        for n in true_neighbours]
    if len(distances):
        i_min_dist = np.argmin(distances)

        def f(p):
            p.lat += 0.1
        mutator[true_neighbours[i_min_dist].id].append(f)


def apply_mutator(pset, mutator):
    # Can also do by index I suppose.
    for p in pset:
        try:
            for m in mutator[p.id]:
                m(p)
        except KeyError:
            pass
