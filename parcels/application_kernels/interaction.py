"""Collection of pre-built interaction kernels"""
import math
import numpy as np

from parcels.tools.statuscodes import OperationCode, StateCode
from parcels.interaction.spherical_utils import relative_3d_distance


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


def AsymmetricAttraction(particle, fieldset, time, neighbours, mutator):
    distances = []
    na_neighbors = []
    if not particle.attractor:
        return StateCode.Success

    for n in neighbours:
        if n.attractor:
            continue
        distances.append(
            relative_3d_distance(particle.lat, particle.lon, particle.depth,
                                 n.lat, n.lon, n.depth))
        na_neighbors.append(n)

    assert fieldset.mesh == "flat"
    velocity_param = 0.01
    for n in na_neighbors:
        assert n.dt == particle.dt
        dx = np.array([particle.lat-n.lat, particle.long-n.long,
                       particle.depth-n.depth])
        dx_norm = np.linalg.norm(dx)
        velocity = velocity_param/(dx_norm**2)

        distance = velocity*n.dt
        d_vec = distance*dx/dx_norm

        def f(n):
            n.lat += d_vec[0]
            n.long += d_vec[1]
            n.depth += d_vec[2]

        mutator[n.id].append(f)

    return StateCode.Success
