"""Collection of pre-built interaction kernels"""
import numpy as np

from parcels.tools.statuscodes import OperationCode, StateCode


__all__ = ['AsymmetricAttraction', 'NearestNeighborWithinRange',
           'MergeWithNearestNeighbor']


def NearestNeighborWithinRange(particle, fieldset, time, neighbors, mutator):
    """Particle has to have the nearest_neighbor property
    """
    min_dist = -1
    neighbor_id = -1
    for n in neighbors:
        dist = np.sqrt(n.surf_dist**2 + n.depth_dist**2)
        # Note that in case of a tie, the particle with the lowest ID
        # wins. In certain adverserial cases, this might lead to
        # undesirable results.
        if dist < min_dist or min_dist == -1:
            min_dist = dist
            neighbor_id = n.id

    def f(p, neighbor):
        p.nearest_neighbor = neighbor
    mutator[particle.id].append((f, [neighbor_id]))

    return StateCode.Success


def MergeWithNearestNeighbor(particle, fieldset, time, neighbors, mutator):
    """Particle has to have the nearest_neighbor and mass properties
    """
    for n in neighbors:
        if n.id == particle.nearest_neighbor:
            if n.nearest_neighbor == particle.id and particle.id < n.id:
                # Merge particles
                def g(p):
                    p.state = OperationCode.Delete
                mutator[n.id].append((g, ()))

                def f(p, nlat, nlon, ndepth, nmass):
                    p.lat = (p.mass * p.lat + nmass * nlat) / (p.mass + nmass)
                    p.lon = (p.mass * p.lon + nmass * nlon) / (p.mass + nmass)
                    p.depth = (p.mass * p.depth + nmass * ndepth) / (p.mass + nmass)
                    p.mass = p.mass + nmass
                args = np.array([n.lat, n.lon, n.depth, n.mass])
                mutator[particle.id].append((f, args))

                return StateCode.Success
            else:
                return StateCode.Success

    return StateCode.Success


def AsymmetricAttraction(particle, fieldset, time, neighbors, mutator):
    distances = []
    na_neighbors = []
    if not particle.attractor:
        return StateCode.Success
    for n in neighbors:
        if n.attractor:
            continue
        x_p = np.array([particle.lat, particle.lon, particle.depth])
        x_n = np.array([n.lat, n.lon, n.depth])
        distances.append(np.linalg.norm(x_p-x_n))
        na_neighbors.append(n)

#     assert fieldset.mesh == "flat"
    velocity_param = 0.000004
    for n in na_neighbors:
        assert n.dt == particle.dt
        dx = np.array([particle.lat-n.lat, particle.lon-n.lon,
                       particle.depth-n.depth])
        dx_norm = np.linalg.norm(dx)
        velocity = velocity_param/(dx_norm**2)

        distance = velocity*n.dt
        d_vec = distance*dx/dx_norm

        def f(n, dlat, dlon, ddepth):
            n.lat += dlat
            n.lon += dlon
            n.depth += ddepth

        mutator[n.id].append((f, d_vec))

    return StateCode.Success
