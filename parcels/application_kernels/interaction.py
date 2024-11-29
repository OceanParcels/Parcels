"""Collection of pre-built interaction kernels."""

import numpy as np

from parcels.tools.statuscodes import StatusCode

__all__ = ["AsymmetricAttraction", "MergeWithNearestNeighbor", "NearestNeighborWithinRange"]


def NearestNeighborWithinRange(particle, fieldset, time, neighbors, mutator):
    """Computes the nearest neighbor within range for each particle.

    Particle has to have the nearest_neighbor property. If no particle
    is in range, set nearest_neighbor property to -1.
    """
    min_dist = -1
    neighbor_id = -1
    for n in neighbors:
        # Note that with interacting particles p.surf_dist, p.depth_dist are
        # automatically set to be the distance along the surface and
        # z-direction respectively.
        dist = np.sqrt(n.horiz_dist**2 + n.vert_dist**2)
        # Note that in case of a tie, the particle with the lowest ID
        # wins. In certain adversarial cases, this might lead to
        # undesirable results.
        if dist < min_dist or min_dist < 0:
            min_dist = dist
            neighbor_id = n.id

    def f(p, neighbor):
        p.nearest_neighbor = neighbor

    mutator[particle.id].append((f, [neighbor_id]))

    return StatusCode.Success


def MergeWithNearestNeighbor(particle, fieldset, time, neighbors, mutator):
    """Perform merge action with nearest neighbor.

    Depends on NearestNeighborWithinRange kernel, or one that provides similar
    functionality. Particle has to have the nearest_neighbor and mass
    properties. Only pairs of particles that have each other as nearest
    neighbors will be merged.
    """

    def delete_particle(p):
        p.state = StatusCode.Delete

    def merge_with_neighbor(p, nlat, nlon, ndepth, nmass):
        p.lat_nextloop = (p.mass * p.lat + nmass * nlat) / (p.mass + nmass)
        p.lon_nextloop = (p.mass * p.lon + nmass * nlon) / (p.mass + nmass)
        p.depth_nextloop = (p.mass * p.depth + nmass * ndepth) / (p.mass + nmass)
        p.mass = p.mass + nmass

    for n in neighbors:
        if n.id == particle.nearest_neighbor:
            if n.nearest_neighbor == particle.id and particle.id < n.id:
                # Merge particles:
                # Delete neighbor
                mutator[n.id].append((delete_particle, ()))
                # Take position at the mid point and sum of masses
                args = np.array([n.lat, n.lon, n.depth, n.mass])
                mutator[particle.id].append((merge_with_neighbor, args))

                return StatusCode.Success
            else:
                return StatusCode.Success

    return StatusCode.Success


def AsymmetricAttraction(particle, fieldset, time, neighbors, mutator):
    """Example of asymmetric attraction between particles.

    Particles should have the attractor attribute. If attractor==True, then
    it attracts particles around it, but doesn't experience any attraction
    itself. Particles with attractor=False are only attracted to attractors.
    Works only properly on a flat mesh (because of vector calculations).
    """
    na_neighbors = []
    if not particle.attractor:
        return StatusCode.Success
    for n in neighbors:
        if n.attractor:
            continue
        na_neighbors.append(n)

    velocity_param = 0.04
    for n in na_neighbors:
        assert n.dt == particle.dt
        dx = np.array([particle.lat - n.lat, particle.lon - n.lon, particle.depth - n.depth])
        dx_norm = np.linalg.norm(dx)
        velocity = velocity_param / (dx_norm**2)

        distance = velocity * n.dt
        d_vec = distance * dx / dx_norm

        def f(n, dlat, dlon, ddepth):
            n.lat_nextloop += dlat
            n.lon_nextloop += dlon
            n.depth_nextloop += ddepth

        mutator[n.id].append((f, d_vec))

    return StatusCode.Success
