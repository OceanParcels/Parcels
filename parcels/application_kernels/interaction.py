"""Collection of pre-built interaction kernels"""
import math
import numpy as np

from parcels.tools.statuscodes import OperationCode, StateCode
from parcels.interaction.spherical_utils import relative_3d_distance


__all__ = ['DummyMoveNeighbour', 'AsymmetricAttraction']


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
        mutator[neighbour_ids[i_min_dist]].append((f,))

    return StateCode.Success


def AsymmetricAttraction(particle, fieldset, time, neighbours, mutator):
    distances = []
    na_neighbors = []
    if not particle.attractor:
        return StateCode.Success
    for n in neighbours:
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


def great_circle(lat_1, long_1, lat_2, long_2):
    from numpy import cos, sin, arctan2
    d_lat = long_2-long_1
    num_alpha_1 = cos(lat_2)*sin(d_lat)
    denom_alpha_1 = cos(lat_1)*sin(lat_2) - sin(lat_1)*cos(lat_2)*cos(d_lat)
    alpha_1 = arctan2(num_alpha_1/denom_alpha_1)
    num_alpha_2 = cos(lat_1)*sin(d_lat)
    denom_alpha_2 = -cos(lat_2)*sin(lat_1) + sin(lat_2)*cos(lat_1)*cos(d_lat)
    alpha_2 = arctan2(num_alpha_2/denom_alpha_2)
    nom_sigma_12 = (cos(lat_1)*sin(lat_2) - sin(lat_1)*cos(lat_2)*cos(d_lat))**2
    nom_sigma_12 += (cos(lat_2)*sin(d_lat))**2
    denom_sigma_12 = sin(lat_1)*sin(lat_2) + cos(lat_1)*cos(lat_2)*cos(d_lat)
    sigma_12 = arctan2(np.sqrt(nom_sigma_12)/denom_sigma_12)
    nom_alpha_0 = sin(alpha_1)*cos(lat_1)
    denom_alpha_0 = cos(alpha_1)**2 + sin(alpha_1)**2*sin(lat_1)**2
    

