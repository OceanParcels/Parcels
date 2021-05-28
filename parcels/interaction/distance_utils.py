import numpy as np


def fast_distance(lat1, long1, lat2, long2):
    '''Compute the arc distance assuming the earth is a sphere.'''
    g = np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(long1-long2)
    return np.arccos(np.minimum(1, g))


def spherical_distance(lat1_deg, long1_deg, depth1_m, lat2_deg, long2_deg,
                       depth2_m):
    "Compute the arc distance, uses degrees as input."
    R_earth = 6371000
    lat1 = np.pi*lat1_deg/180
    long1 = np.pi*long1_deg/180
    lat2 = np.pi*lat2_deg/180
    long2 = np.pi*long2_deg/180

    surface_dist = R_earth*fast_distance(lat1, long1, lat2, long2)

    depth_dist = np.abs(depth1_m-depth2_m)
    return (surface_dist, depth_dist)
