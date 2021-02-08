import numpy as np


def fast_distance(lat1, long1, lat2, long2):
    '''Compute the arc distance assuming the earth is a sphere.'''
    g = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(long1-long2)
    return np.arccos(np.minimum(1, g))


def fast_3d_distance(lat1, long1, depth1, lat2, long2, depth2, depth_factor=1):
    lat_long_dist = fast_distance(lat1, long1, lat2, long2)
    dist = np.sqrt(lat_long_dist**2 + ((depth1-depth2)/depth_factor)**2)
    return dist
