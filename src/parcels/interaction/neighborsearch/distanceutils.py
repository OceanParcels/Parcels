import numpy as np


def fast_distance(lat1, lon1, lat2, lon2):
    """Compute the arc distance assuming the earth is a sphere.

    This is not the only possible implementation. It was taken from:
    https://www.mkompf.com/gps/distcalc.html
    """
    g = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    return np.arccos(np.minimum(1, g))


def spherical_distance(depth1_m, lat1_deg, lon1_deg, depth2_m, lat2_deg, lon2_deg):
    """Compute the arc distance, uses degrees as input."""
    R_earth = 6371000
    lat1 = np.pi * lat1_deg / 180
    lon1 = np.pi * lon1_deg / 180
    lat2 = np.pi * lat2_deg / 180
    lon2 = np.pi * lon2_deg / 180

    horiz_dist = R_earth * fast_distance(lat1, lon1, lat2, lon2)

    vert_dist = np.abs(depth1_m - depth2_m)
    return (vert_dist, horiz_dist)
