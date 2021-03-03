import numpy as np


def fast_distance(lat1, long1, lat2, long2):
    '''Compute the arc distance assuming the earth is a sphere.'''
    g = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(long1-long2)
    return np.arccos(np.minimum(1, g))


def relative_3d_distance(lat1_deg, long1_deg, depth1_m, lat2_deg, long2_deg,
                         depth2_m, interaction_distance=1, interaction_depth=1):
    R_earth = 6371000
    lat1 = np.pi*lat1_deg/180
    long1 = np.pi*long1_deg/180
    lat2 = np.pi*lat2_deg/180
    long2 = np.pi*long2_deg/180

    surface_dist = R_earth*fast_distance(lat1, long1, lat2, long2)

    depth_dist = np.abs(depth1_m-depth2_m)
    rel_dist = np.sqrt((surface_dist/interaction_distance)**2
                       + (depth_dist/interaction_depth)**2)
#     dist = np.sqrt(lat_long_dist**2 + ((depth1-depth2)/depth_factor)**2)
    return rel_dist
