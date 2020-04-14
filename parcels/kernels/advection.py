"""Collection of pre-built advection kernels"""
import math

from parcels.tools.error import ErrorCode


__all__ = ['AdvectionRK4', 'AdvectionEE', 'AdvectionRK45', 'AdvectionRK4_3D',
           'AdvectionAnalytical']


def AdvectionRK4(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.

    Function needs to be converted to Kernel object before execution"""
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
    (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
    lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
    (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
    lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt


def AdvectionRK4_3D(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.

    Function needs to be converted to Kernel object before execution"""
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt


def AdvectionEE(particle, fieldset, time):
    """Advection of particles using Explicit Euler (aka Euler Forward) integration.

    Function needs to be converted to Kernel object before execution"""
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    particle.lon += u1 * particle.dt
    particle.lat += v1 * particle.dt


def AdvectionRK45(particle, fieldset, time):
    """Advection of particles using adadptive Runge-Kutta 4/5 integration.

    Times-step dt is halved if error is larger than tolerance, and doubled
    if error is smaller than 1/10th of tolerance, with tolerance set to
    1e-5 * dt by default."""
    tol = [1e-5]
    c = [1./4., 3./8., 12./13., 1., 1./2.]
    A = [[1./4., 0., 0., 0., 0.],
         [3./32., 9./32., 0., 0., 0.],
         [1932./2197., -7200./2197., 7296./2197., 0., 0.],
         [439./216., -8., 3680./513., -845./4104., 0.],
         [-8./27., 2., -3544./2565., 1859./4104., -11./40.]]
    b4 = [25./216., 0., 1408./2565., 2197./4104., -1./5.]
    b5 = [16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.]

    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    lon1, lat1 = (particle.lon + u1 * A[0][0] * particle.dt,
                  particle.lat + v1 * A[0][0] * particle.dt)
    (u2, v2) = fieldset.UV[time + c[0] * particle.dt, particle.depth, lat1, lon1]
    lon2, lat2 = (particle.lon + (u1 * A[1][0] + u2 * A[1][1]) * particle.dt,
                  particle.lat + (v1 * A[1][0] + v2 * A[1][1]) * particle.dt)
    (u3, v3) = fieldset.UV[time + c[1] * particle.dt, particle.depth, lat2, lon2]
    lon3, lat3 = (particle.lon + (u1 * A[2][0] + u2 * A[2][1] + u3 * A[2][2]) * particle.dt,
                  particle.lat + (v1 * A[2][0] + v2 * A[2][1] + v3 * A[2][2]) * particle.dt)
    (u4, v4) = fieldset.UV[time + c[2] * particle.dt, particle.depth, lat3, lon3]
    lon4, lat4 = (particle.lon + (u1 * A[3][0] + u2 * A[3][1] + u3 * A[3][2] + u4 * A[3][3]) * particle.dt,
                  particle.lat + (v1 * A[3][0] + v2 * A[3][1] + v3 * A[3][2] + v4 * A[3][3]) * particle.dt)
    (u5, v5) = fieldset.UV[time + c[3] * particle.dt, particle.depth, lat4, lon4]
    lon5, lat5 = (particle.lon + (u1 * A[4][0] + u2 * A[4][1] + u3 * A[4][2] + u4 * A[4][3] + u5 * A[4][4]) * particle.dt,
                  particle.lat + (v1 * A[4][0] + v2 * A[4][1] + v3 * A[4][2] + v4 * A[4][3] + v5 * A[4][4]) * particle.dt)
    (u6, v6) = fieldset.UV[time + c[4] * particle.dt, particle.depth, lat5, lon5]

    lon_4th = particle.lon + (u1 * b4[0] + u2 * b4[1] + u3 * b4[2] + u4 * b4[3] + u5 * b4[4]) * particle.dt
    lat_4th = particle.lat + (v1 * b4[0] + v2 * b4[1] + v3 * b4[2] + v4 * b4[3] + v5 * b4[4]) * particle.dt
    lon_5th = particle.lon + (u1 * b5[0] + u2 * b5[1] + u3 * b5[2] + u4 * b5[3] + u5 * b5[4] + u6 * b5[5]) * particle.dt
    lat_5th = particle.lat + (v1 * b5[0] + v2 * b5[1] + v3 * b5[2] + v4 * b5[3] + v5 * b5[4] + v6 * b5[5]) * particle.dt

    kappa = math.sqrt(math.pow(lon_5th - lon_4th, 2) + math.pow(lat_5th - lat_4th, 2))
    if kappa <= math.fabs(particle.dt * tol[0]):
        particle.lon = lon_4th
        particle.lat = lat_4th
        if kappa <= math.fabs(particle.dt * tol[0] / 10):
            particle.update_next_dt(particle.dt * 2)
    else:
        particle.dt /= 2
        return ErrorCode.Repeat


def AdvectionAnalytical(particle, fieldset, time):
    """Advection of particles using 'analytical advection' integration

    Based on Ariane/TRACMASS algorithm, as detailed in e.g. Doos et al (https://doi.org/10.5194/gmd-10-1733-2017)"""
    import numpy as np
    import parcels.tools.interpolation_utils as i_u

    # set tolerance of when something is considered 0
    tol = 1e-8

    # request velocity at particle position
    up, vp = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    # request corner indices and xsi, eta (indices are to the bottom left of particle)
    rx, ry, _, xi, yi, _ = fieldset.U.search_indices(particle.lon, particle.lat, particle.depth, 0, 0)
    if (up > 0) & (abs(rx - 1) < tol):
        xi += 1
        rx = 0
    if (vp > 0) & (abs(ry - 1) < tol):
        yi += 1
        ry = 0

    grid = fieldset.U.grid
    if grid.gtype < 2:
        px = np.array([grid.lon[xi], grid.lon[xi + 1], grid.lon[xi + 1], grid.lon[xi]])
        py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi + 1], grid.lat[yi + 1]])
    else:
        px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
        py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])
    if grid.mesh == 'spherical':
        px[0] = px[0]+360 if px[0] < particle.lon-225 else px[0]
        px[0] = px[0]-360 if px[0] > particle.lat+225 else px[0]
        px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
        px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])

    c1 = fieldset.UV.dist(px[0], px[1], py[0], py[1], grid.mesh, np.dot(i_u.phi2D_lin(rx, 0.), py))
    c2 = fieldset.UV.dist(px[1], px[2], py[1], py[2], grid.mesh, np.dot(i_u.phi2D_lin(1., ry), py))
    c3 = fieldset.UV.dist(px[2], px[3], py[2], py[3], grid.mesh, np.dot(i_u.phi2D_lin(rx, 1.), py))
    c4 = fieldset.UV.dist(px[3], px[0], py[3], py[0], grid.mesh, np.dot(i_u.phi2D_lin(0., ry), py))
    F_w = fieldset.U.data[0, yi+1, xi] * c4  # TODO time-varying
    F_e = fieldset.U.data[0, yi+1, xi+1] * c2
    F_s = fieldset.V.data[0, yi, xi+1] * c1
    F_n = fieldset.V.data[0, yi+1, xi+1] * c3
    dx = (c4 + c2)/2.
    dy = (c1 + c3)/2.

    ry_target = 1. if vp >= 0. else 0.
    rx_target = 1. if up >= 0. else 0.

    # calculate betas
    B_x = F_w - F_e
    B_y = F_s - F_n

    # calculate deltas
    delta_x = - F_w - B_x * 0.
    delta_y = - F_s - B_y * 0.

    # calculate F(r0) and F(r1) for both directions (unless beta == 0)
    if B_x != 0.:
        Fu_r1 = rx_target + delta_x / B_x
        Fu_r0 = rx + delta_x / B_x
    else:
        Fu_r0, Fu_r1 = None, None
    if B_y != 0.:
        Fv_r1 = ry_target + delta_y / B_y
        Fv_r0 = ry + delta_y / B_y
    else:
        Fv_r0, Fv_r1 = None, None

    # set betas accordingly
    B_x = 0 if abs(B_x) < tol else B_x
    B_y = 0 if abs(B_y) < tol else B_y

    # calculate delta s for x direction
    if B_x == 0 and delta_x == 0:
        ds_x = float('inf')
    elif B_x == 0:
        ds_x = -(rx_target - rx) / delta_x
    elif Fu_r1 * Fu_r0 < 0:
        ds_x = float('inf')
    else:
        ds_x = - 1. / B_x * math.log(Fu_r1 / Fu_r0)

    # calculate delta s for y direction
    if B_y == 0 and delta_y == 0:
        ds_y = float('inf')
    elif B_y == 0:
        ds_y = -(ry_target - ry) / delta_y
    elif Fv_r1 * Fv_r0 < 0:
        ds_y = float('inf')
    else:
        ds_y = - 1. / B_y * math.log(Fv_r1 / Fv_r0)

    if ds_x < tol:
        ds_x = float('inf')
    if ds_y < tol:
        ds_y = float('inf')

    # take the minimum travel time
    s_min = min(ds_x, ds_y, particle.dt / (dx*dy))

    # calculate end position in time s_min
    if B_x == 0:
        rs_x = -delta_x * s_min + rx
    else:
        rs_x = (rx + delta_x/B_x) * math.exp(-B_x*s_min) - delta_x / B_x

    if B_y == 0:
        rs_y = -delta_y * s_min + ry
    else:
        rs_y = (ry + delta_y/B_y) * math.exp(-B_y*s_min) - delta_y / B_y

    particle.lon = (1.-rs_x)*(1.-rs_y) * px[0] + rs_x * (1.-rs_y) * px[1] + rs_x * rs_y * px[2] + (1.-rs_x)*rs_y * px[3]
    particle.lat = (1.-rs_x)*(1.-rs_y) * py[0] + rs_x * (1.-rs_y) * py[1] + rs_x * rs_y * py[2] + (1.-rs_x)*rs_y * py[3]
    # print(particle.lon, particle.lat)

    # update the passed time for the main loop
    particle.dt = s_min * (dx * dy)
