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

    Based on Ariane/TRACMASS algorithm, as detailed in e.g. Doos et al (https://doi.org/10.5194/gmd-10-1733-2017)

    Only works in scipy mode for now on a 2d stationary flow (so no time evolution)

    Requires an extra fieldset constant: fieldset.grid_type='C' or
    fieldset.grid_type='A', for C and A grids respectively.
    """
    if fieldset.grid_type == 'C':
        # get the lat/lon arrays (C-grid)
        lats_u, lons_u = fieldset.U.grid.lat, fieldset.U.grid.lon
        lats_v, lons_v = fieldset.V.grid.lat, fieldset.V.grid.lon
        lats_p, lons_p = fieldset.P.grid.lat, fieldset.P.grid.lon

        # request corner indices of grid cell (P-grid!) and rx, ry (indices are to the bottom left of particle)
        rx, ry, _, xi, yi, _ = fieldset.P.search_indices_rectilinear(particle.lon, particle.lat, particle.depth, 0, 0)

        # calculate grid resolution
        dx = lons_p[xi + 1] - lons_p[xi]
        dy = lats_p[yi + 1] - lats_p[yi]

        # request velocity at particle position
        up, vp = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

        # shift the grid cell corner indices 1 to the west and/or south if necessary
        # also move rx, ry to '1' if they move west/south and are on a grid face
        if up >= 0 or rx > 0:
            a1, a2 = 0, 1
        else:
            a1, a2 = -1, 0
            rx = 1.

        if vp >= 0 or ry > 0:
            b1, b2 = 0, 1
        else:
            b1, b2 = -1, 0
            ry = 1.

        # set the r_1 target value based on the particle flow direction
        ry_target = 1. if vp >= 0. else 0.
        rx_target = 1. if up >= 0. else 0.

        # get velocities at the surrounding grid boxes
        u_w = fieldset.U[time, particle.depth, lats_u[yi+b1], lons_u[xi+a1]]
        u_e = fieldset.U[time, particle.depth, lats_u[yi+b1], lons_u[xi+a2]]
        v_s = fieldset.V[time, particle.depth, lats_v[yi+b1], lons_v[xi+a1]]
        v_n = fieldset.V[time, particle.depth, lats_v[yi+b2], lons_v[xi+a1]]
    elif fieldset.grid_type == 'A':
        # request corner indices and xsi, eta (indices are to the bottom left of particle)
        rx, ry, _, xi, yi, _ = fieldset.U.search_indices_rectilinear(particle.lon, particle.lat, particle.depth, 0, 0)

        # request velocity at particle position
        up, vp = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

        # get the lat/lon arrays (A-grid!)
        lats = fieldset.P.grid.lat
        lons = fieldset.P.grid.lon

        # set the grid box indices based on the velocity direction or position
        # also move rx, ry to '1' if they move west/south and are on a grid face
        if up >= 0 or rx > 0:
            a1, a2 = 0, 1
        else:
            a1, a2 = -1, 0
            rx = 1.

        if vp >= 0 or ry > 0:
            b1, b2 = 0, 1
        else:
            b1, b2 = -1, 0
            ry = 1.

        ry_target = 1. if vp >= 0. else 0.
        rx_target = 1. if up >= 0. else 0.

        # get velocities at the surrounding grid boxes
        u1, v1 = fieldset.UV[time, particle.depth, lats[yi + b1], lons[xi + a1]]
        u2, v2 = fieldset.UV[time, particle.depth, lats[yi + b1], lons[xi + a2]]
        u3, v3 = fieldset.UV[time, particle.depth, lats[yi + b2], lons[xi + a1]]
        u4, v4 = fieldset.UV[time, particle.depth, lats[yi + b2], lons[xi + a2]]

        # define new variables for velocity for C-grid
        u_w = (u1 + u3) / 2.
        u_e = (u2 + u4) / 2.
        v_s = (v1 + v2) / 2.
        v_n = (v3 + v4) / 2.

        # get the dx/dy
        dx = lons[xi + 1] - lons[xi]
        dy = lats[yi + 1] - lats[yi]
    else:
        raise NotImplementedError('Only A and C grids implemented')

    # calculate the zonal and meridional grid face fluxes
    F_w = u_w * dy
    F_e = u_e * dy
    F_s = v_s * dx
    F_n = v_n * dx

    # calculate betas
    B_x = F_w - F_e
    B_y = F_s - F_n

    # delta
    delta_x = - F_w - B_x * 0.  # where r_(i-1) = 0 by definition
    delta_y = - F_s - B_y * 0.  # where r_(j-1) = 0 by definition

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

    # set tolerance of when something is considered 0
    tol = 1e-8

    # set betas accordingly
    B_x = 0 if abs(B_x) < tol else B_x
    B_y = 0 if abs(B_y) < tol else B_y

    # calculate delta s for x direction
    if B_x == 0 and delta_x == 0:
        ds_x = float('inf')
    elif B_x == 0:
        ds_x = (rx_target - rx) / delta_x
    elif Fu_r1 * Fu_r0 < 0:
        ds_x = float('inf')
    else:
        ds_x = - 1 / B_x * math.log(Fu_r1 / Fu_r0)

    # calculate delta s for y direction
    if B_y == 0 and delta_y == 0:
        ds_y = float('inf')
    elif B_y == 0:
        ds_y = (ry_target - ry) / delta_y
    elif Fv_r1 * Fv_r0 < 0:
        ds_y = float('inf')
    else:
        ds_y = - 1 / B_y * math.log(Fv_r1 / Fv_r0)

    # take the minimum travel time
    s_min = min(ds_x, ds_y)

    # calculate end position in time s_min
    if ds_y == float('inf'):
        rs_x = rx_target
        rs_y = ry
    elif ds_x == float('inf'):
        rs_x = rx
        rs_y = ry_target
    else:
        if B_x == 0:
            rs_x = -delta_x * s_min + rx
        else:
            rs_x = (rx + delta_x/B_x) * math.exp(-B_x*s_min) - delta_x / B_x

        if B_y == 0:
            rs_y = -delta_y * s_min + ry
        else:
            rs_y = (ry + delta_y/B_y) * math.exp(-B_y*s_min) - delta_y / B_y

    # calculate the change in position in cartesian coordinates
    dlon = (rs_x - rx) * dx
    dlat = (rs_y - ry) * dy

    # set new position (round to 8th decimal, due to floating point precision issues)
    particle.lat = round(particle.lat + dlat, 8)
    particle.lon = round(particle.lon + dlon, 8)

    # feedback the passed time to main loop (does not work as intended)
    s_min_real = s_min * (dx * dy)
    particle.dt = s_min_real
