"""Collection of pre-built advection-diffusion kernels"""
import math

import parcels.rng as ParcelsRandom


__all__ = ['DiffusionUniformKh', 'AdvectionDiffusionM1', 'AdvectionRK4DiffusionM1',
           'AdvectionDiffusionEM', 'AdvectionRK4DiffusionEM', ]


def DiffusionUniformKh(particle, fieldset, time):
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.
    Assumes that fieldset has fields Kh_zonal and Kh_meridional.

    This kernel neglects gradients in the diffusivity field and is
    therefore more efficient in cases with uniform diffusivity.
    Since the perturbation due to diffusion is in this case spatially
    independent, this kernel contains no advection and can be used in
    combination with a seperate advection kernel.

    The Wiener increment `dW` should be normally distributed with zero
    mean and a standard deviation of sqrt(dt). Instead, here a uniform
    distribution with the same mean and std is used for efficiency and
    to keep random increments bounded. This substitution is valid for
    the convergence of particle distributions. If convergence of
    individual particle paths is required, use normally distributed
    random increments instead. See Gräwe et al (2012)
    doi.org/10.1007/s10236-012-0523-y for more information.
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    dWy = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)

    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon])
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon])

    particle.lon += bx * dWx
    particle.lat += by * dWy


def AdvectionDiffusionM1(particle, fieldset, time):
    """Kernel for 2D advection-diffusion, solved using the Milstein scheme
    at first order (M1).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central difference
    gradient approximation. This should be at least an order of magnitude
    less than the typical grid resolution.

    The Milstein scheme is superior to the Euler-Maruyama scheme, experiencing
    less spurious background diffusivity by including extra correction
    terms that are computationally cheap.

    The Wiener increment `dW` should be normally distributed with zero
    mean and a standard deviation of sqrt(dt). Instead, here a uniform
    distribution with the same mean and std is used for efficiency and
    to keep random increments bounded. This substitution is valid for
    the convergence of particle distributions. If convergence of
    individual particle paths is required, use normally distributed
    random increments instead. See Gräwe et al (2012)
    doi.org/10.1007/s10236-012-0523-y for more information.
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    dWy = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)

    Kxp1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon + fieldset.dres]
    Kxm1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon - fieldset.dres]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)

    u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon])

    Kyp1 = fieldset.Kh_meridional[time, particle.depth, particle.lat + fieldset.dres, particle.lon]
    Kym1 = fieldset.Kh_meridional[time, particle.depth, particle.lat - fieldset.dres, particle.lon]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)

    v = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon])

    # Particle positions are updated only after evaluating all terms.
    particle.lon += u * particle.dt + 0.5 * dKdx * (dWx**2 + particle.dt) + bx * dWx
    particle.lat += v * particle.dt + 0.5 * dKdy * (dWy**2 + particle.dt) + by * dWy


def AdvectionRK4DiffusionM1(particle, fieldset, time):
    """Kernel for 2D advection-diffusion, with advection solved
    using fourth order Runge-Kutta (RK4) and diffusion using the Milstein
    scheme at first order (M1). Using the RK4 scheme for diffusion is only
    advantageous in areas where the contribution from diffusion is
    negligible.

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central difference
    gradient approximation. This should be at least an order of magnitude
    less than the typical grid resolution.

    The Milstein scheme is superior to the Euler-Maruyama scheme, experiencing
    less spurious background diffusivity by including extra correction
    terms that are computationally cheap.

    The Wiener increment `dW` should be normally distributed with zero
    mean and a standard deviation of sqrt(dt). Instead, here a uniform
    distribution with the same mean and std is used for efficiency and
    to keep random increments bounded. This substitution is valid for
    the convergence of particle distributions. If convergence of
    individual particle paths is required, use normally distributed
    random increments instead. See Gräwe et al (2012)
    doi.org/10.1007/s10236-012-0523-y for more information.
    """
    # RK4 terms
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    lon1, lat1 = (particle.lon + u1 * .5 * particle.dt, particle.lat + v1 * .5 * particle.dt)
    (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
    lon2, lat2 = (particle.lon + u2 * .5 * particle.dt, particle.lat + v2 * .5 * particle.dt)
    (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
    lon3, lat3 = (particle.lon + u3 * particle.dt, particle.lat + v3 * particle.dt)
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]

    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    dWy = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)

    Kxp1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon + fieldset.dres]
    Kxm1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon - fieldset.dres]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon])

    Kyp1 = fieldset.Kh_meridional[time, particle.depth, particle.lat + fieldset.dres, particle.lon]
    Kym1 = fieldset.Kh_meridional[time, particle.depth, particle.lat - fieldset.dres, particle.lon]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon])

    # Particle positions are updated only after evaluating all terms.
    particle.lon += ((u1 + 2 * u2 + 2 * u3 + u4) / 6.) * particle.dt + 0.5 * dKdx * (dWx**2 + particle.dt) + bx * dWx
    particle.lat += ((v1 + 2 * v2 + 2 * v3 + v4) / 6.) * particle.dt + 0.5 * dKdy * (dWy**2 + particle.dt) + by * dWy


def AdvectionDiffusionEM(particle, fieldset, time):
    """Kernel for 2D advection-diffusion, solved using the Euler-Maruyama
    scheme (EM).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central
    difference gradient approximation. This should be at least an order of
    magnitude less than the typical grid resolution.

    The Wiener increment `dW` should be normally distributed with zero
    mean and a standard deviation of sqrt(dt). Instead, here a uniform
    distribution with the same mean and std is used for efficiency and
    to keep random increments bounded. This substitution is valid for
    the convergence of particle distributions. If convergence of
    individual particle paths is required, use normally distributed
    random increments instead. See Gräwe et al (2012)
    doi.org/10.1007/s10236-012-0523-y for more information.
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    dWy = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)

    Kxp1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon + fieldset.dres]
    Kxm1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon - fieldset.dres]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)
    ax = fieldset.U[time, particle.depth, particle.lat, particle.lon] + dKdx
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon])

    Kyp1 = fieldset.Kh_meridional[time, particle.depth, particle.lat + fieldset.dres, particle.lon]
    Kym1 = fieldset.Kh_meridional[time, particle.depth, particle.lat - fieldset.dres, particle.lon]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)
    ay = fieldset.V[time, particle.depth, particle.lat, particle.lon] + dKdy
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon])

    # Particle positions are updated only after evaluating all terms.
    particle.lon += ax * particle.dt + bx * dWx
    particle.lat += ay * particle.dt + by * dWy


def AdvectionRK4DiffusionEM(particle, fieldset, time):
    """Kernel for 2D advection-diffusion,  with advection solved
    using fourth order Runge-Kutta (RK4) and diffusion using the
    Euler-Maruyama scheme (EM). Using the RK4 scheme for diffusion is
    only advantageous in areas where the contribution from diffusion
    is negligible.

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central difference
    gradient approximation. This should be at least an order of magnitude
    less than the typical grid resolution.

    The Wiener increment `dW` should be normally distributed with zero
    mean and a standard deviation of sqrt(dt). Instead, here a uniform
    distribution with the same mean and std is used for efficiency and
    to keep random increments bounded. This substitution is valid for
    the convergence of particle distributions. If convergence of
    individual particle paths is required, use normally distributed
    random increments instead. See Gräwe et al (2012)
    doi.org/10.1007/s10236-012-0523-y for more information.
    """
    # RK4 terms
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    lon1, lat1 = (particle.lon + u1 * .5 * particle.dt, particle.lat + v1 * .5 * particle.dt)
    (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
    lon2, lat2 = (particle.lon + u2 * .5 * particle.dt, particle.lat + v2 * .5 * particle.dt)
    (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
    lon3, lat3 = (particle.lon + u3 * particle.dt, particle.lat + v3 * particle.dt)
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]

    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    dWy = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)

    Kxp1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon + fieldset.dres]
    Kxm1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon - fieldset.dres]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon])

    Kyp1 = fieldset.Kh_meridional[time, particle.depth, particle.lat + fieldset.dres, particle.lon]
    Kym1 = fieldset.Kh_meridional[time, particle.depth, particle.lat - fieldset.dres, particle.lon]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon])

    # Particle positions are updated only after evaluating all terms.
    particle.lon += ((u1 + 2 * u2 + 2 * u3 + u4) / 6. + dKdx) * particle.dt + bx * dWx
    particle.lat += ((v1 + 2 * v2 + 2 * v3 + v4) / 6. + dKdy) * particle.dt + by * dWy
