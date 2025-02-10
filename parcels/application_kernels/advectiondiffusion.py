"""Collection of pre-built advection-diffusion kernels.

See `this tutorial <../examples/tutorial_diffusion.ipynb>`__ for a detailed explanation.
"""

import math

import parcels

__all__ = ["AdvectionDiffusionEM", "AdvectionDiffusionM1", "DiffusionUniformKh"]


def AdvectionDiffusionM1(particle, fieldset, time):  # pragma: no cover
    """Kernel for 2D advection-diffusion, solved using the Milstein scheme at first order (M1).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central
    difference gradient approximation. This should be (of the order of) the
    local gridsize.

    This Milstein scheme is of strong and weak order 1, which is higher than the
    Euler-Maruyama scheme. It experiences less spurious diffusivity by
    including extra correction terms that are computationally cheap.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = parcels.rng.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = parcels.rng.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    Kxp1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon + fieldset.dres]
    Kxm1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon - fieldset.dres]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)

    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon])

    Kyp1 = fieldset.Kh_meridional[time, particle.depth, particle.lat + fieldset.dres, particle.lon]
    Kym1 = fieldset.Kh_meridional[time, particle.depth, particle.lat - fieldset.dres, particle.lon]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)

    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon])

    # Particle positions are updated only after evaluating all terms.
    particle_dlon += u * particle.dt + 0.5 * dKdx * (dWx**2 + particle.dt) + bx * dWx  # noqa
    particle_dlat += v * particle.dt + 0.5 * dKdy * (dWy**2 + particle.dt) + by * dWy  # noqa


def AdvectionDiffusionEM(particle, fieldset, time):  # pragma: no cover
    """Kernel for 2D advection-diffusion, solved using the Euler-Maruyama scheme (EM).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central
    difference gradient approximation. This should be (of the order of) the
    local gridsize.

    The Euler-Maruyama scheme is of strong order 0.5 and weak order 1.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = parcels.rng.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = parcels.rng.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    Kxp1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon + fieldset.dres]
    Kxm1 = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon - fieldset.dres]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)
    ax = u + dKdx
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon])

    Kyp1 = fieldset.Kh_meridional[time, particle.depth, particle.lat + fieldset.dres, particle.lon]
    Kym1 = fieldset.Kh_meridional[time, particle.depth, particle.lat - fieldset.dres, particle.lon]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)
    ay = v + dKdy
    by = math.sqrt(2 * fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon])

    # Particle positions are updated only after evaluating all terms.
    particle_dlon += ax * particle.dt + bx * dWx  # noqa
    particle_dlat += ay * particle.dt + by * dWy  # noqa


def DiffusionUniformKh(particle, fieldset, time):  # pragma: no cover
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.

    Assumes that fieldset has constant fields `Kh_zonal` and `Kh_meridional`.
    These can be added via e.g.
    `fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)`
    or
    `fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)`
    where mesh is either 'flat' or 'spherical'

    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
    Since the perturbation due to diffusion is in this case isotropic independent, this
    kernel contains no advection and can be used in combination with a separate
    advection kernel.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = parcels.rng.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    dWy = parcels.rng.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
    by = math.sqrt(2 * fieldset.Kh_meridional[particle])

    particle_dlon += bx * dWx  # noqa
    particle_dlat += by * dWy  # noqa
