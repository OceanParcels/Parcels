"""Collection of pre-built advection-diffusion kernels.

See `this tutorial <../examples/tutorial_diffusion.ipynb>`__ for a detailed explanation.
"""

import numpy as np

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
    dt = particle.dt / np.timedelta64(1, "s")  # TODO: improve API for converting dt to seconds
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(dt)))

    Kxp1 = fieldset.Kh_zonal[particle.time, particle.depth, particle.lat, particle.lon + fieldset.dres, particle]
    Kxm1 = fieldset.Kh_zonal[particle.time, particle.depth, particle.lat, particle.lon - fieldset.dres, particle]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)

    u, v = fieldset.UV[particle.time, particle.depth, particle.lat, particle.lon, particle]
    bx = np.sqrt(2 * fieldset.Kh_zonal[particle.time, particle.depth, particle.lat, particle.lon, particle])

    Kyp1 = fieldset.Kh_meridional[particle.time, particle.depth, particle.lat + fieldset.dres, particle.lon, particle]
    Kym1 = fieldset.Kh_meridional[particle.time, particle.depth, particle.lat - fieldset.dres, particle.lon, particle]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)

    by = np.sqrt(2 * fieldset.Kh_meridional[particle.time, particle.depth, particle.lat, particle.lon, particle])

    # Particle positions are updated only after evaluating all terms.
    particle.dlon += u * dt + 0.5 * dKdx * (dWx**2 + dt) + bx * dWx
    particle.dlat += v * dt + 0.5 * dKdy * (dWy**2 + dt) + by * dWy


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
    dt = particle.dt / np.timedelta64(1, "s")
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(dt)))

    u, v = fieldset.UV[particle.time, particle.depth, particle.lat, particle.lon, particle]

    Kxp1 = fieldset.Kh_zonal[particle.time, particle.depth, particle.lat, particle.lon + fieldset.dres, particle]
    Kxm1 = fieldset.Kh_zonal[particle.time, particle.depth, particle.lat, particle.lon - fieldset.dres, particle]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)
    ax = u + dKdx
    bx = np.sqrt(2 * fieldset.Kh_zonal[particle.time, particle.depth, particle.lat, particle.lon, particle])

    Kyp1 = fieldset.Kh_meridional[particle.time, particle.depth, particle.lat + fieldset.dres, particle.lon, particle]
    Kym1 = fieldset.Kh_meridional[particle.time, particle.depth, particle.lat - fieldset.dres, particle.lon, particle]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)
    ay = v + dKdy
    by = np.sqrt(2 * fieldset.Kh_meridional[particle.time, particle.depth, particle.lat, particle.lon, particle])

    # Particle positions are updated only after evaluating all terms.
    particle.dlon += ax * dt + bx * dWx
    particle.dlat += ay * dt + by * dWy


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
    dt = particle.dt / np.timedelta64(1, "s")
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(dt)))

    print(particle)

    bx = np.sqrt(2 * fieldset.Kh_zonal[particle])
    by = np.sqrt(2 * fieldset.Kh_meridional[particle])

    particle.dlon += bx * dWx
    particle.dlat += by * dWy
