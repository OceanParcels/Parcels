import random

import numpy as np
import pytest
from scipy import stats

from parcels import (
    Particle,
    ParticleSet,
)
from tests.utils import create_fieldset_zeros_conversion


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("lambd", [1, 5])
def test_randomexponential(lambd):
    fieldset = create_fieldset_zeros_conversion()
    npart = 1000

    # Rate parameter for random.expovariate
    fieldset.lambd = lambd

    # Set random seed
    random.seed(1234)

    pset = ParticleSet(
        fieldset=fieldset, pclass=Particle, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart)
    )

    def vertical_randomexponential(particle, fieldset, time):  # pragma: no cover
        # Kernel for random exponential variable in depth direction
        particle.depth = random.expovariate(fieldset.lambd)

    pset.execute(vertical_randomexponential, runtime=1, dt=1)

    depth = pset.depth
    expected_mean = 1.0 / fieldset.lambd
    assert np.allclose(np.mean(depth), expected_mean, rtol=0.1)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.parametrize("mu", [0.8 * np.pi, np.pi])
@pytest.mark.parametrize("kappa", [2, 4])
def test_randomvonmises(mu, kappa):
    npart = 10000
    fieldset = create_fieldset_zeros_conversion()

    # Parameters for random.vonmisesvariate
    fieldset.mu = mu
    fieldset.kappa = kappa

    # Set random seed
    random.seed(1234)

    AngleParticle = Particle.add_variable("angle")
    pset = ParticleSet(
        fieldset=fieldset, pclass=AngleParticle, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart)
    )

    def vonmises(particle, fieldset, time):  # pragma: no cover
        particle.angle = random.vonmisesvariate(fieldset.mu, fieldset.kappa)

    pset.execute(vonmises, runtime=1, dt=1)

    angles = np.array([p.angle for p in pset])

    assert np.allclose(np.mean(angles), mu, atol=0.1)
    vonmises_mean = stats.vonmises.mean(kappa=kappa, loc=mu)
    assert np.allclose(np.mean(angles), vonmises_mean, atol=0.1)
    vonmises_var = stats.vonmises.var(kappa=kappa, loc=mu)
    assert np.allclose(np.var(angles), vonmises_var, atol=0.1)
