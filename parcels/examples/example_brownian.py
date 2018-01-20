from parcels import FieldSet, Field, ParticleSet, ScipyParticle, JITParticle, BrownianMotion2D
import numpy as np
from datetime import timedelta as delta
from parcels import random
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def zeros_fieldset(xdim=2, ydim=2):
    """Generates a zero velocity field"""
    lon = np.linspace(-20, 20, xdim, dtype=np.float32)
    lat = np.linspace(-20, 20, ydim, dtype=np.float32)

    dimensions = {'lon': lon, 'lat': lat}
    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh='spherical')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_brownian_example(mode, npart=3000):
    fieldset = zeros_fieldset()

    # Set diffusion constants.
    kh_zonal = 100
    kh_meridional = 100

    # Create field of Kh_zonal and Kh_meridional, using same grid as U
    grid = fieldset.U.grid
    fieldset.add_field(Field('Kh_zonal', kh_zonal*np.ones((2, 2)), grid=grid))
    fieldset.add_field(Field('Kh_meridional', kh_meridional*np.ones((2, 2)), grid=grid))

    # Set random seed
    random.seed(123456)

    runtime = delta(days=1)

    random.seed(1234)
    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode],
                       lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(BrownianMotion2D),
                 runtime=runtime, dt=delta(hours=1))

    expected_std_x = np.sqrt(2*kh_zonal*runtime.total_seconds())
    expected_std_y = np.sqrt(2*kh_meridional*runtime.total_seconds())

    conversion = (1852 * 60)  # to convert from degrees to m
    ys = np.array([p.lat for p in pset]) * conversion
    xs = np.array([p.lon for p in pset]) * conversion  # since near equator, we do not need to care about curvature effect

    tol = 100  # 100m tolerance
    assert np.allclose(np.std(xs), expected_std_x, atol=tol)
    assert np.allclose(np.std(ys), expected_std_y, atol=tol)
    assert np.allclose(np.mean(xs), 0, atol=tol)
    assert np.allclose(np.mean(ys), 0, atol=tol)


if __name__ == "__main__":
    test_brownian_example('jit', npart=2000)
