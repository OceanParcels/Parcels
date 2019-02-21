from parcels import FieldSet, Field, ParticleSet, ScipyParticle, JITParticle, BrownianMotion2D
import numpy as np
from datetime import timedelta as delta
from parcels import random
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def mesh_conversion(mesh):
    return (1852. * 60) if mesh == 'spherical' else 1.


def zeros_fieldset(xdim=2, ydim=2, mesh='flat'):
    """Generates a zero velocity field"""

    lon = np.linspace(-2e5/mesh_conversion(mesh), 2e5/mesh_conversion(mesh), xdim, dtype=np.float32)
    lat = np.linspace(-2e5/mesh_conversion(mesh), 2e5/mesh_conversion(mesh), ydim, dtype=np.float32)

    dimensions = {'lon': lon, 'lat': lat}
    data = {'U': np.zeros((ydim, xdim), dtype=np.float32),
            'V': np.zeros((ydim, xdim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_brownian_example(mode, mesh, npart=3000):
    fieldset = zeros_fieldset(mesh=mesh)

    # Set diffusion constants.
    kh_zonal = 100  # in m^2/s
    kh_meridional = 100  # in m^2/s

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

    ys = np.array([p.lat for p in pset]) * mesh_conversion(mesh)
    xs = np.array([p.lon for p in pset]) * mesh_conversion(mesh)  # since near equator, we do not need to care about curvature effect

    tol = 200  # 200m tolerance
    assert np.allclose(np.std(xs), expected_std_x, atol=tol)
    assert np.allclose(np.std(ys), expected_std_y, atol=tol)
    assert np.allclose(np.mean(xs), 0, atol=tol)
    assert np.allclose(np.mean(ys), 0, atol=tol)


if __name__ == "__main__":
    test_brownian_example('jit', 'spherical', npart=2000)
