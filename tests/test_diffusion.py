from parcels import (FieldSet, Field, RectilinearZGrid, ParticleSet, BrownianMotion2D,
                     JITParticle, ScipyParticle)
from parcels import rng as random
from datetime import timedelta as delta
import numpy as np
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def zeros_fieldset(mesh='spherical', xdim=200, ydim=100, mesh_conversion=1):
    """Generates a zero velocity field"""
    lon = np.linspace(-1e5*mesh_conversion, 1e5*mesh_conversion, xdim, dtype=np.float32)
    lat = np.linspace(-1e5*mesh_conversion, 1e5*mesh_conversion, ydim, dtype=np.float32)

    dimensions = {'lon': lon, 'lat': lat}
    data = {'U': np.zeros((ydim, xdim), dtype=np.float32),
            'V': np.zeros((ydim, xdim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


@pytest.mark.parametrize('mesh', ['spherical', 'flat'])
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldKh_Brownian(mesh, mode, xdim=200, ydim=100, kh_zonal=100, kh_meridional=50):
    mesh_conversion = 1/1852./60 if mesh is 'spherical' else 1
    fieldset = zeros_fieldset(mesh=mesh, xdim=xdim, ydim=ydim, mesh_conversion=mesh_conversion)

    vec = np.linspace(-1e5*mesh_conversion, 1e5*mesh_conversion, 2)
    grid = RectilinearZGrid(lon=vec, lat=vec, mesh=mesh)

    fieldset.add_field(Field('Kh_zonal', kh_zonal*np.ones((2, 2)), grid=grid))
    fieldset.add_field(Field('Kh_meridional', kh_meridional*np.ones((2, 2)), grid=grid))

    npart = 1000
    runtime = delta(days=1)

    random.seed(1234)
    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode],
                       lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(BrownianMotion2D),
                 runtime=runtime, dt=delta(hours=1))

    expected_std_lon = np.sqrt(2*kh_zonal*mesh_conversion**2*runtime.total_seconds())
    expected_std_lat = np.sqrt(2*kh_meridional*mesh_conversion**2*runtime.total_seconds())

    lats = np.array([p.lat for p in pset])
    lons = np.array([p.lon for p in pset])

    tol = 200*mesh_conversion  # effectively 200 m errors
    assert np.allclose(np.std(lats), expected_std_lat, atol=tol)
    assert np.allclose(np.std(lons), expected_std_lon, atol=tol)
    assert np.allclose(np.mean(lons), 0, atol=tol)
    assert np.allclose(np.mean(lats), 0, atol=tol)
