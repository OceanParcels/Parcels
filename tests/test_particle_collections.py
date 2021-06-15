from parcels import FieldSet, ScipyParticle, JITParticle
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
import numpy as np
import pytest

pset_modes = ['soa', 'aos']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


def fieldset(xdim=40, ydim=100):
    U = np.zeros((ydim, xdim), dtype=np.float32)
    V = np.zeros((ydim, xdim), dtype=np.float32)
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(-60, 60, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth}
    return FieldSet.from_data(data, dimensions)


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=40, ydim=100):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_iteration_forward(fieldset, pset_mode, npart=10):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart), pclass=JITParticle)
    assert np.all(np.isclose(np.array([p.id for p in pset]), range(npart)+pset[0].id))


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_iteration_backward(fieldset, pset_mode, npart=10):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart), pclass=JITParticle)
    assert np.all(np.isclose(np.array([p.id for p in reversed(pset)]), pset[0].id+np.arange(npart-1, -1, -1)))


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_get(fieldset, pset_mode, npart=10):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart), pclass=JITParticle)
    assert np.all(np.isclose([pset.collection.get(i).lon for i in range(npart)], np.linspace(0, 1, npart)))


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_get_single_by_index(fieldset, pset_mode, npart=10):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart), pclass=JITParticle)
    assert np.all(np.isclose([pset.collection.get_single_by_index(i).lon for i in range(npart)], np.linspace(0, 1, npart)))


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_get_single_by_ID(fieldset, pset_mode, npart=10):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart), pclass=JITParticle)
    ids = None
    if pset_mode == 'soa':
        ids = pset.collection._data['id']
    elif pset_mode == 'aos':
        ids = np.array([pset.collection._data[i].id for i in range(len(pset))], dtype=np.int64)
    assert np.all(np.isclose([pset.collection.get_single_by_ID(np.int64(i)).lon for i in ids], np.linspace(0, 1, npart)))


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_getattr(fieldset, pset_mode, npart=10):
    lats = np.random.random(npart)
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.linspace(0, 1, npart), lat=lats, pclass=JITParticle)
    assert np.allclose(pset.lat, lats)
