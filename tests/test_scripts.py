from parcels import (FieldSet, JITParticle, AdvectionRK4, plotTrajectoriesFile)
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
import pytest
from os import path

pset_modes = ['soa', 'aos']
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_plotting(pset_mode, tmpdir):
    datafile = path.join(path.dirname(__file__), 'test_data', 'testfields')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    pset = pset_type[pset_mode]['pset'](fieldset=fieldset, lon=0.1, lat=0.2, pclass=JITParticle)

    fp = tmpdir.join("DelayParticle.zarr")
    output_file = pset.ParticleFile(name=fp, outputdt=1)

    pset.execute(AdvectionRK4, runtime=10, dt=1, output_file=output_file)

    for mode in ['2d', '3d', 'movie2d', 'hist2d']:
        plotTrajectoriesFile(fp, tracerfield=fieldset.U, mode=mode, show_plt=False)
