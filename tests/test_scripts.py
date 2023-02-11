from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4, plotTrajectoriesFile)
from parcels.scripts.plottrajectoriesfile import main
import pytest
from os import path


@pytest.mark.parametrize('from_main', [True, False])
def test_plotting(from_main, tmpdir):
    datafile = path.join(path.dirname(__file__), 'test_data', 'testfields')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    pset = ParticleSet(fieldset=fieldset, lon=0.1, lat=0.2, pclass=JITParticle)

    fp = tmpdir.join("DelayParticle.zarr")
    output_file = pset.ParticleFile(name=fp, outputdt=1)

    pset.execute(AdvectionRK4, runtime=10, dt=1, output_file=output_file)

    for mode in ['2d', '3d', 'movie2d', 'hist2d']:
        if from_main:
            pset.show(field=fieldset.U)
            main([mode, f'-p{fp}'])
        else:
            plotTrajectoriesFile(fp, tracerfield=fieldset.U, mode=mode, show_plt=False)
