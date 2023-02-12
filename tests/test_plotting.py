from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4, plotTrajectoriesFile)
from parcels.scripts.plottrajectoriesfile import main
from os import path


def test_plotting(tmpdir):
    datafile = path.join(path.dirname(__file__), 'test_data', 'testfields')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    pset = ParticleSet(fieldset=fieldset, lon=0.1, lat=0.2, pclass=JITParticle)

    fp = tmpdir.join("DelayParticle.zarr")
    output_file = pset.ParticleFile(name=fp, outputdt=1)

    pset.execute(AdvectionRK4, runtime=10, dt=1, output_file=output_file)

    try:  # only testing when cartopy is found because Github Action hangs otherwise (see https://github.com/OceanParcels/parcels/actions/runs/4151055784/jobs/7183000370)
        import cartopy  # noqa
        for fld in [fieldset.U, 'vector', None]:
            pset.show(field=fld)
    except:
        pass

    for mode in ['2d', '3d', 'movie2d', 'hist2d']:
        main([mode, f'-p{fp}'])
        plotTrajectoriesFile(fp, tracerfield=fieldset.U, mode=mode, show_plt=False)
