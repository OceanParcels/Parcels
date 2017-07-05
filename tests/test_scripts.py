from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     plotTrajectoriesFile, convert_IndexedOutputToArray)
from datetime import timedelta as delta
import numpy as np
import pytest
from os import path, pardir


def create_outputfiles(dir):
    datafile = path.join(path.dirname(__file__), pardir, 'examples',
                         'Peninsula_data', 'peninsula')

    fieldset = FieldSet.from_nemo(datafile, allow_time_extrapolation=True)
    pset = ParticleSet(fieldset=fieldset, lon=[], lat=[], pclass=JITParticle)
    npart = 10
    delaytime = delta(hours=1)
    endtime = delta(hours=24)
    x = 3. * (1. / 1.852 / 60)
    y = (fieldset.U.lat[0] + x, fieldset.U.lat[-1] - x)
    lat = np.linspace(y[0], y[1], npart, dtype=np.float32)

    fp_index = dir.join("DelayParticle")
    output_file = pset.ParticleFile(name=fp_index, type="indexed")

    for t in range(npart):
        pset.add(JITParticle(lon=x, lat=lat[t], fieldset=fieldset))
        pset.execute(AdvectionRK4, runtime=delaytime, dt=delta(minutes=5),
                     interval=delaytime, starttime=delaytime*t, output_file=output_file)

    pset.execute(AdvectionRK4, runtime=endtime-npart*delaytime, starttime=delaytime*npart,
                 dt=delta(minutes=5), interval=delta(hours=1), output_file=output_file)

    fp_array = dir.join("DelayParticle_array")
    convert_IndexedOutputToArray(fp_index+'.nc', fp_array+'.nc')
    return fp_index, fp_array


@pytest.mark.parametrize('mode', ['2d', '3d', 'movie2d'])
@pytest.mark.parametrize('fp_type', ['index', 'array'])
def test_plotting(mode, tmpdir, fp_type):
    fp_index, fp_array = create_outputfiles(tmpdir)
    fp = fp_array if fp_type == 'array' else fp_index
    plotTrajectoriesFile(fp+'.nc', mode=mode, show_plt=False)
