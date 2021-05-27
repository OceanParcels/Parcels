from parcels import (FieldSet, JITParticle, AdvectionRK4, plotTrajectoriesFile)
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
from datetime import timedelta as delta
import numpy as np
import pytest
from os import path
from parcels.tools.loggers import logger
import sys

pset_modes = ['soa', 'aos']
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


def create_outputfiles(dir, pset_mode):
    datafile = path.join(path.dirname(__file__), 'test_data', 'testfields')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    pset = pset_type[pset_mode]['pset'](fieldset=fieldset, lon=[], lat=[], pclass=JITParticle)
    npart = 10
    delaytime = delta(hours=1)
    endtime = delta(hours=24)
    x = 3. * (1. / 1.852 / 60)
    y = (fieldset.U.lat[0] + x, fieldset.U.lat[-1] - x)
    lat = np.linspace(y[0], y[1], npart)

    fp = dir.join("DelayParticle.nc")
    output_file = pset.ParticleFile(name=fp, outputdt=delaytime)

    for t in range(npart):
        time = 0 if len(pset) == 0 else pset[0].time
        pset.add(pset_type[pset_mode]['pset'](pclass=JITParticle, lon=x, lat=lat[t], fieldset=fieldset, time=time))
        pset.execute(AdvectionRK4, runtime=delaytime, dt=delta(minutes=5), output_file=output_file)

    pset.execute(AdvectionRK4, runtime=endtime-npart*delaytime,
                 dt=delta(minutes=5), output_file=output_file)
    output_file.close()

    return fp


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['2d', '3d', 'movie2d', 'hist2d'])
def test_plotting(pset_mode, mode, tmpdir):
    if mode == '3d' and sys.platform in ['linux', 'linux2']:
        logger.info('Skipping 3d test in linux Travis, since it fails to find display to connect')
        return
    fp = create_outputfiles(tmpdir, pset_mode)
    plotTrajectoriesFile(fp, mode=mode, show_plt=False)
