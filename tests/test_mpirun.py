import sys
from glob import glob
from os import path, system

import numpy as np
import pytest
import xarray as xr


@pytest.mark.skipif(sys.platform.startswith("win"), reason="skipping windows as mpi4py not available for windows")
@pytest.mark.parametrize('repeatdt, maxage', [(200*86400, 600*86400), (100*86400, 100*86400)])
@pytest.mark.parametrize('nump', [8])
def test_mpi_run(tmpdir, repeatdt, maxage, nump):
    stommel_file = path.join(path.dirname(__file__), '..', 'docs', 'examples', 'example_stommel.py')
    outputMPI = tmpdir.join('StommelMPI')
    outputMPI_partition_function = tmpdir.join('StommelMPI_partition_function')
    outputNoMPI = tmpdir.join('StommelNoMPI.zarr')

    system('mpirun -np 2 python %s -p %d -o %s -r %d -a %d -wf False -cpf True' % (stommel_file, nump, outputMPI_partition_function, repeatdt, maxage))
    system('mpirun -np 2 python %s -p %d -o %s -r %d -a %d -wf False' % (stommel_file, nump, outputMPI, repeatdt, maxage))
    system('python %s -p %d -o %s -r %d -a %d -wf False' % (stommel_file, nump, outputNoMPI, repeatdt, maxage))

    ds2 = xr.open_zarr(outputNoMPI)

    for mpi_run in [outputMPI, outputMPI_partition_function]:
        files = glob(path.join(mpi_run, "proc*"))
        ds1 = xr.concat([xr.open_zarr(f) for f in files], dim='trajectory',
                        compat='no_conflicts', coords='minimal').sortby(['trajectory'])

        for v in ds2.variables.keys():
            if v == 'time':
                continue  # skip because np.allclose does not work well on np.datetime64
            assert np.allclose(ds1.variables[v][:], ds2.variables[v][:], equal_nan=True)

        for a in ds2.attrs:
            if a != 'parcels_version':
                assert ds1.attrs[a] == ds2.attrs[a]

        ds1.close()
    ds2.close()
