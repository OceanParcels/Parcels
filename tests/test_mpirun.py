from os import path, system
import numpy as np
import pytest
import sys
import xarray as xr
try:
    from mpi4py import MPI
except:
    MPI = None


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="skipping macOS test as problem with file in pytest")
@pytest.mark.parametrize('pset_mode', ['soa', 'aos'])
@pytest.mark.parametrize('repeatdt', [200*86400, 10*86400])
@pytest.mark.parametrize('maxage', [600*86400, 10*86400])
def test_mpi_run(pset_mode, tmpdir, repeatdt, maxage):
    if MPI:
        stommel_file = path.join(path.dirname(__file__), '..', 'parcels',
                                 'examples', 'example_stommel.py')
        outputMPI = tmpdir.join('StommelMPI.zarr')
        outputNoMPI = tmpdir.join('StommelNoMPI.zarr')

        system('mpirun -np 2 python %s -p 4 -o %s -r %d -a %d -psm %s' % (stommel_file, outputMPI, repeatdt, maxage, pset_mode))
        system('python %s -p 4 -o %s -r %d -a %d -psm %s' % (stommel_file, outputNoMPI, repeatdt, maxage, pset_mode))

        ds1 = xr.open_zarr(outputMPI)
        ds2 = xr.open_zarr(outputNoMPI)

        for v in ds2.variables.keys():
            if v == 'time':
                continue  # skip because np.allclose does not work well on np.datetime64
            assert np.allclose(ds1.variables[v][:], ds2.variables[v][:], equal_nan=True)

        for a in ds2.attrs:
            if a != 'parcels_version':
                assert ds1.attrs[a] == ds2.attrs[a]

        ds1.close()
        ds2.close()
