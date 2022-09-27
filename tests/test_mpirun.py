from os import path, system
from netCDF4 import Dataset
import numpy as np
import pytest
import sys
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
        outputMPI = tmpdir.join('StommelMPI.nc')
        outputNoMPI = tmpdir.join('StommelNoMPI.nc')

        system('mpirun -np 2 python %s -p 4 -o %s -r %d -a %d -psm %s' % (stommel_file, outputMPI, repeatdt, maxage, pset_mode))
        system('python %s -p 4 -o %s -r %d -a %d -psm %s' % (stommel_file, outputNoMPI, repeatdt, maxage, pset_mode))

        ncfile1 = Dataset(outputMPI, 'r', 'NETCDF4')
        ncfile2 = Dataset(outputNoMPI, 'r', 'NETCDF4')

        for v in ncfile2.variables.keys():
            assert np.allclose(ncfile1.variables[v][:], ncfile2.variables[v][:])

        for a in ncfile2.ncattrs():
            if a != 'parcels_version':
                assert getattr(ncfile1, a) == getattr(ncfile2, a)

        ncfile1.close()
        ncfile2.close()
