from os import path, system
from netCDF4 import Dataset
import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None


def test_mpi_run(tmpdir):
    if MPI:
        repeatdt = 200*86400
        stommel_file = path.join(path.dirname(__file__), '..', 'parcels',
                                 'examples', 'example_stommel.py')
        outputMPI = tmpdir.join('StommelMPI.nc')
        outputNoMPI = tmpdir.join('StommelNoMPI.nc')

        system('mpirun -np 2 python %s -p 4 -o %s -r %d' % (stommel_file, outputMPI, repeatdt))
        system('python %s -p 4 -o %s -r %d' % (stommel_file, outputNoMPI, repeatdt))

        ncfile1 = Dataset(outputMPI, 'r', 'NETCDF4')
        ncfile2 = Dataset(outputNoMPI, 'r', 'NETCDF4')

        for v in ncfile2.variables.keys():
            assert np.allclose(ncfile1.variables[v][:], ncfile2.variables[v][:])

        for a in ncfile2.ncattrs():
            if a != 'parcels_version':
                assert getattr(ncfile1, a) == getattr(ncfile2, a)

        ncfile1.close()
        ncfile2.close()
