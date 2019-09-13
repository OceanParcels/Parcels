import os
from netCDF4 import Dataset
import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None


def test_mpi_run():
    if MPI:
        repeatdt = 200*86400
        parcels_path = os.path.dirname(__file__) + '/../parcels'
        os.system('mpirun -np 2 python %s/examples/example_stommel.py -p 4 -o StommelMPI.nc -r %d' % (parcels_path, repeatdt))
        os.system('python %s/examples/example_stommel.py -p 4 -o StommelNoMPI.nc -r %d' % (parcels_path, repeatdt))

        ncfile1 = Dataset('StommelMPI.nc', 'r', 'NETCDF4')
        ncfile2 = Dataset('StommelNoMPI.nc', 'r', 'NETCDF4')

        for v in ncfile2.variables.keys():
            assert np.allclose(ncfile1.variables[v][:], ncfile2.variables[v][:])

        for a in ncfile2.ncattrs():
            if a != 'parcels_version':
                assert getattr(ncfile1, a) == getattr(ncfile2, a)

        ncfile1.close()
        ncfile2.close()
