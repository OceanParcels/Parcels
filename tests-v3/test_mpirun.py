import os
from glob import glob

import numpy as np
import pytest
import xarray as xr

from parcels._compat import MPI
from tests.utils import PROJECT_ROOT


@pytest.mark.skipif(MPI is None, reason="MPI not installed")
@pytest.mark.parametrize("repeatdt, maxage", [(200 * 86400, 600 * 86400), (100 * 86400, 100 * 86400)])
@pytest.mark.parametrize("nump", [8])
def test_mpi_run(tmpdir, repeatdt, maxage, nump):
    stommel_file = PROJECT_ROOT / "docs/examples/example_stommel.py"
    outputMPI = tmpdir.join("StommelMPI")
    outputMPI_partition_function = tmpdir.join("StommelMPI_partition_function")
    outputNoMPI = tmpdir.join("StommelNoMPI.zarr")

    os.system(
        f"mpirun -np 2 python {stommel_file} -p {nump} -o {outputMPI_partition_function} -r {repeatdt} -a {maxage} -cpf True"
    )
    os.system(f"mpirun -np 2 python {stommel_file} -p {nump} -o {outputMPI} -r {repeatdt} -a {maxage}")
    os.system(f"python {stommel_file} -p {nump} -o {outputNoMPI} -r {repeatdt} -a {maxage}")

    ds2 = xr.open_zarr(outputNoMPI)

    for mpi_run in [outputMPI, outputMPI_partition_function]:
        files = glob(os.path.join(mpi_run, "proc*"))
        ds1 = xr.concat(
            [xr.open_zarr(f) for f in files], dim="trajectory", compat="no_conflicts", coords="minimal"
        ).sortby(["trajectory"])

        for v in ds2.variables.keys():
            if v == "time":
                continue  # skip because np.allclose does not work well on np.datetime64
            assert np.allclose(ds1.variables[v][:], ds2.variables[v][:], equal_nan=True)

        for a in ds2.attrs:
            if a != "parcels_version":
                assert ds1.attrs[a] == ds2.attrs[a]

        ds1.close()
    ds2.close()
