"""Module controlling the writing of ParticleSets to parquet file."""
import os
import shutil
from abc import ABC
from datetime import timedelta as delta
from datetime import datetime as datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parcels.tools.loggers import logger

try:
    from mpi4py import MPI
except:
    MPI = None
try:
    from parcels._version import version as parcels_version
except:
    raise OSError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


__all__ = ['BaseParticleFile']


class BaseParticleFile(ABC):
    """Initialise trajectory output.

    Parameters
    ----------
    name : str
        Basename of the output file. This can also be a Zarr store object.  # TODO make sure can also write to parquet store?
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a timedelta object or a positive double.

    Returns
    -------
    BaseParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    outputdt = None
    lasttime_written = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None

    def __init__(self, name, particleset, outputdt=np.infty):

        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

        self.particleset = particleset
        self.parcels_mesh = 'spherical'
        if self.particleset.fieldset is not None:
            self.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
        self.time_origin = self.particleset.time_origin
        self.lonlatdepth_dtype = self.particleset.collection.lonlatdepth_dtype
        self.vars_to_write = {}
        for var in self.particleset.collection.ptype.variables:
            if var.to_write:
                self.vars_to_write[var.name] = var.dtype
        self.mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

        self.metadata = {"feature_type": "trajectory",
                         "Conventions": "CF-1.6/CF-1.7",
                         "parcels_version": parcels_version,
                         "parcels_mesh": self.parcels_mesh}

        if False:  # if issubclass(type(name), zarr.storage.Store):
            #     # If we already got a Zarr store, we won't need any of the naming logic below.
            #     # But we need to handle incompatibility with MPI mode for now:
            #     if MPI and MPI.COMM_WORLD.Get_size() > 1:
            #         raise ValueError("Currently, MPI mode is not compatible with directly passing a Zarr store.")
            #     self.fname = name
            #     self.store = name
            pass  # TODO implement parquet store?
        else:
            extension = os.path.splitext(str(name))[1]
            if extension in ['.parquet', '.pqt', '.parq', '']:
                pass
            elif extension in ['.nc', '.nc4']:
                raise RuntimeError('Output in NetCDF is not supported anymore. Use .parquet or extension for ParticleFile name.')
            elif extension in ['.zarr']:
                raise RuntimeError('Output in zarr is not supported anymore. Use .parquet extension for ParticleFile name.')
            else:
                raise RuntimeError(f"Output format {extension} not supported. Use .parquet extension for ParticleFile name.")

            if MPI and MPI.COMM_WORLD.Get_size() > 1:
                self.fname = os.path.join(name, f"proc{self.mpi_rank:02d}.parquet")
                if extension in ['.parquet', '.pqt', '.parq']:
                    logger.warning(f'The ParticleFile name contains .parquet extension, but parquet files will be written per processor in MPI mode at {self.fname}')
            else:
                self.fname = name if extension in ['.parquet', '.pqt', '.parq'] else "%s.parquet" % name
                self.nfiles = 0
                parquet_folder = Path(self.fname)

                if parquet_folder.exists():
                    shutil.rmtree(parquet_folder)
                parquet_folder.mkdir(parents=True)

    def add_metadata(self, name, message):  # TODO check if metadata can be added in parquet
        """Add metadata to :class:`parcels.particleset.ParticleSet`.

        Parameters
        ----------
        name : str
            Name of the metadata variabale
        message : str
            message to be written
        """
        self.metadata[name] = str(message)

    def _convert_varout_name(self, var):
        if var == 'depth':
            return 'z'
        elif var == 'id':
            return 'trajectory'
        else:
            return var

    def write(self, pset, time):
        """Write all data from one time step to the parquet file.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet
        """
        time = time.total_seconds() if isinstance(time, delta) else time

        if (self.lasttime_written is None or ~np.isclose(self.lasttime_written, time)):
            if pset.collection._ncount == 0:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)
                return

            indices_to_write = pset.collection._to_write_particles(pset.collection._data, time)
            self.lasttime_written = time

            if len(indices_to_write) > 0:
                trajectory = pset.collection.getvardata('id', indices_to_write)

                dfdict = {}
                for var in self.vars_to_write:
                    varout = self._convert_varout_name(var)
                    if varout == 'time':
                        dftime = self.time_origin.fulltime(pset.collection.getvardata(var, indices_to_write))
                        if self.time_origin.calendar in ['360_day']:
                            logger.warning_once("360_day calendar is not supported in Parquet output. Converting output to standard calendar.")
                            dftime = [datetime.strptime(str(t), '%Y-%m-%d %H:%M:%S') for t in dftime]
                        elif self.time_origin.calendar is None:
                            dftime = (np.round(dftime*1e9)).astype('timedelta64[ns]')  # to avoid rounding errors for negative times
                    elif varout not in ['trajectory', 'obs']:  # because 'trajectory' and 'obs' are written as index
                        dfdict[varout] = pset.collection.getvardata(var, indices_to_write)
                index = pd.MultiIndex.from_tuples(list(zip(trajectory, dftime)), names=['trajectory', 'time'])
                table = pa.Table.from_pandas(pd.DataFrame(data=dfdict, index=index))
                metadata = {**self.metadata, **(table.schema.metadata or {})}
                table = table.replace_schema_metadata(metadata)

                fname = os.path.join(f"{self.fname}", f"p{self.nfiles:03d}.parquet")
                pq.write_table(table, fname, compression='GZIP')

                self.nfiles += 1
