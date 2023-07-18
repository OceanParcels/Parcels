"""Module controlling the writing of ParticleSets to parquet file."""
import os
import shutil
from abc import ABC
from datetime import timedelta as delta
from pathlib import Path

# import fastparquet as fpq  # needed because pyarrow can't append to parquet files (https://github.com/apache/arrow/issues/33362)
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parcels.tools.loggers import logger
from parcels.tools.statuscodes import OperationCode

try:
    from mpi4py import MPI
except:
    MPI = None
try:
    from parcels._version import version as parcels_version
except:
    raise OSError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


__all__ = ['BaseParticleFile']


def _set_calendar(origin_calendar):
    if origin_calendar == 'np_datetime64':
        return 'standard'
    else:
        return origin_calendar


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
    write_ondelete : bool
        Whether to write particle data only when they are deleted. Default is False

    Returns
    -------
    BaseParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    write_ondelete = None
    outputdt = None
    lasttime_written = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False):

        self.write_ondelete = write_ondelete
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

        # Create dictionary to translate datatypes and fill_values
        self.fmt_map = {np.float16: 'f2', np.float32: 'f4', np.float64: 'f8',
                        np.bool_: 'i1', np.int8: 'i1', np.int16: 'i2',
                        np.int32: 'i4', np.int64: 'i8', np.uint8: 'u1',
                        np.uint16: 'u2', np.uint32: 'u4', np.uint64: 'u8'}
        self.fill_value_map = {np.float16: np.nan, np.float32: np.nan, np.float64: np.nan,
                               np.bool_: np.iinfo(np.int8).max, np.int8: np.iinfo(np.int8).max,
                               np.int16: np.iinfo(np.int16).max, np.int32: np.iinfo(np.int32).max,
                               np.int64: np.iinfo(np.int64).max, np.uint8: np.iinfo(np.uint8).max,
                               np.uint16: np.iinfo(np.uint16).max, np.uint32: np.iinfo(np.uint32).max,
                               np.uint64: np.iinfo(np.uint64).max}
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

    def _create_variables_attribute_dict(self):
        """Creates the dictionary with variable attributes.

        Notes
        -----
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        # TODO check if attributes can be added in parquet
        attrs = {'z': {"long_name": "",
                       "standard_name": "depth",
                       "units": "m",
                       "positive": "down"},
                 'trajectory': {"long_name": "Unique identifier for each particle",
                                "cf_role": "trajectory_id",
                                "_FillValue": self.fill_value_map[np.int64]},
                 'time': {"long_name": "",
                          "standard_name": "time",
                          "units": "seconds",
                          "axis": "T"},
                 'lon': {"long_name": "",
                         "standard_name": "longitude",
                         "units": "degrees_east",
                         "axis":
                             "X"},
                 'lat': {"long_name": "",
                         "standard_name": "latitude",
                         "units": "degrees_north",
                         "axis": "Y"}}

        if self.time_origin.calendar is not None:
            attrs['time']['units'] = "seconds since " + str(self.time_origin)
            attrs['time']['calendar'] = 'standard' if self.time_origin.calendar == 'np_datetime64' else self.time_origin.calendar

        for vname in self.vars_to_write:
            attrs[vname] = {"_FillValue": self.fill_value_map[self.vars_to_write[vname]],
                            "long_name": "",
                            "standard_name": vname,
                            "units": "unknown"}

        return attrs

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

    def write(self, pset, time, deleted_only=False):
        """Write all data from one time step to the parquet file.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet
        deleted_only :
            Flag to write only the deleted Particles (Default value = False)
        """
        time = time.total_seconds() if isinstance(time, delta) else time

        if self.lasttime_written != time and (self.write_ondelete is False or deleted_only is not False):
            if pset.collection._ncount == 0:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)
                return

            if deleted_only is not False:
                if type(deleted_only) not in [list, np.ndarray] and deleted_only in [True, 1]:
                    indices_to_write = np.where(np.isin(pset.collection.getvardata('state'), [OperationCode.Delete]))[0]
                elif type(deleted_only) == np.ndarray:
                    if set(deleted_only).issubset([0, 1]):
                        indices_to_write = np.where(deleted_only)[0]
                    else:
                        indices_to_write = deleted_only
                elif type(deleted_only) == list:
                    indices_to_write = np.array(deleted_only)
            else:
                indices_to_write = pset.collection._to_write_particles(pset.collection._data, time)
                self.lasttime_written = time

            if len(indices_to_write) > 0:
                trajectory = pset.collection.getvardata('id', indices_to_write)
                obs = pset.collection.getvardata('obs', indices_to_write)
                index = pd.MultiIndex.from_tuples(list(zip(trajectory, obs)), names=['trajectory', 'obs'])

                dfdict = {}
                for var in self.vars_to_write:
                    varout = self._convert_varout_name(var)
                    if varout == 'time':
                        dfdict[varout] = self.time_origin.fulltime(pset.collection.getvardata(var, indices_to_write))
                        if self.time_origin.calendar is None:
                            dfdict[varout] = dfdict[varout].astype('timedelta64[s]')
                    elif varout not in ['trajectory', 'obs']:  # because 'trajectory' and 'obs' are written as index
                        dfdict[varout] = pset.collection.getvardata(var, indices_to_write)

                table = pa.Table.from_pandas(pd.DataFrame(data=dfdict, index=index))
                metadata = {**self.metadata, **(table.schema.metadata or {})}
                table = table.replace_schema_metadata(metadata)

                fname = self.fname + '/p%03d.parquet' % self.nfiles
                pq.write_table(table, fname, compression='GZIP')

                self.nfiles += 1
                pset.collection.setvardata('obs', indices_to_write, obs+1)

                # TODO remove this version using fastparquet
                # if self.create_new_zarrfile:
                #     fpq.write(self.fname, df, compression='GZIP', append=False)
                #     self.create_new_zarrfile = False
                # else:
                #     fpq.write(self.fname, df, compression='GZIP', append=True)
