"""Module controlling the writing of ParticleSets to Zarr file"""
from abc import ABC
from abc import abstractmethod
from datetime import timedelta as delta
import os
import numpy as np
import xarray as xr
import zarr

from parcels.tools.loggers import logger
from parcels.tools.statuscodes import OperationCode

try:
    from mpi4py import MPI
except:
    MPI = None
try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


__all__ = ['BaseParticleFile']


def _set_calendar(origin_calendar):
    if origin_calendar == 'np_datetime64':
        return 'standard'
    else:
        return origin_calendar


class BaseParticleFile(ABC):
    """Initialise trajectory output.

    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param chunks: Tuple (trajs, obs) to control the size of chunks in the zarr output.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param create_new_zarrfile: Boolean to create a new file. Default is True
    """
    write_ondelete = None
    outputdt = None
    lasttime_written = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None

    def __init__(self, name, particleset, outputdt=np.infty, chunks=None, write_ondelete=False,
                 create_new_zarrfile=True):

        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.chunks = chunks
        self.lasttime_written = None  # variable to check if time has been written already

        self.particleset = particleset
        self.parcels_mesh = 'spherical'
        if self.particleset.fieldset is not None:
            self.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
        self.time_origin = self.particleset.time_origin
        self.lonlatdepth_dtype = self.particleset.collection.lonlatdepth_dtype
        self.maxids = 0
        self.obs_written = np.empty((0,), dtype=int)
        self.pids_written = {}
        self.create_new_zarrfile = create_new_zarrfile
        self.vars_to_write = {}
        for var in self.particleset.collection.ptype.variables:
            if var.to_write:
                self.vars_to_write[var.name] = var.dtype
        self.mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

        # Reset once-written flag of each particle, in case new ParticleFile created for a ParticleSet
        particleset.collection.setallvardata('once_written', 0)

        self.metadata = {"feature_type": "trajectory", "Conventions": "CF-1.6/CF-1.7",
                         "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
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

        extension = os.path.splitext(str(name))[1]
        if extension in ['.nc', '.nc4']:
            raise RuntimeError('Output in NetCDF is not supported anymore. Use .zarr extension for ParticleFile name.')
        if MPI and MPI.COMM_WORLD.Get_size() > 1:
            self.fname = os.path.join(name, f"proc{self.mpi_rank:02d}.zarr")
            if extension in ['.zarr']:
                logger.warning(f'The ParticleFile name contains .zarr extension, but zarr files will be written per processor in MPI mode at {self.fname}')
        else:
            self.fname = name if extension in ['.zarr'] else "%s.zarr" % name

    @abstractmethod
    def _reserved_var_names(self):
        """
        returns the reserved dimension names not to be written just once.
        """
        pass

    def _create_variables_attribute_dict(self):
        """
        creates the dictionary with variable attributes.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """

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
            if vname not in self._reserved_var_names():
                attrs[vname] = {"_FillValue": self.fill_value_map[self.vars_to_write[vname]],
                                "long_name": "",
                                "standard_name": vname,
                                "units": "unknown"}

        return attrs

    def __del__(self):
        self.close()

    def close(self, delete_tempfiles=True):
        pass

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`

        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        self.metadata[name] = message

    def _convert_varout_name(self, var):
        if var == 'depth':
            return 'z'
        elif var == 'id':
            return 'trajectory'
        else:
            return var

    def write_once(self, var):
        return self.particleset.collection.ptype[var].to_write == 'once'

    def _extend_zarr_dims(self, Z, store, dtype, axis):
        if axis == 1:
            a = np.full((Z.shape[0], self.chunks[1]), np.nan, dtype=dtype)
            obs = zarr.group(store=store, overwrite=False)["obs"]
            if len(obs) == Z.shape[1]:
                obs.append(np.arange(self.chunks[1])+obs[-1]+1)
        else:
            extra_trajs = max(self.maxids - Z.shape[0], self.chunks[0])
            if len(Z.shape) == 2:
                a = np.full((extra_trajs, Z.shape[1]), np.nan, dtype=dtype)
            else:
                a = np.full((extra_trajs,), np.nan, dtype=dtype)
        Z.append(a, axis=axis)
        zarr.consolidate_metadata(store)

    def write(self, pset, time, deleted_only=False):
        """Write all data from one time step to the zarr file

        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
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
                pids = pset.collection.getvardata('id', indices_to_write)
                to_add = sorted(set(pids) - set(self.pids_written.keys()))
                for i, pid in enumerate(to_add):
                    self.pids_written[pid] = self.maxids + i
                ids = np.array([self.pids_written[p] for p in pids], dtype=int)
                self.maxids = len(self.pids_written)

                once_ids = np.where(pset.collection.getvardata('once_written', indices_to_write) == 0)[0]
                ids_once = ids[once_ids]
                indices_to_write_once = indices_to_write[once_ids]
                pset.collection.setvardata('once_written', indices_to_write_once, np.ones(len(ids_once)))

                if self.maxids > len(self.obs_written):
                    self.obs_written = np.append(self.obs_written, np.zeros((self.maxids-len(self.obs_written)), dtype=int))

                if self.create_new_zarrfile:
                    if self.chunks is None:
                        self.chunks = (len(ids), 1)
                    elif self.chunks[0] > len(ids):
                        logger.warning(f'Chunk size for trajectory ({self.chunks[0]}) is larger than length of initial set to write. '
                                       f'Reducing ParticleFile chunks to ({len(ids)}, {self.chunks[1]})')
                        self.chunks = (len(ids), self.chunks[1])
                    if (self.maxids > len(ids)) or (self.maxids > self.chunks[0]):
                        arrsize = (self.maxids, self.chunks[1])
                    else:
                        arrsize = self.chunks
                    ds = xr.Dataset(attrs=self.metadata, coords={"trajectory": ("trajectory", pids),
                                                                 "obs": ("obs", np.arange(arrsize[1], dtype=np.int32))})
                    attrs = self._create_variables_attribute_dict()
                    for var in self.vars_to_write:
                        varout = self._convert_varout_name(var)
                        if varout not in ['trajectory']:  # because 'trajectory' is written as coordinate
                            if self.write_once(var):
                                data = np.full((arrsize[0],), np.nan, dtype=self.vars_to_write[var])
                                data[ids_once] = pset.collection.getvardata(var, indices_to_write_once)
                                dims = ["trajectory"]
                            else:
                                data = np.full(arrsize, np.nan, dtype=self.vars_to_write[var])
                                data[ids, 0] = pset.collection.getvardata(var, indices_to_write)
                                dims = ["trajectory", "obs"]
                            ds[varout] = xr.DataArray(data=data, dims=dims, attrs=attrs[varout])
                            ds[varout].encoding['chunks'] = self.chunks[0] if self.write_once(var) else self.chunks
                    ds.to_zarr(self.fname, mode='w')
                    self.create_new_zarrfile = False
                else:
                    store = zarr.DirectoryStore(self.fname)
                    Z = zarr.group(store=store, overwrite=False)
                    obs = self.obs_written[np.array(ids)]
                    for var in self.vars_to_write:
                        varout = self._convert_varout_name(var)
                        if self.maxids > Z[varout].shape[0]:
                            self._extend_zarr_dims(Z[varout], store, dtype=self.vars_to_write[var], axis=0)
                        if self.write_once(var):
                            if len(ids_once) > 0:
                                Z[varout].vindex[ids_once] = pset.collection.getvardata(var, indices_to_write_once)
                        else:
                            if max(obs) >= Z[varout].shape[1]:
                                self._extend_zarr_dims(Z[varout], store, dtype=self.vars_to_write[var], axis=1)
                            Z[varout].vindex[ids, obs] = pset.collection.getvardata(var, indices_to_write)

                self.obs_written[np.array(ids)] += 1
