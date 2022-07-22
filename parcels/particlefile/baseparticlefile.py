"""Module controlling the writing of ParticleSets to Zarr file"""
from abc import ABC
from abc import abstractmethod
import os
import numpy as np
import xarray as xr
import zarr

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
    """
    write_ondelete = None
    outputdt = None
    lasttime_written = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None

    def __init__(self, name, particleset, outputdt=np.infty, chunks=None, write_ondelete=False):

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
        self.vars_to_write = {}
        self.vars_to_write_once = {}
        for v in self.particleset.collection.ptype.variables:
            if v.to_write == 'once':
                self.vars_to_write_once[v.name] = v.dtype
            elif v.to_write is True:
                self.vars_to_write[v.name] = v.dtype
        if len(self.vars_to_write_once) > 0:
            self.written_once = []
        self.IDs_written = {}
        self.maxobs = {}
        self.written_first = False

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

        for vname in self.vars_to_write_once:
            attrs[vname] = {"_FillValue": self.fill_value_map[self.vars_to_write_once[vname]],
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

    def write(self, pset, time, deleted_only=False):
        """Write all data from one time step to the zarr file

        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        """
        data_dict, data_dict_once = pset.to_dict(self, time, deleted_only=deleted_only)

        if MPI:
            all_data_dict = MPI.COMM_WORLD.gather(data_dict, root=0)
            all_data_dict_once = MPI.COMM_WORLD.gather(data_dict_once, root=0)
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            all_data_dict = [data_dict]
            all_data_dict_once = [data_dict_once]
            rank = 0

        if rank == 0:

            maxtraj = len(self.IDs_written)
            for data_dict, data_dict_once in zip(all_data_dict, all_data_dict_once):
                if len(data_dict) > 0:
                    for i in data_dict['id']:
                        if i not in self.IDs_written:
                            self.IDs_written[i] = maxtraj
                            self.maxobs[i] = 0
                            maxtraj += 1
                        else:
                            self.maxobs[i] += 1

                if len(data_dict_once) > 0:
                    for i in data_dict_once['id']:
                        if i not in self.IDs_written:
                            self.IDs_written[i] = maxtraj
                            self.maxobs[i] = -1
                            maxtraj += 1

                if len(data_dict) > 0:
                    if not self.written_first:
                        if self.chunks is None:
                            self.chunks = (maxtraj, 1)
                        if self.chunks[0] < maxtraj:
                            raise RuntimeError(f"chunks[0] is smaller than the size of the initial particleset ({self.chunks[0]} < {maxtraj}). "
                                               "Please increase 'chunks' in your ParticleFile.")
                        ds = xr.Dataset(attrs=self.metadata)
                        attrs = self._create_variables_attribute_dict()
                        ids = [self.IDs_written[i] for i in data_dict['id']]
                        for var in data_dict:
                            varout = 'z' if var == 'depth' else var
                            varout = 'trajectory' if varout == 'id' else varout
                            data = np.full(self.chunks, np.nan, dtype=self.vars_to_write[var])
                            data[ids, 0] = data_dict[var]
                            ds[varout] = xr.DataArray(data=data, dims=["traj", "obs"], attrs=attrs[varout])
                            ds[varout].encoding['chunks'] = self.chunks
                        for var in data_dict_once:
                            if var != 'id':  # TODO check if needed
                                data = np.full((self.chunks[0],), np.nan, dtype=self.vars_to_write_once[var])
                                data[ids] = data_dict_once[var]
                                ds[var] = xr.DataArray(data=data, dims=["traj"], attrs=attrs[var])
                        ds.to_zarr(self.fname, mode='w')
                        self.written_first = True
                    else:
                        store = zarr.DirectoryStore(self.fname)
                        Z = zarr.group(store=store, overwrite=False)
                        ids = [self.IDs_written[i] for i in data_dict['id']]
                        maxobs = [self.maxobs[i] for i in data_dict['id']]

                        for var in data_dict:
                            varout = 'z' if var == 'depth' else var
                            varout = 'trajectory' if varout == 'id' else varout
                            if max(maxobs) >= Z[varout].shape[1]:
                                a = np.full((Z[varout].shape[0], self.chunks[1]), np.nan,
                                            dtype=self.vars_to_write[var])
                                Z[varout].append(a, axis=1)
                                zarr.consolidate_metadata(store)
                            if max(ids) >= Z[varout].shape[0]:
                                extra_trajs = max(maxtraj-Z[varout].shape[0], self.chunks[0])
                                a = np.full((extra_trajs, Z[varout].shape[1]), np.nan,
                                            dtype=self.vars_to_write[var])
                                Z[varout].append(a, axis=0)
                                zarr.consolidate_metadata(store)
                            Z[varout].vindex[ids, maxobs] = data_dict[var]
                        if len(data_dict_once) > 0:
                            ids = [self.IDs_written[i] for i in data_dict_once['id']]
                            for var in data_dict_once:
                                if var != 'id':  # TODO check if needed
                                    if max(ids) >= Z[var].shape[0]:
                                        a = np.full((maxtraj - Z[var].shape[0],), np.nan,
                                                    dtype=self.vars_to_write_once[var])
                                        Z[var].append(a, axis=0)
                                        zarr.consolidate_metadata(store)
                                    Z[var].vindex[ids] = data_dict_once[var]
