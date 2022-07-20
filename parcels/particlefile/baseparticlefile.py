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
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param convert_at_end: Boolean to convert npy files to netcdf at end of run. Default is True
    """
    write_ondelete = None
    convert_at_end = None
    outputdt = None
    lasttime_written = None
    name = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, convert_at_end=True):

        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

        self.name = name
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

        extension = os.path.splitext(str(self.name))[1]
        self.fname = self.name if extension in ['.nc', '.nc4', '.zarr'] else "%s.zarr" % self.name
        if extension == '':
            extension = '.zarr'
        self.outputformat = extension

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

        maxtraj = len(self.IDs_written)
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
                ds = xr.Dataset(attrs=self.metadata)
                attrs = self._create_variables_attribute_dict()
                ids = [self.IDs_written[i] for i in data_dict['id']]
                for var in data_dict:
                    varout = 'z' if var == 'depth' else var
                    varout = 'trajectory' if varout == 'id' else varout
                    data = np.full((maxtraj, 1), np.nan, dtype=self.vars_to_write[var])
                    data[ids, 0] = data_dict[var]
                    ds[varout] = xr.DataArray(data=data, dims=["traj", "obs"], attrs=attrs[varout])
                for var in data_dict_once:
                    if var != 'id':  # TODO check if needed
                        data = np.full((maxtraj,), np.nan, dtype=self.vars_to_write_once[var])
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
                    for i, t, v in zip(ids, maxobs, data_dict[var]):
                        if t >= Z[varout].shape[1]:
                            a = np.full((Z[varout].shape[0], 1), np.nan, dtype=self.vars_to_write[var])
                            Z[varout].append(a, axis=1)
                            zarr.consolidate_metadata(store)
                        if i >= Z[varout].shape[0]:
                            a = np.full((maxtraj-Z[varout].shape[0], Z[varout].shape[1]), np.nan, dtype=self.vars_to_write[var])
                            Z[varout].append(a, axis=0)
                            zarr.consolidate_metadata(store)
                        Z[varout][i, t] = v
                if len(data_dict_once) > 0:
                    ids = [self.IDs_written[i] for i in data_dict_once['id']]
                    for var in data_dict_once:
                        if var != 'id':  # TODO check if needed
                            for i, v in zip(ids, data_dict_once[var]):
                                if i >= Z[var].shape[0]:
                                    a = np.full((maxtraj - Z[var].shape[0],), np.nan,
                                                dtype=self.vars_to_write_once[var])
                                    Z[var].append(a, axis=0)
                                    zarr.consolidate_metadata(store)
                                Z[var][i] = v

        # if expanded_trajs and self.written_first:
        #     for z in Z:
        #         zin = 'id' if z == 'trajectory' else z
        #         zin = 'depth' if zin == 'z' else zin
        #
        #         if Z[z].ndim == 2:
        #             a = np.full((expanded_trajs, Z[z].shape[1]), np.nan, dtype=self.vars_to_write[zin])
        #         else:
        #             a = np.full((expanded_trajs,), np.nan, dtype=self.vars_to_write_once[zin])
        #         Z[z].append(a)
        #
        #
        # if len(data_dict) > 0:
        #     ids = [self.IDs_written[i] for i in data_dict['id']]
        #     for var in data_dict:
        #         varout = 'z' if var == 'depth' else var
        #         varout = 'trajectory' if varout == 'id' else varout
        #         data = np.full((datalen, 1), np.nan, dtype=self.vars_to_write[var])
        #         data[ids, 0] = data_dict[var]
        #         ds[varout] = xr.DataArray(data=data, dims=["traj", "obs"], attrs=attrs[varout])
        #         if self.written_first and "_FillValue" in ds[varout].attrs:
        #             del ds[varout].attrs["_FillValue"]
        #
        # if len(data_dict_once) > 0:
        #     ids = [self.IDs_written[i] for i in data_dict_once['id']]
        #     if self.written_first:
        #         ds_in = xr.open_zarr(self.name)
        #     for var in data_dict_once:
        #         if var != 'id':
        #             if self.written_first:
        #                 data = ds_in[var].values
        #                 print(data, ids)
        #             else:
        #                 data = np.full((datalen,), np.nan, dtype=self.vars_to_write_once[var])
        #             data[ids] = data_dict_once[var]
        #             ds[var] = xr.DataArray(data=data, dims=["traj"], attrs=attrs[var])
        #             if self.written_first and "_FillValue" in ds[var].attrs:
        #                 del ds[var].attrs["_FillValue"]
        #
        # if len(ds) > 0:
        #     if not self.written_first:
        #         ds.to_zarr(self.fname, mode='w')
        #         self.written_first = True
        #     else:
        #         ds.to_zarr(self.fname, mode='a', append_dim='obs')
