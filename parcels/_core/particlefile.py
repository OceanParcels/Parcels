"""Module controlling the writing of ParticleSets to Zarr file."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cftime
import numpy as np
import xarray as xr
import zarr
from zarr.storage import DirectoryStore

import parcels
from parcels._core.particle import _SAME_AS_FIELDSET_TIME_INTERVAL, ParticleClass
from parcels.utils._helpers import timedelta_to_float

if TYPE_CHECKING:
    from parcels._core.particle import Variable
    from parcels._core.particleset import ParticleSet
    from parcels._core.utils.time import TimeInterval

__all__ = ["ParticleFile"]

_DATATYPES_TO_FILL_VALUES = {
    np.dtype(np.float16): np.nan,
    np.dtype(np.float32): np.nan,
    np.dtype(np.float64): np.nan,
    np.dtype(np.bool_): np.iinfo(np.int8).max,
    np.dtype(np.int8): np.iinfo(np.int8).max,
    np.dtype(np.int16): np.iinfo(np.int16).max,
    np.dtype(np.int32): np.iinfo(np.int32).max,
    np.dtype(np.int64): np.iinfo(np.int64).max,
    np.dtype(np.uint8): np.iinfo(np.uint8).max,
    np.dtype(np.uint16): np.iinfo(np.uint16).max,
    np.dtype(np.uint32): np.iinfo(np.uint32).max,
    np.dtype(np.uint64): np.iinfo(np.uint64).max,
}


class ParticleFile:
    """Initialise trajectory output.

    Parameters
    ----------
    name : str
        Basename of the output file. This can also be a Zarr store object.
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a timedelta object or a positive double.
    chunks :
        Tuple (trajs, obs) to control the size of chunks in the zarr output.
    create_new_zarrfile : bool
        Whether to create a new file. Default is True

    Returns
    -------
    ParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    def __init__(self, store, outputdt, chunks=None, create_new_zarrfile=True):
        if isinstance(outputdt, timedelta):
            outputdt = np.timedelta64(int(outputdt.total_seconds()), "s")

        if not isinstance(outputdt, np.timedelta64):
            raise ValueError(f"Expected outputdt to be a np.timedelta64 or datetime.timedelta, got {type(outputdt)}")

        self._outputdt = outputdt

        _assert_valid_chunks_tuple(chunks)
        self._chunks = chunks
        self._maxids = 0
        self._pids_written = {}
        self.metadata = {}
        self._create_new_zarrfile = create_new_zarrfile

        if not isinstance(store, zarr.storage.Store):
            store = _get_store_from_pathlike(store)

        self._store = store

        # TODO v4: Enable once updating to zarr v3
        # if store.read_only:
        #     raise ValueError(f"Store {store} is read-only. Please provide a writable store.")

        # TODO v4: Add check that if create_new_zarrfile is False, the store already exists

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"outputdt={self.outputdt!r}, "
            f"chunks={self.chunks!r}, "
            f"create_new_zarrfile={self.create_new_zarrfile!r})"
        )

    def set_metadata(self, parcels_grid_mesh: Literal["spherical", "flat"]):
        self.metadata.update(
            {
                "feature_type": "trajectory",
                "Conventions": "CF-1.6/CF-1.7",
                "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
                "parcels_version": parcels.__version__,
                "parcels_grid_mesh": parcels_grid_mesh,
            }
        )

    @property
    def outputdt(self):
        return self._outputdt

    @property
    def chunks(self):
        return self._chunks

    @property
    def store(self):
        return self._store

    @property
    def create_new_zarrfile(self):
        return self._create_new_zarrfile

    def _convert_varout_name(self, var):
        if var == "depth":
            return "z"
        else:
            return var

    def _extend_zarr_dims(self, Z, store, dtype, axis):
        if axis == 1:
            a = np.full((Z.shape[0], self.chunks[1]), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
            obs = zarr.group(store=store, overwrite=False)["obs"]
            if len(obs) == Z.shape[1]:
                obs.append(np.arange(self.chunks[1]) + obs[-1] + 1)
        else:
            extra_trajs = self._maxids - Z.shape[0]
            if len(Z.shape) == 2:
                a = np.full((extra_trajs, Z.shape[1]), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
            else:
                a = np.full((extra_trajs,), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
        Z.append(a, axis=axis)
        zarr.consolidate_metadata(store)

    def write(self, pset: ParticleSet, time, indices=None):
        """Write all data from one time step to the zarr file,
        before the particle locations are updated.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet (same time object as fieldset)
        """
        pclass = pset._ptype
        time_interval = pset.fieldset.time_interval
        particle_data = pset._data
        time = timedelta_to_float(time - time_interval.left)
        particle_data = _convert_particle_data_time_to_float_seconds(particle_data, time_interval)

        self._write_particle_data(
            particle_data=particle_data, pclass=pclass, time_interval=time_interval, time=time, indices=indices
        )

    def _write_particle_data(self, *, particle_data, pclass, time_interval, time, indices=None):
        # if pset._data._ncount == 0:
        #     warnings.warn(
        #         f"ParticleSet is empty on writing as array at time {time:g}",
        #         RuntimeWarning,
        #         stacklevel=2,
        #     )
        #     return
        nparticles = len(particle_data["trajectory"])
        vars_to_write = _get_vars_to_write(pclass)
        if indices is None:
            indices_to_write = _to_write_particles(particle_data, time)
        else:
            indices_to_write = indices

        if len(indices_to_write) == 0:
            return

        pids = particle_data["trajectory"][indices_to_write]
        to_add = sorted(set(pids) - set(self._pids_written.keys()))
        for i, pid in enumerate(to_add):
            self._pids_written[pid] = self._maxids + i
        ids = np.array([self._pids_written[p] for p in pids], dtype=int)
        self._maxids = len(self._pids_written)

        once_ids = np.where(particle_data["obs_written"][indices_to_write] == 0)[0]
        if len(once_ids) > 0:
            ids_once = ids[once_ids]
            indices_to_write_once = indices_to_write[once_ids]

        store = self.store
        if self.create_new_zarrfile:
            if self.chunks is None:
                self._chunks = (nparticles, 1)
            if (self._maxids > len(ids)) or (self._maxids > self.chunks[0]):  # type: ignore[index]
                arrsize = (self._maxids, self.chunks[1])  # type: ignore[index]
            else:
                arrsize = (len(ids), self.chunks[1])  # type: ignore[index]
            ds = xr.Dataset(
                attrs=self.metadata,
                coords={"trajectory": ("trajectory", pids), "obs": ("obs", np.arange(arrsize[1], dtype=np.int32))},
            )
            attrs = _create_variables_attribute_dict(pclass, time_interval)
            obs = np.zeros((self._maxids), dtype=np.int32)
            for var in vars_to_write:
                dtype = _maybe_convert_time_dtype(var.dtype)
                varout = self._convert_varout_name(var.name)
                if varout not in ["trajectory"]:  # because 'trajectory' is written as coordinate
                    if var.to_write == "once":
                        data = np.full(
                            (arrsize[0],),
                            _DATATYPES_TO_FILL_VALUES[dtype],
                            dtype=dtype,
                        )
                        data[ids_once] = particle_data[var.name][indices_to_write_once]
                        dims = ["trajectory"]
                    else:
                        data = np.full(arrsize, _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
                        data[ids, 0] = particle_data[var.name][indices_to_write]
                        dims = ["trajectory", "obs"]
                    ds[varout] = xr.DataArray(data=data, dims=dims, attrs=attrs[var.name])
                    ds[varout].encoding["chunks"] = self.chunks[0] if var.to_write == "once" else self.chunks  # type: ignore[index]
            ds.to_zarr(store, mode="w")
            self._create_new_zarrfile = False
        else:
            Z = zarr.group(store=store, overwrite=False)
            obs = particle_data["obs_written"][indices_to_write]
            for var in vars_to_write:
                dtype = _maybe_convert_time_dtype(var.dtype)
                varout = self._convert_varout_name(var.name)
                if self._maxids > Z[varout].shape[0]:
                    self._extend_zarr_dims(Z[varout], store, dtype=dtype, axis=0)
                if var.to_write == "once":
                    if len(once_ids) > 0:
                        Z[varout].vindex[ids_once] = particle_data[var.name][indices_to_write_once]
                else:
                    if max(obs) >= Z[varout].shape[1]:  # type: ignore[type-var]
                        self._extend_zarr_dims(Z[varout], store, dtype=dtype, axis=1)
                    Z[varout].vindex[ids, obs] = particle_data[var.name][indices_to_write]

        particle_data["obs_written"][indices_to_write] = obs + 1

    def write_latest_locations(self, pset, time):
        """Write the current (latest) particle locations to zarr file.
        This can be useful at the end of a pset.execute(), when the last locations are not written yet.
        Note that this only updates the locations, not any of the other Variables. Therefore, use with care.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet. Note that typically this would be pset.time_nextloop
        """
        for var in ["lon", "lat", "depth"]:
            pset._data[f"{var}"] += pset._data[f"d{var}"]
        pset._data["time"] = pset._data["time_nextloop"]
        self.write(pset, time)


def _get_store_from_pathlike(path: Path | str) -> DirectoryStore:
    path = str(Path(path))  # Ensure valid path, and convert to string
    extension = os.path.splitext(path)[1]
    if extension != ".zarr":
        raise ValueError(f"ParticleFile name must end with '.zarr' extension. Got path {path!r}.")

    return DirectoryStore(path)


def _get_vars_to_write(particle: ParticleClass) -> list[Variable]:
    return [v for v in particle.variables if v.to_write is not False]


def _create_variables_attribute_dict(particle: ParticleClass, time_interval: TimeInterval) -> dict:
    """Creates the dictionary with variable attributes.

    Notes
    -----
    For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
    """
    attrs = {}

    vars = [var for var in particle.variables if var.to_write is not False]
    for var in vars:
        fill_value = {}
        if var.dtype is not _SAME_AS_FIELDSET_TIME_INTERVAL.VALUE:
            fill_value = {"_FillValue": _DATATYPES_TO_FILL_VALUES[var.dtype]}

        attrs[var.name] = {**var.attrs, **fill_value}

    attrs["time"].update(_get_calendar_and_units(time_interval))

    return attrs


def _to_write_particles(particle_data, time):
    """Return the Particles that need to be written at time: if particle.time is between time-dt/2 and time+dt (/2)"""
    return np.where(
        (
            np.less_equal(
                time - np.abs(particle_data["dt"] / 2),
                particle_data["time_nextloop"],
                where=np.isfinite(particle_data["time_nextloop"]),
            )
            & np.greater_equal(
                time + np.abs(particle_data["dt"] / 2),
                particle_data["time_nextloop"],
                where=np.isfinite(particle_data["time_nextloop"]),
            )  # check time - dt/2 <= particle_data["time"] <= time + dt/2
            | (
                (np.isnan(particle_data["dt"]))
                & np.equal(time, particle_data["time_nextloop"], where=np.isfinite(particle_data["time_nextloop"]))
            )  # or dt is NaN and time matches particle_data["time"]
        )
        & (np.isfinite(particle_data["trajectory"]))
        & (np.isfinite(particle_data["time_nextloop"]))
    )[0]


def _convert_particle_data_time_to_float_seconds(particle_data, time_interval):
    #! Important that this is a shallow copy, so that updates to this propogate back to the original data
    particle_data = particle_data.copy()

    particle_data["time"] = ((particle_data["time"] - time_interval.left) / np.timedelta64(1, "s")).astype(np.float64)
    particle_data["time_nextloop"] = (
        (particle_data["time_nextloop"] - time_interval.left) / np.timedelta64(1, "s")
    ).astype(np.float64)
    particle_data["dt"] = (particle_data["dt"] / np.timedelta64(1, "s")).astype(np.float64)
    return particle_data


def _maybe_convert_time_dtype(dtype: np.dtype | _SAME_AS_FIELDSET_TIME_INTERVAL) -> np.dtype:
    """Convert the dtype of time to float64 if it is not already."""
    if dtype is _SAME_AS_FIELDSET_TIME_INTERVAL.VALUE:
        return np.dtype(
            np.uint64
        )  #! We need to have here some proper mechanism for converting particle data to the data that is to be output to zarr (namely the time needs to be converted to float seconds by subtracting the time_interval.left)
    return dtype


def _get_calendar_and_units(time_interval: TimeInterval) -> dict[str, str]:
    calendar = None
    units = "seconds"
    if isinstance(time_interval.left, (np.datetime64, datetime)):
        calendar = "standard"
    elif isinstance(time_interval.left, cftime.datetime):
        calendar = time_interval.left.calendar

    if calendar is not None:
        units += f" since {time_interval.left}"

    attrs = {"units": units}
    if calendar is not None:
        attrs["calendar"] = calendar

    return attrs


def _assert_valid_chunks_tuple(chunks):
    e = ValueError(f"chunks must be a tuple of integers with length 2, got {chunks=!r} instead.")
    if chunks is None:
        return

    if not isinstance(chunks, tuple):
        raise e
    if len(chunks) != 2:
        raise e
    if not all(isinstance(c, int) for c in chunks):
        raise e
