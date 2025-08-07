"""Module controlling the writing of ParticleSets to Zarr file."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
import zarr
from zarr.storage import DirectoryStore

import parcels
from parcels._constants import DATATYPES_TO_FILL_VALUES
from parcels.particle import _SAME_AS_FIELDSET_TIME_INTERVAL, ParticleClass
from parcels.tools._helpers import timedelta_to_float

if TYPE_CHECKING:
    from pathlib import Path

    from parcels.particle import Variable
    from parcels.particleset import ParticleSet

__all__ = ["ParticleFile"]


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
        self._outputdt = timedelta_to_float(outputdt)
        self._chunks = chunks
        self._maxids = 0
        self._pids_written = {}
        self.metadata = None
        self.create_new_zarrfile = create_new_zarrfile

        if isinstance(store, zarr.abc.store.Store):
            self.store = store
        else:
            self.store = _get_store_from_pathlike(store)

        if store.read_only:
            raise ValueError(f"Store {store} is read-only. Please provide a writable store.")

        # TODO v4: Add check that if create_new_zarrfile is False, the store already exists

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.fname!r}, "
            f"outputdt={self.outputdt!r}, "
            f"chunks={self.chunks!r}, "
            f"create_new_zarrfile={self.create_new_zarrfile!r})"
        )

    def set_metadata(self, parcels_mesh: Literal["spherical", "flat"]):
        self.metadata = {
            "feature_type": "trajectory",
            "Conventions": "CF-1.6/CF-1.7",
            "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
            "parcels_version": parcels.__version__,
            "parcels_mesh": parcels_mesh,
        }

    @property
    def outputdt(self):
        return self._outputdt

    @property
    def chunks(self):
        return self._chunks

    @property
    def fname(self):
        return self._fname

    def _convert_varout_name(self, var):
        if var == "depth":
            return "z"
        else:
            return var

    def _extend_zarr_dims(self, Z, store, dtype, axis):
        if axis == 1:
            a = np.full((Z.shape[0], self.chunks[1]), DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
            obs = zarr.group(store=store, overwrite=False)["obs"]
            if len(obs) == Z.shape[1]:
                obs.append(np.arange(self.chunks[1]) + obs[-1] + 1)
        else:
            extra_trajs = self._maxids - Z.shape[0]
            if len(Z.shape) == 2:
                a = np.full((extra_trajs, Z.shape[1]), DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
            else:
                a = np.full((extra_trajs,), DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
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
        vars_to_write = _get_vars_to_write(pclass)
        # if pset._data._ncount == 0:
        #     warnings.warn(
        #         f"ParticleSet is empty on writing as array at time {time:g}",
        #         RuntimeWarning,
        #         stacklevel=2,
        #     )
        #     return

        if indices is None:
            indices_to_write = _to_write_particles(pset._data, time)
        else:
            indices_to_write = indices

        if len(indices_to_write) == 0:
            return

        pids = pset._data["trajectory"][indices_to_write]
        to_add = sorted(set(pids) - set(self._pids_written.keys()))
        for i, pid in enumerate(to_add):
            self._pids_written[pid] = self._maxids + i
        ids = np.array([self._pids_written[p] for p in pids], dtype=int)
        self._maxids = len(self._pids_written)

        once_ids = np.where(pset._data["obs_written"][indices_to_write] == 0)[0]
        if len(once_ids) > 0:
            ids_once = ids[once_ids]
            indices_to_write_once = indices_to_write[once_ids]

        store = self.store
        if self.create_new_zarrfile:
            if self.chunks is None:
                self._chunks = (len(pset), 1)
            if (self._maxids > len(ids)) or (self._maxids > self.chunks[0]):  # type: ignore[index]
                arrsize = (self._maxids, self.chunks[1])  # type: ignore[index]
            else:
                arrsize = (len(ids), self.chunks[1])  # type: ignore[index]
            ds = xr.Dataset(
                attrs=self.metadata,
                coords={"trajectory": ("trajectory", pids), "obs": ("obs", np.arange(arrsize[1], dtype=np.int32))},
            )
            attrs = _create_variables_attribute_dict(pclass)
            obs = np.zeros((self._maxids), dtype=np.int32)
            for var in vars_to_write:
                varout = self._convert_varout_name(var)
                if varout not in ["trajectory"]:  # because 'trajectory' is written as coordinate
                    if var.to_write == "once":
                        data = np.full(
                            (arrsize[0],),
                            DATATYPES_TO_FILL_VALUES[vars_to_write.dtype],
                            dtype=var.dtype,
                        )
                        data[ids_once] = pset._data[var][indices_to_write_once]
                        dims = ["trajectory"]
                    else:
                        data = np.full(arrsize, DATATYPES_TO_FILL_VALUES[var.dtype], dtype=var.dtype)
                        data[ids, 0] = pset._data[var][indices_to_write]
                        dims = ["trajectory", "obs"]
                    ds[varout] = xr.DataArray(data=data, dims=dims, attrs=attrs[varout])
                    ds[varout].encoding["chunks"] = self.chunks[0] if var.to_write == "once" else self.chunks  # type: ignore[index]
            ds.to_zarr(store, mode="w")
            self.create_new_zarrfile = False
        else:
            Z = zarr.group(store=store, overwrite=False)
            obs = pset._data["obs_written"][indices_to_write]
            for var in vars_to_write:
                varout = self._convert_varout_name(var.name)
                if self._maxids > Z[varout].shape[0]:
                    self._extend_zarr_dims(Z[varout], store, dtype=var.dtype, axis=0)
                if var.to_write == "once":
                    if len(once_ids) > 0:
                        Z[varout].vindex[ids_once] = pset._data[var][indices_to_write_once]
                else:
                    if max(obs) >= Z[varout].shape[1]:  # type: ignore[type-var]
                        self._extend_zarr_dims(Z[varout], store, dtype=var.dtype, axis=1)
                    Z[varout].vindex[ids, obs] = pset._data[var][indices_to_write]

        pset._data["obs_written"][indices_to_write] = obs + 1

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
        for var in ["lon", "lat", "depth", "time"]:
            pset._data[f"{var}"] = pset._data[f"{var}_nextloop"]

        self.write(pset, time)


def _get_store_from_pathlike(path: Path | str) -> DirectoryStore:
    path = str(Path(path))  # Ensure valid path, and convert to string
    extension = os.path.splitext(path)[1]
    if extension != ".zarr":
        raise ValueError(f"ParticleFile name must end with '.zarr' extension. Got path {path!r}.")

    return DirectoryStore(path)


def _get_vars_to_write(particle: ParticleClass) -> list[Variable]:
    return [v for v in particle.variables if v.to_write is not False]


def _create_variables_attribute_dict(particle: ParticleClass) -> dict:
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
            fill_value = {"_FillValue": DATATYPES_TO_FILL_VALUES[var.dtype]}

        attrs[var.name] = {**var.attrs, **fill_value}

    attrs["time"]["units"] = "seconds since "  # TODO fix units
    attrs["time"]["calendar"] = "None"  # TODO fix calendar

    return attrs


def _to_write_particles(particle_data, time):
    """Return the Particles that need to be written at time: if particle.time is between time-dt/2 and time+dt (/2)"""
    return np.where(
        (
            np.less_equal(
                time - np.abs(particle_data["dt"] / 2), particle_data["time"], where=np.isfinite(particle_data["time"])
            )
            & np.greater_equal(
                time + np.abs(particle_data["dt"] / 2), particle_data["time"], where=np.isfinite(particle_data["time"])
            )
            | (
                (np.isnan(particle_data["dt"]))
                & np.equal(time, particle_data["time"], where=np.isfinite(particle_data["time"]))
            )
        )
        & (np.isfinite(particle_data["trajectory"]))
        & (np.isfinite(particle_data["time"]))
    )[0]
