import sys
import warnings
from collections.abc import Iterable
from typing import Literal

import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from tqdm import tqdm

from parcels._core.utils.time import TimeInterval
from parcels._reprs import particleset_repr
from parcels.application_kernels.advection import AdvectionRK4
from parcels.basegrid import GridType
from parcels.kernel import Kernel
from parcels.particle import KernelParticle, Particle, create_particle_data
from parcels.particlefile import ParticleFile
from parcels.tools.converters import convert_to_flat_array
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import StatusCode
from parcels.tools.warnings import ParticleSetWarning

__all__ = ["ParticleSet"]


class ParticleSet:
    """Class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets, meaning that the particle set only
    holds the particles defined on construction. Individual particles can neither be added nor deleted individually,
    and individual particles can only be deleted as a set procedurally (i.e. by 'particle.delete()'-call during
    kernel execution).

    Parameters
    ----------
    fieldset :
        mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
    pclass : parcels.particle.Particle
        Optional object that inherits from :mod:`parcels.particle.Particle` object that defines custom particle
    lon :
        List of initial longitude values for particles
    lat :
        List of initial latitude values for particles
    depth :
        Optional list of initial depth values for particles. Default is 0m
    time :
        Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    repeatdt : datetime.timedelta or float, optional
        Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
    lonlatdepth_dtype :
        Floating precision for lon, lat, depth particle coordinates.
        It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
        and np.float64 if the interpolation method is 'cgrid_velocity'
    trajectory_ids :
        Optional list of "trajectory" values (integers) for the particle IDs
    partition_function :
        Function to use for partitioning particles over processors. Default is to use kMeans
    periodic_domain_zonal :
        Zonal domain size, used to apply zonally periodic boundaries for particle-particle
        interaction. If None, no zonally periodic boundaries are applied

        Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(
        self,
        fieldset,
        pclass=Particle,
        lon=None,
        lat=None,
        depth=None,
        time=None,
        trajectory_ids=None,
        **kwargs,
    ):
        self._data = None
        self._repeat_starttime = None
        self._kernel = None
        self._interaction_kernel = None

        self.fieldset = fieldset
        lon = np.empty(shape=0) if lon is None else convert_to_flat_array(lon)
        lat = np.empty(shape=0) if lat is None else convert_to_flat_array(lat)
        time = np.empty(shape=0) if time is None else convert_to_flat_array(time)

        if trajectory_ids is None:
            trajectory_ids = np.arange(lon.size)

        if depth is None:
            mindepth = 0
            for field in self.fieldset.fields.values():
                if field.grid.depth is not None:
                    mindepth = min(mindepth, field.grid.depth[0])
            depth = np.ones(lon.size) * mindepth
        else:
            depth = convert_to_flat_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, "lon, lat, depth don't all have the same lenghts"

        if time is None or len(time) == 0:
            # do not set a time yet (because sign_dt not known)
            if fieldset.time_interval is None:
                time = np.timedelta64("NaT", "ns")
            else:
                time = type(fieldset.time_interval.left)("NaT", "ns")
        elif type(time[0]) in [np.datetime64, np.timedelta64]:
            pass  # already in the right format
        else:
            raise TypeError("particle time must be a datetime, timedelta, or date object")
        time = np.repeat(time, lon.size) if time.size == 1 else time

        assert lon.size == time.size, "time and positions (lon, lat, depth) do not have the same lengths."

        if fieldset.time_interval:
            _warn_particle_times_outside_fieldset_time_bounds(time, fieldset.time_interval)

        for kwvar in kwargs:
            if kwvar not in ["partition_function"]:
                kwargs[kwvar] = convert_to_flat_array(kwargs[kwvar])
                assert lon.size == kwargs[kwvar].size, (
                    f"{kwvar} and positions (lon, lat, depth) don't have the same lengths."
                )

        self._data = create_particle_data(
            pclass=pclass,
            nparticles=lon.size,
            ngrids=len(fieldset.gridset),
            time_interval=fieldset.time_interval,
            initial=dict(
                lon=lon,
                lat=lat,
                depth=depth,
                time=time,
                lon_nextloop=lon,
                lat_nextloop=lat,
                depth_nextloop=depth,
                time_nextloop=time,
                trajectory=trajectory_ids,
            ),
        )
        self._ptype = pclass

        # update initial values provided on ParticleSet creation # TODO: Wrap this into create_particle_data
        particle_variables = [v.name for v in pclass.variables]
        for kwvar, kwval in kwargs.items():
            if kwvar not in particle_variables:
                raise RuntimeError(f"Particle class does not have Variable {kwvar}")
            self._data[kwvar][:] = kwval

        self._kernel = None

    def __del__(self):
        if self._data is not None and isinstance(self._data, xr.Dataset):
            del self._data
        self._data = None

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self):
            p = self.__getitem__(self._index)
            self._index += 1
            return p
        raise StopIteration

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        Parameters
        ----------
        name : str
            Name of the property
        """
        return self._data[name]

    def __getitem__(self, index):
        """Get a single particle by index."""
        return KernelParticle(self._data, index=index)

    def __setattr__(self, name, value):
        if name in ["_data"]:
            object.__setattr__(self, name, value)
        elif isinstance(self._data, dict) and name in self._data.keys():
            self._data[name][:] = value
        else:
            object.__setattr__(self, name, value)

    @staticmethod
    def lonlatdepth_dtype_from_field_interp_method(field):
        # TODO update this when now interp methods are implemented
        if field.interp_method == "cgrid_velocity":
            return np.float64
        return np.float32

    @property
    def size(self):
        return len(self)

    def __repr__(self):
        return particleset_repr(self)

    def __len__(self):
        return len(self._data["trajectory"])

    def add(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet

        """
        assert particles is not None, (
            f"Trying to add another {type(self)} to this one, but the other one is None - invalid operation."
        )
        assert type(particles) is type(self)

        if len(particles) == 0:
            return

        if len(self) == 0:
            self._data = particles._data
            return

        if isinstance(particles, type(self)):
            if len(self._data["trajectory"]) > 0:
                offset = self._data["trajectory"].max() + 1
            else:
                offset = 0
            particles._data["trajectory"] = particles._data["trajectory"] + offset

        for d in self._data:
            self._data[d] = np.concatenate((self._data[d], particles._data[d]))

        # Adding particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        return self

    def __iadd__(self, particles):
        """Add particles to the ParticleSet.

        Note that this is an incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet
        """
        self.add(particles)
        return self

    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`."""
        for d in self._data:
            self._data[d] = np.delete(self._data[d], indices, axis=0)

    def _active_particles_mask(self, time, dt):
        active_indices = (time - self._data["time"]) / dt >= 0
        non_err_indices = np.isin(self._data["state"], [StatusCode.Success, StatusCode.Evaluate])
        active_indices = np.logical_and(active_indices, non_err_indices)
        self._active_particle_idx = np.where(active_indices)[0]
        return active_indices

    def _compute_neighbor_tree(self, time, dt):
        active_mask = self._active_particles_mask(time, dt)

        self._values = np.vstack(
            (
                self._data["depth"],
                self._data["lat"],
                self._data["lon"],
            )
        )
        if self._dirty_neighbor:
            self._neighbor_tree.rebuild(self._values, active_mask=active_mask)
            self._dirty_neighbor = False
        else:
            self._neighbor_tree.update_values(self._values, new_active_mask=active_mask)

    def _neighbors_by_index(self, particle_idx):
        neighbor_idx, distances = self._neighbor_tree.find_neighbors_by_idx(particle_idx)
        neighbor_idx = self._active_particle_idx[neighbor_idx]
        mask = neighbor_idx != particle_idx
        neighbor_idx = neighbor_idx[mask]
        if "horiz_dist" in self._data._ptype.variables:
            self._data["vert_dist"][neighbor_idx] = distances[0, mask]
            self._data["horiz_dist"][neighbor_idx] = distances[1, mask]
        return True  # TODO fix for v4 ParticleDataIterator(self.particledata, subset=neighbor_idx)

    def _neighbors_by_coor(self, coor):
        neighbor_idx = self._neighbor_tree.find_neighbors_by_coor(coor)
        neighbor_ids = self._data["trajectory"][neighbor_idx]
        return neighbor_ids

    # TODO: This method is only tested in tutorial notebook. Add unit test?
    def populate_indices(self):
        """Pre-populate guesses of particle ei (element id) indices using a kdtree.

        This is only intended for curvilinear grids, where the initial index search
        may be quite expensive.
        """
        for i, grid in enumerate(self.fieldset.gridset):
            if grid._gtype not in [GridType.CurvilinearZGrid, GridType.CurvilinearSGrid]:
                continue

            tree_data = np.stack((grid.lon.flat, grid.lat.flat), axis=-1)
            IN = np.all(~np.isnan(tree_data), axis=1)
            tree = KDTree(tree_data[IN, :])
            # stack all the particle positions for a single query
            pts = np.stack((self._data["lon"], self._data["lat"]), axis=-1)
            # query datatype needs to match tree datatype
            _, idx_nan = tree.query(pts.astype(tree_data.dtype))

            idx = np.where(IN)[0][idx_nan]

            self._data["ei"][:, i] = idx  # assumes that we are in the surface layer (zi=0)

    @classmethod
    def from_particlefile(cls, fieldset, pclass, filename, restart=True, restarttime=None, **kwargs):
        """Initialise the ParticleSet from a zarr ParticleFile.
        This creates a new ParticleSet based on locations of all particles written
        in a zarr ParticleFile at a certain time. Particle IDs are preserved if restart=True

        Parameters
        ----------
        fieldset : parcels.fieldset.FieldSet
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass :
            Particle class. May be a parcels.particle.Particle class as defined in parcels, or a subclass defining a custom particle.
        filename : str
            Name of the particlefile from which to read initial conditions
        restart : bool
            BSignal if pset is used for a restart (default is True).
            In that case, Particle IDs are preserved.
        restarttime :
            time at which the Particles will be restarted. Default is the last time written.
            Alternatively, restarttime could be a time value (including np.datetime64) or
            a callable function such as np.nanmin. The last is useful when running with dt < 0.
        repeatdt : datetime.timedelta or float, optional
            Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
        **kwargs :
            Keyword arguments passed to the particleset constructor.
        """
        raise NotImplementedError(
            "ParticleSet.from_particlefile is not yet implemented in v4."
        )  # TODO implement this when ParticleFile is implemented in v4

    def Kernel(self, pyfunc):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object.

        Conversion is based on `fieldset` and `ptype` of the ParticleSet.

        Parameters
        ----------
        pyfunc : function or list of functions
            Python function to convert into kernel. If a list of functions is provided,
            the functions will be converted to kernels and combined into a single kernel.
        """
        if isinstance(pyfunc, list):
            return Kernel.from_list(
                self.fieldset,
                self._ptype,
                pyfunc,
            )
        return Kernel(
            self.fieldset,
            self._ptype,
            pyfuncs=[pyfunc],
        )

    def InteractionKernel(self, pyfunc_inter):
        from parcels.interaction.interactionkernel import InteractionKernel

        if pyfunc_inter is None:
            return None
        return InteractionKernel(self.fieldset, self._ptype, pyfunc=pyfunc_inter)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile` object from the ParticleSet."""
        return ParticleFile(*args, particleset=self, **kwargs)

    def data_indices(self, variable_name, compare_values, invert=False):
        """Get the indices of all particles where the value of `variable_name` equals (one of) `compare_values`.

        Parameters
        ----------
        variable_name : str
            Name of the variable to check.
        compare_values :
            Value or list of values to compare to.
        invert :
            Whether to invert the selection. I.e., when True,
            return all indices that do not equal (one of)
            `compare_values`. (Default value = False)

        Returns
        -------
        np.ndarray
            Numpy array of indices that satisfy the test.

        """
        compare_values = (
            np.array([compare_values]) if type(compare_values) not in [list, dict, np.ndarray] else compare_values
        )
        return np.where(np.isin(self._data[variable_name], compare_values, invert=invert))[
            0
        ]  # TODO check if this can be faster with xarray indexing?

    @property
    def _error_particles(self):
        """Get indices of all particles that are in an error state.

        Returns
        -------
        indices
            Indices of error particles.
        """
        return self.data_indices("state", [StatusCode.Success, StatusCode.Evaluate], invert=True)

    @property
    def _num_error_particles(self):
        """Get the number of particles that are in an error state.

        Returns
        -------
        int
            Number of error particles.
        """
        return np.sum(np.isin(self._data["state"], [StatusCode.Success, StatusCode.Evaluate], invert=True))

    def set_variable_write_status(self, var, write_status):
        """Method to set the write status of a Variable.

        Parameters
        ----------
        var :
            Name of the variable (string)
        write_status :
            Write status of the variable (True, False or 'once')
        """
        self._data[var].set_variable_write_status(write_status)

    def execute(
        self,
        pyfunc=AdvectionRK4,
        endtime: np.timedelta64 | np.datetime64 | None = None,
        runtime: np.timedelta64 | None = None,
        dt: np.timedelta64 | None = None,
        output_file=None,
        verbose_progress=True,
    ):
        """Execute a given kernel function over the particle set for multiple timesteps.

        Optionally also provide sub-timestepping
        for particle output.

        Parameters
        ----------
        pyfunc :
            Kernel function to execute. This can be the name of a
            defined Python function or a :class:`parcels.kernel.Kernel` object.
            Kernels can be concatenated using the + operator (Default value = AdvectionRK4)
        endtime (np.datetime64 or np.timedelta64): :
            End time for the timestepping loop. If a np.timedelta64 is provided, it is interpreted as the total simulation time. In this case,
            the absolute end time is the start of the fieldset's time interval plus the np.timedelta64.
            If a datetime is provided, it is interpreted as the absolute end time of the simulation.
        runtime (np.timedelta64):
            The duration of the simuulation execution. Must be a np.timedelta64 object and is required to be set when the `fieldset.time_interval` is not defined.
            If the `fieldset.time_interval` is defined and the runtime is provided, the end time will be the start of the fieldset's time interval plus the runtime.
        dt (np.timedelta64):
            Timestep interval (as a np.timedelta64 object) to be passed to the kernel.
            Use a negative value for a backward-in-time simulation. (Default value = 1 second)
        output_file :
            mod:`parcels.particlefile.ParticleFile` object for particle output (Default value = None)
        verbose_progress : bool
            Boolean for providing a progress bar for the kernel execution loop. (Default value = True)

        Notes
        -----
        ``ParticleSet.execute()`` acts as the main entrypoint for simulations, and provides the simulation time-loop. This method encapsulates the logic controlling the switching between kernel execution, output file writing, reading in fields for new timesteps, adding new particles to the simulation domain, stopping the simulation, and executing custom functions (``postIterationCallbacks`` provided by the user).
        """
        # check if particleset is empty. If so, return immediately
        if len(self) == 0:
            return

        if not isinstance(pyfunc, Kernel):
            pyfunc = self.Kernel(pyfunc)

        self._kernel = pyfunc

        if output_file:
            output_file.metadata["parcels_kernels"] = self._kernel.name

        if dt is None:
            dt = np.timedelta64(1, "s")

        if not isinstance(dt, np.timedelta64) or np.isnat(dt) or (sign_dt := np.sign(dt).astype(int)) not in [-1, 1]:
            raise ValueError(f"dt must be a positive or negative np.timedelta64 object, got {dt=!r}")

        self._data["dt"][:] = dt

        start_time, end_time = _get_simulation_start_and_end_times(
            self.fieldset.time_interval, self._data["time_nextloop"], runtime, endtime, sign_dt
        )

        # Set the time of the particles if it hadn't been set on initialisation
        if np.isnat(self._data["time"]).any():
            self._data["time"][:] = start_time
            self._data["time_nextloop"][:] = start_time

        outputdt = output_file.outputdt if output_file else None

        # Set up pbar
        if output_file:
            logger.info(f"Output files are stored in {output_file.fname}.")

        if verbose_progress:
            pbar = tqdm(total=(end_time - start_time) / np.timedelta64(1, "s"), file=sys.stdout)

        next_output = outputdt if output_file else None

        time = start_time
        while sign_dt * (time - end_time) < 0:
            if sign_dt > 0:
                next_time = end_time  # TODO update to min(next_output, end_time) when ParticleFile works
            else:
                next_time = end_time  # TODO update to max(next_output, end_time) when ParticleFile works
            self._kernel.execute(self, endtime=next_time, dt=dt)

            # TODO: Handle IO timing based of timedelta or datetime objects
            if next_output:
                if abs(next_time - next_output) < 1e-12:
                    if output_file:
                        output_file.write(self, next_output)
                    if np.isfinite(outputdt):
                        next_output += outputdt

            if verbose_progress:
                pbar.update((next_time - time) / np.timedelta64(1, "s"))

            time = next_time

        if verbose_progress:
            pbar.close()


def _warn_outputdt_release_desync(outputdt: float, starttime: float, release_times: Iterable[float]):
    """Gives the user a warning if the release time isn't a multiple of outputdt."""
    if any((np.isfinite(t) and (t - starttime) % outputdt != 0) for t in release_times):
        warnings.warn(
            "Some of the particles have a start time difference that is not a multiple of outputdt. "
            "This could cause the first output of some of the particles that start later "
            "in the simulation to be at a different time than expected.",
            ParticleSetWarning,
            stacklevel=2,
        )


def _warn_particle_times_outside_fieldset_time_bounds(release_times: np.ndarray, time: TimeInterval):
    if np.isnat(release_times).all():
        return

    if isinstance(time.left, np.datetime64) and isinstance(release_times[0], np.timedelta64):
        release_times = np.array([t + time.left for t in release_times])
    if np.any(release_times < time.left) or np.any(release_times > time.right):
        warnings.warn(
            "Some particles are set to be released outside the FieldSet's executable time domain.",
            ParticleSetWarning,
            stacklevel=2,
        )


def _get_simulation_start_and_end_times(
    time_interval: TimeInterval,
    particle_release_times: np.ndarray,
    runtime: np.timedelta64 | None,
    endtime: np.datetime64 | None,
    sign_dt: Literal[-1, 1],
) -> tuple[np.datetime64, np.datetime64]:
    if runtime is not None and endtime is not None:
        raise ValueError(
            f"runtime and endtime are mutually exclusive - provide one or the other. Got {runtime=!r}, {endtime=!r}"
        )

    if runtime is None and time_interval is None:
        raise ValueError("The runtime must be provided when the time_interval is not defined for a fieldset.")

    if sign_dt == 1:
        first_release_time = particle_release_times.min()
    else:
        first_release_time = particle_release_times.max()

    start_time = _get_start_time(first_release_time, time_interval, sign_dt, runtime)

    if endtime is None:
        if not isinstance(runtime, np.timedelta64):
            raise ValueError(f"The runtime must be a np.timedelta64 object. Got {type(runtime)}")

        endtime = start_time + sign_dt * runtime

    if time_interval is not None:
        if type(endtime) != type(time_interval.left):  # noqa: E721
            raise ValueError(
                f"The endtime must be of the same type as the fieldset.time_interval start time. Got {endtime=!r} with {time_interval=!r}"
            )
        if endtime not in time_interval:
            msg = (
                f"Calculated/provided end time of {endtime!r} is not in fieldset time interval {time_interval!r}. Either reduce your runtime, modify your "
                "provided endtime, or change your release timing."
                "Important info:\n"
                f"    First particle release: {first_release_time!r}\n"
                f"    runtime: {runtime!r}\n"
                f"    (calculated) endtime: {endtime!r}"
            )
            raise ValueError(msg)

    return start_time, endtime


def _get_start_time(first_release_time, time_interval, sign_dt, runtime):
    if time_interval is None:
        time_interval = TimeInterval(left=np.timedelta64(0, "s"), right=runtime)

    if sign_dt == 1:
        fieldset_start = time_interval.left
    else:
        fieldset_start = time_interval.right

    start_time = first_release_time if not np.isnat(first_release_time) else fieldset_start
    return start_time
