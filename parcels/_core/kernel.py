from __future__ import annotations

import math  # noqa: F401
import random  # noqa: F401
import types
import warnings
from typing import TYPE_CHECKING

import numpy as np

from parcels._core.basegrid import GridType
from parcels._core.statuscodes import (
    StatusCode,
    _raise_field_interpolation_error,
    _raise_field_out_of_bound_error,
    _raise_field_out_of_bound_surface_error,
    _raise_general_error,
    _raise_grid_searching_error,
    _raise_time_extrapolation_error,
)
from parcels._core.warnings import KernelWarning
from parcels.kernels import (
    AdvectionAnalytical,
    AdvectionRK4,
    AdvectionRK45,
)
from parcels.utils._helpers import _assert_same_function_signature

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Kernel"]


ErrorsToThrow = {
    StatusCode.ErrorTimeExtrapolation: _raise_time_extrapolation_error,
    StatusCode.ErrorOutOfBounds: _raise_field_out_of_bound_error,
    StatusCode.ErrorThroughSurface: _raise_field_out_of_bound_surface_error,
    StatusCode.ErrorInterpolation: _raise_field_interpolation_error,
    StatusCode.ErrorGridSearching: _raise_grid_searching_error,
    StatusCode.Error: _raise_general_error,
}


class Kernel:
    """Kernel object that encapsulates auto-generated code.

    Parameters
    ----------
    fieldset : parcels.Fieldset
        FieldSet object providing the field information (possibly None)
    ptype :
        PType object for the kernel particle
    pyfunc :
        (aggregated) Kernel function

    Notes
    -----
    A Kernel is either created from a <function ...> object
    or an ast.FunctionDef object.
    """

    def __init__(
        self,
        fieldset,
        ptype,
        pyfuncs: list[types.FunctionType],
    ):
        for f in pyfuncs:
            if not isinstance(f, types.FunctionType):
                raise TypeError(f"Argument pyfunc should be a function or list of functions. Got {type(f)}")
            _assert_same_function_signature(f, ref=AdvectionRK4, context="Kernel")

        if len(pyfuncs) == 0:
            raise ValueError("List of `pyfuncs` should have at least one function.")

        self._fieldset = fieldset
        self._ptype = ptype

        self._positionupdate_kernels_added = False

        for f in pyfuncs:
            self.check_fieldsets_in_kernels(f)

        # # TODO will be implemented when we support CROCO again
        # if (pyfunc is AdvectionRK4_3D) and fieldset.U.gridindexingtype == "croco":
        #     pyfunc = AdvectionRK4_3D_CROCO

        self._pyfuncs: list[Callable] = pyfuncs

    @property  #! Ported from v3. To be removed in v4? (/find another way to name kernels in output file)
    def funcname(self):
        ret = ""
        for f in self._pyfuncs:
            ret += f.__name__
        return ret

    @property
    def ptype(self):
        return self._ptype

    @property
    def fieldset(self):
        return self._fieldset

    def remove_deleted(self, pset):
        """Utility to remove all particles that signalled deletion."""
        bool_indices = pset._data["state"] == StatusCode.Delete
        indices = np.where(bool_indices)[0]
        # TODO v4: need to implement ParticleFile writing of deleted particles
        # if len(indices) > 0 and self.fieldset.particlefile is not None:
        #     self.fieldset.particlefile.write(pset, None, indices=indices)
        if len(indices) > 0:
            pset.remove_indices(indices)

    def add_positionupdate_kernels(self):
        # Adding kernels that set and update the coordinate changes
        def Setcoords(particles, fieldset):  # pragma: no cover
            import numpy as np  # noqa

            particles.lon += particles.dlon
            particles.lat += particles.dlat
            particles.depth += particles.ddepth

            particles.dlon = 0
            particles.dlat = 0
            particles.ddepth = 0

            particles.time = particles.time_nextloop

        def UpdateTime(particles, fieldset):  # pragma: no cover
            particles.time_nextloop = particles.time + particles.dt

        self._pyfuncs = (Setcoords + self + UpdateTime)._pyfuncs

    def check_fieldsets_in_kernels(self, pyfunc):  # TODO v4: this can go into another method? assert_is_compatible()?
        """
        Checks the integrity of the fieldset with the kernels.

        This function is to be called from the derived class when setting up the 'pyfunc'.
        """
        if self.fieldset is not None:
            if pyfunc is AdvectionAnalytical:
                if self._fieldset.U.interp_method != "cgrid_velocity":
                    raise NotImplementedError("Analytical Advection only works with C-grids")
                if self._fieldset.U.grid._gtype not in [GridType.CurvilinearZGrid, GridType.RectilinearZGrid]:
                    raise NotImplementedError("Analytical Advection only works with Z-grids in the vertical")
            elif pyfunc is AdvectionRK45:
                if not hasattr(self.fieldset, "RK45_tol"):
                    warnings.warn(
                        "Setting RK45 tolerance to 10 m. Use fieldset.add_constant('RK45_tol', [distance]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_constant("RK45_tol", 10)
                if self.fieldset.U.grid._mesh == "spherical":
                    self.fieldset.RK45_tol /= (
                        1852 * 60
                    )  # TODO does not account for zonal variation in meter -> degree conversion
                if not hasattr(self.fieldset, "RK45_min_dt"):
                    warnings.warn(
                        "Setting RK45 minimum timestep to 1 s. Use fieldset.add_constant('RK45_min_dt', [timestep]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_constant("RK45_min_dt", 1)
                if not hasattr(self.fieldset, "RK45_max_dt"):
                    warnings.warn(
                        "Setting RK45 maximum timestep to 1 day. Use fieldset.add_constant('RK45_max_dt', [timestep]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_constant("RK45_max_dt", 60 * 60 * 24)

    def merge(self, kernel):
        if not isinstance(kernel, type(self)):
            raise TypeError(f"Cannot merge {type(kernel)} with {type(self)}. Both should be of type {type(self)}.")

        assert self.fieldset == kernel.fieldset, "Cannot merge kernels with different fieldsets"
        assert self.ptype == kernel.ptype, "Cannot merge kernels with different particle types"

        return type(self)(
            self.fieldset,
            self.ptype,
            pyfuncs=self._pyfuncs + kernel._pyfuncs,
        )

    def __add__(self, kernel):
        if isinstance(kernel, types.FunctionType):
            kernel = type(self)(self.fieldset, self.ptype, pyfuncs=[kernel])
        return self.merge(kernel)

    def __radd__(self, kernel):
        if isinstance(kernel, types.FunctionType):
            kernel = type(self)(self.fieldset, self.ptype, pyfuncs=[kernel])
        return kernel.merge(self)

    @classmethod
    def from_list(cls, fieldset, ptype, pyfunc_list):
        """Create a combined kernel from a list of functions.

        Takes a list of functions, converts them to kernels, and joins them
        together.

        Parameters
        ----------
        fieldset : parcels.Fieldset
            FieldSet object providing the field information (possibly None)
        ptype :
            PType object for the kernel particle
        pyfunc_list : list of functions
            List of functions to be combined into a single kernel.
        *args :
            Additional arguments passed to first kernel during construction.
        **kwargs :
            Additional keyword arguments passed to first kernel during construction.
        """
        if not isinstance(pyfunc_list, list):
            raise TypeError(f"Argument `pyfunc_list` should be a list of functions. Got {type(pyfunc_list)}")
        if not all([isinstance(f, types.FunctionType) for f in pyfunc_list]):
            raise ValueError("Argument `pyfunc_list` should be a list of functions.")

        return cls(fieldset, ptype, pyfunc_list)

    def execute(self, pset, endtime, dt):
        """Execute this Kernel over a ParticleSet for several timesteps.

        Parameters
        ----------
        pset :
            object of (sub-)type ParticleSet
        endtime :
            endtime of this overall kernel evaluation step
        dt :
            computational integration timestep from pset.execute
        """
        compute_time_direction = 1 if dt > 0 else -1

        pset._data["state"][:] = StatusCode.Evaluate

        if not self._positionupdate_kernels_added:
            self.add_positionupdate_kernels()
            self._positionupdate_kernels_added = True

        while (len(pset) > 0) and np.any(np.isin(pset.state, [StatusCode.Evaluate, StatusCode.Repeat])):
            time_to_endtime = compute_time_direction * (endtime - pset.time_nextloop)

            if all(time_to_endtime <= 0):
                return StatusCode.Success

            # adapt dt to end exactly on endtime
            if compute_time_direction == 1:
                pset.dt = np.maximum(np.minimum(pset.dt, time_to_endtime), 0)
            else:
                pset.dt = np.minimum(np.maximum(pset.dt, -time_to_endtime), 0)

            # run kernels for all particles that need to be evaluated
            evaluate_particles = (pset.state == StatusCode.Evaluate) & (pset.dt != 0)
            for f in self._pyfuncs:
                f(pset[evaluate_particles], self._fieldset)

                # check for particles that have to be repeated
                repeat_particles = pset.state == StatusCode.Repeat
                while np.any(repeat_particles):
                    f(pset[repeat_particles], self._fieldset)
                    repeat_particles = pset.state == StatusCode.Repeat

            # revert to original dt (unless in RK45 mode)
            if not hasattr(self.fieldset, "RK45_tol"):
                pset._data["dt"][:] = dt

            # Reset particle state for particles that signalled success and have not reached endtime yet
            particles_to_evaluate = (pset.state == StatusCode.Success) & (time_to_endtime > 0)
            pset[particles_to_evaluate].state = StatusCode.Evaluate

            # delete particles that signalled deletion
            self.remove_deleted(pset)

            # check and throw errors
            if np.any(pset.state == StatusCode.StopAllExecution):
                return StatusCode.StopAllExecution

            for error_code, error_func in ErrorsToThrow.items():
                if np.any(pset.state == error_code):
                    inds = pset.state == error_code
                    if error_code == StatusCode.ErrorTimeExtrapolation:
                        error_func(pset[inds].time)
                    else:
                        error_func(pset[inds].depth, pset[inds].lat, pset[inds].lon)

        return pset
