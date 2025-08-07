from __future__ import annotations

import math  # noqa: F401
import random  # noqa: F401
import types
import warnings
from typing import TYPE_CHECKING

import numpy as np

from parcels.application_kernels.advection import (
    AdvectionAnalytical,
    AdvectionRK45,
)
from parcels.basegrid import GridType
from parcels.tools.statuscodes import (
    StatusCode,
    TimeExtrapolationError,
    _raise_field_out_of_bound_error,
    _raise_field_out_of_bound_surface_error,
    _raise_field_sampling_error,
)
from parcels.tools.warnings import KernelWarning

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Kernel"]


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

    @property  #! Ported from v3. To be removed in v4? (/find another way to name kernels in output file)
    def name(self):
        return f"{self._ptype.name}{self.funcname}"

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
        def Setcoords(particle, fieldset, time):  # pragma: no cover
            import numpy as np  # noqa

            particle.dlon = 0
            particle.dlat = 0
            particle.ddepth = 0
            particle.lon = particle.lon_nextloop
            particle.lat = particle.lat_nextloop
            particle.depth = particle.depth_nextloop
            particle.time = particle.time_nextloop

        def Updatecoords(particle, fieldset, time):  # pragma: no cover
            particle.lon_nextloop = particle.lon + particle.dlon
            particle.lat_nextloop = particle.lat + particle.dlat
            particle.depth_nextloop = particle.depth + particle.ddepth
            particle.time_nextloop = particle.time + particle.dt

        self._pyfuncs = (Setcoords + self + Updatecoords)._pyfuncs

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
                if self.fieldset.U.grid.mesh == "spherical":
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
        """Execute this Kernel over a ParticleSet for several timesteps."""
        pset._data["state"][:] = StatusCode.Evaluate

        if abs(dt) < np.timedelta64(1000, "ns"):  # TODO still needed?
            warnings.warn(
                "'dt' is too small, causing numerical accuracy limit problems. Please chose a higher 'dt' and rather scale the 'time' axis of the field accordingly. (related issue #762)",
                RuntimeWarning,
                stacklevel=2,
            )

        if not self._positionupdate_kernels_added:
            self.add_positionupdate_kernels()
            self._positionupdate_kernels_added = True

        self.evaluate_pset(pset, endtime)
        if any(pset.state == StatusCode.StopAllExecution):
            return StatusCode.StopAllExecution

        # Remove all particles that signalled deletion
        self.remove_deleted(pset)

        # Identify particles that threw errors
        n_error = pset._num_error_particles

        while n_error > 0:
            for i in pset._error_particles:
                p = pset[i]
                if p.state == StatusCode.StopExecution:
                    return
                if p.state == StatusCode.StopAllExecution:
                    return StatusCode.StopAllExecution
                if p.state == StatusCode.Repeat:
                    p.state = StatusCode.Evaluate
                elif p.state == StatusCode.ErrorTimeExtrapolation:
                    raise TimeExtrapolationError(p.time)
                elif p.state == StatusCode.ErrorOutOfBounds:
                    _raise_field_out_of_bound_error(p.depth, p.lat, p.lon)
                elif p.state == StatusCode.ErrorThroughSurface:
                    _raise_field_out_of_bound_surface_error(p.depth, p.lat, p.lon)
                elif p.state == StatusCode.Error:
                    _raise_field_sampling_error(p.depth, p.lat, p.lon)
                elif p.state == StatusCode.Delete:
                    pass
                else:
                    warnings.warn(
                        f"Deleting particle {p.trajectory} because of non-recoverable error",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    p.delete()

            # Remove all particles that signalled deletion
            self.remove_deleted(pset)  # Generalizable version!

            # Re-execute Kernels to retry particles with StatusCode.Repeat
            self.evaluate_pset(pset, endtime)

            n_error = pset._num_error_particles

    def evaluate_pset(self, pset, endtime):
        """Execute the kernel evaluation of for the entire particle set.

        Parameters
        ----------
        pset :
            object of (sub-)type ParticleSet
        endtime :
            endtime of this overall kernel evaluation step
        dt :
            computational integration timestep
        """
        sign_dt = np.where(pset.dt >= 0, 1, -1)
        while pset[0].state in [StatusCode.Evaluate, StatusCode.Repeat]:
            if all(sign_dt * (endtime - pset.time_nextloop) <= 0):
                return pset

            pre_dt = pset.dt
            try:  # Use next_dt from AdvectionRK45 if it is set
                pset.next_dt = np.where(
                    sign_dt * (endtime - pset.time_nextloop) <= pset.next_dt,
                    np.where(sign_dt * (endtime - pset.time_nextloop) < 0, 0, sign_dt * (endtime - pset.time_nextloop)),
                    pset.next_dt,
                )
            except KeyError:
                pset.dt = np.where(
                    sign_dt * (endtime - pset.time_nextloop) <= pset.dt,
                    np.where(sign_dt * (endtime - pset.time_nextloop) < 0, 0, sign_dt * (endtime - pset.time_nextloop)),
                    pset.dt,
                )
            res = None
            for f in self._pyfuncs:
                # TODO remove "time" from kernel signature in v4; because it doesn't make sense for vectorized particles
                res_tmp = f(pset, self._fieldset, pset.time_nextloop[0])
                if res_tmp is not None:  # TODO v4: Remove once all kernels return StatusCode
                    res = res_tmp
                if res in [StatusCode.StopExecution, StatusCode.Repeat]:
                    break

            if res is None:
                pset.state = np.where(
                    (pset.state == StatusCode.Success) & (sign_dt * (pset.time - endtime) > 0),
                    StatusCode.Evaluate,
                    pset.state,
                )
            else:  # TODO need to think how the kernel exitcode works on vectorized particleset
                pset.state = res

            pset.dt = pre_dt
        return pset
