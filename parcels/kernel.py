import abc
import ast
import functools
import inspect
import math  # noqa: F401
import random  # noqa: F401
import textwrap
import types
import warnings

import numpy as np

from parcels.application_kernels.advection import (
    AdvectionAnalytical,
    AdvectionRK4_3D,
    AdvectionRK4_3D_CROCO,
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

__all__ = ["BaseKernel", "Kernel"]


class BaseKernel(abc.ABC):  # noqa # TODO v4: check if we need this BaseKernel class (gave a "B024 `BaseKernel` is an abstract base class, but it has no abstract methods or properties" error)
    """Superclass for 'normal' and Interactive Kernels"""

    def __init__(
        self,
        fieldset,
        ptype,
        pyfunc=None,
        funcname=None,
        funccode=None,
        py_ast=None,
        funcvars=None,
    ):
        self._fieldset = fieldset
        self.field_args = None
        self.const_args = None
        self._ptype = ptype

        # Derive meta information from pyfunc, if not given
        self._pyfunc = None
        self.funcname = funcname or pyfunc.__name__
        self.name = f"{ptype.name}{self.funcname}"
        self.funcvars = funcvars
        self.funccode = funccode
        self.py_ast = py_ast  # TODO v4: check if this is needed
        self._positionupdate_kernels_added = False

    @property
    def ptype(self):
        return self._ptype

    @property
    def pyfunc(self):
        return self._pyfunc

    @property
    def fieldset(self):
        return self._fieldset

    def remove_deleted(self, pset):
        """Utility to remove all particles that signalled deletion."""
        bool_indices = pset._data["state"] == StatusCode.Delete
        indices = np.where(bool_indices)[0]
        if len(indices) > 0 and self.fieldset.particlefile is not None:
            self.fieldset.particlefile.write(pset, None, indices=indices)
        pset.remove_indices(indices)


class Kernel(BaseKernel):
    """Kernel object that encapsulates auto-generated code.

    Parameters
    ----------
    fieldset : parcels.Fieldset
        FieldSet object providing the field information (possibly None)
    ptype :
        PType object for the kernel particle
    pyfunc :
        (aggregated) Kernel function
    funcname : str
        function name

    Notes
    -----
    A Kernel is either created from a <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(
        self,
        fieldset,
        ptype,
        pyfunc=None,
        funcname=None,
        funccode=None,
        py_ast=None,
        funcvars=None,
    ):
        super().__init__(
            fieldset=fieldset,
            ptype=ptype,
            pyfunc=pyfunc,
            funcname=funcname,
            funccode=funccode,
            py_ast=py_ast,
            funcvars=funcvars,
        )

        # Derive meta information from pyfunc, if not given
        self.check_fieldsets_in_kernels(pyfunc)

        if (pyfunc is AdvectionRK4_3D) and fieldset.U.gridindexingtype == "croco":
            pyfunc = AdvectionRK4_3D_CROCO
            self.funcname = "AdvectionRK4_3D_CROCO"

        if funcvars is not None:  # TODO v4: check if needed from here onwards
            self.funcvars = funcvars
        elif hasattr(pyfunc, "__code__"):
            self.funcvars = list(pyfunc.__code__.co_varnames)
        else:
            self.funcvars = None
        self.funccode = funccode or inspect.getsource(pyfunc.__code__)
        self.funccode = (  # Remove parcels. prefix (see #1608)
            self.funccode.replace("parcels.StatusCode", "StatusCode")
        )

        # Parse AST if it is not provided explicitly
        self.py_ast = (
            py_ast or ast.parse(textwrap.dedent(self.funccode)).body[0]
        )  # Dedent allows for in-lined kernel definitions
        if pyfunc is None:
            # Extract user context by inspecting the call stack
            stack = inspect.stack()
            try:
                user_ctx = stack[-1][0].f_globals
                user_ctx["math"] = globals()["math"]
                user_ctx["random"] = globals()["random"]
                user_ctx["StatusCode"] = globals()["StatusCode"]
            except:
                warnings.warn(
                    "Could not access user context when merging kernels",
                    KernelWarning,
                    stacklevel=2,
                )
                user_ctx = globals()
            finally:
                del stack  # Remove cyclic references
            # Generate Python function from AST
            py_mod = ast.parse("")
            py_mod.body = [self.py_ast]
            exec(compile(py_mod, "<ast>", "exec"), user_ctx)
            self._pyfunc = user_ctx[self.funcname]
        else:
            self._pyfunc = pyfunc

        self.name = f"{ptype.name}{self.funcname}"

    @property
    def ptype(self):
        return self._ptype

    @property
    def pyfunc(self):
        return self._pyfunc

    @property
    def fieldset(self):
        return self._fieldset

    def add_positionupdate_kernels(self):
        # Adding kernels that set and update the coordinate changes
        def Setcoords(particle, fieldset, time):  # pragma: no cover
            import numpy as np  # noqa

            particle_dlon = 0  # noqa
            particle_dlat = 0  # noqa
            particle_ddepth = 0  # noqa
            particle.lon = particle.lon_nextloop
            particle.lat = particle.lat_nextloop
            particle.depth = particle.depth_nextloop
            particle.time = particle.time_nextloop

        def Updatecoords(particle, fieldset, time):  # pragma: no cover
            particle.lon_nextloop = particle.lon + particle_dlon  # type: ignore[name-defined] # noqa
            particle.lat_nextloop = particle.lat + particle_dlat  # type: ignore[name-defined] # noqa
            particle.depth_nextloop = particle.depth + particle_ddepth  # type: ignore[name-defined] # noqa
            particle.time_nextloop = particle.time + particle.dt

        self._pyfunc = (Setcoords + self + Updatecoords)._pyfunc

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

    def merge(self, kernel, kclass):
        funcname = self.funcname + kernel.funcname
        func_ast = None
        if self.py_ast is not None:
            func_ast = ast.FunctionDef(
                name=funcname,
                args=self.py_ast.args,
                body=self.py_ast.body + kernel.py_ast.body,
                decorator_list=[],
                lineno=1,
                col_offset=0,
            )
        return kclass(
            self.fieldset,
            self.ptype,
            pyfunc=None,
            funcname=funcname,
            funccode=self.funccode + kernel.funccode,
            py_ast=func_ast,
            funcvars=self.funcvars + kernel.funcvars,
        )

    def __add__(self, kernel):
        if not isinstance(kernel, type(self)):
            kernel = type(self)(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, type(self))

    def __radd__(self, kernel):
        if not isinstance(kernel, type(self)):
            kernel = type(self)(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, type(self))

    @classmethod
    def from_list(cls, fieldset, ptype, pyfunc_list, *args, **kwargs):
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
            raise TypeError(f"Argument function_list should be a list of functions. Got {type(pyfunc_list)}")
        if len(pyfunc_list) == 0:
            raise ValueError("Argument function_list should have at least one function.")
        if not all([isinstance(f, types.FunctionType) for f in pyfunc_list]):
            raise ValueError("Argument function_lst should be a list of functions.")

        pyfunc_list = pyfunc_list.copy()
        pyfunc_list[0] = cls(fieldset, ptype, pyfunc_list[0], *args, **kwargs)
        return functools.reduce(lambda x, y: x + y, pyfunc_list)

    def execute(self, pset, endtime, dt):
        """Execute this Kernel over a ParticleSet for several timesteps."""
        pset._data["state"][:] = StatusCode.Evaluate

        if abs(dt) < np.timedelta64(1, "ns"):  # TODO still needed?
            warnings.warn(
                "'dt' is too small, causing numerical accuracy limit problems. Please chose a higher 'dt' and rather scale the 'time' axis of the field accordingly. (related issue #762)",
                RuntimeWarning,
                stacklevel=2,
            )

        if not self._positionupdate_kernels_added:
            self.add_positionupdate_kernels()
            self._positionupdate_kernels_added = True

        for p in pset:
            self.evaluate_particle(p, endtime)
            if p.state == StatusCode.StopAllExecution:
                return StatusCode.StopAllExecution

        # Remove all particles that signalled deletion
        self.remove_deleted(pset)

        # Identify particles that threw errors
        n_error = pset._num_error_particles

        while n_error > 0:
            error_pset = pset._error_particles
            # Check for StatusCodes
            for p in error_pset:
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
                        f"Deleting particle {p.id} because of non-recoverable error",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    p.delete()

            # Remove all particles that signalled deletion
            self.remove_deleted(pset)  # Generalizable version!

            # Re-execute Kernels to retry particles with StatusCode.Repeat
            for p in pset:
                self.evaluate_particle(p, endtime)

            n_error = pset._num_error_particles

    def evaluate_particle(self, p, endtime):
        """Execute the kernel evaluation of for an individual particle.

        Parameters
        ----------
        p :
            object of (sub-)type Particle
        endtime :
            endtime of this overall kernel evaluation step
        dt :
            computational integration timestep
        """
        while p.state in [StatusCode.Evaluate, StatusCode.Repeat]:
            pre_dt = p.dt

            sign_dt = np.sign(p.dt).astype(int)
            if sign_dt * (p.time_nextloop - endtime) > np.timedelta64(0, "ns"):
                return p

            # TODO implement below later again
            # try:  # Use next_dt from AdvectionRK45 if it is set
            #     if abs(endtime - p.time_nextloop) < abs(p.next_dt) - 1e-6:
            #         p.next_dt = abs(endtime - p.time_nextloop) * sign_dt
            # except AttributeError:
            # if abs(endtime - p.time_nextloop) < abs(p.dt) - 1e-6:
            #     p.dt = abs(endtime - p.time_nextloop) * sign_dt
            res = self._pyfunc(p, self._fieldset, p.time_nextloop)

            if res is None:
                if p.state == StatusCode.Success:
                    if sign_dt * (p.time - endtime) > np.timedelta64(0, "ns"):
                        p.state = StatusCode.Evaluate
            else:
                p.state = res

            p.dt = pre_dt
        return p
