"""
Typing support for Parcels.

This module contains type aliases used throughout Parcels as well as functions that are
used for runtime parameter validation (to ensure users are only using the right params).

"""

import ast
import datetime
import os
from collections.abc import Callable
from typing import Any, Literal, get_args


class ParcelsAST(ast.AST):
    ccode: str


InterpMethodOption = Literal[
    "linear",
    "nearest",
    "freeslip",
    "partialslip",
    "bgrid_velocity",
    "bgrid_w_velocity",
    "cgrid_velocity",
    "linear_invdist_land_tracer",
    "bgrid_tracer",
    "cgrid_tracer",
]  # corresponds with `tracer_interp_method`
InterpMethod = (
    InterpMethodOption | dict[str, InterpMethodOption]
)  # corresponds with `interp_method` (which can also be dict mapping field names to method)
PathLike = str | os.PathLike
Mesh = Literal["spherical", "flat"]  # corresponds with `mesh`
VectorType = Literal["3D", "3DSigma", "2D"] | None  # corresponds with `vector_type`
ChunkMode = Literal["auto", "specific", "failsafe"]  # corresponds with `chunk_mode`
GridIndexingType = Literal["pop", "mom5", "mitgcm", "nemo", "croco"]  # corresponds with `gridindexingtype`
UpdateStatus = Literal["not_updated", "first_updated", "updated"]  # corresponds with `_update_status`
TimePeriodic = float | datetime.timedelta | Literal[False]  # corresponds with `time_periodic`
NetcdfEngine = Literal["netcdf4", "xarray", "scipy"]


KernelFunction = Callable[..., None]


def _validate_against_pure_literal(value, typing_literal):
    """Uses a Literal type alias to validate.

    Can't be used with ``Literal[...] | None`` etc. as its not a pure literal.
    """
    if value not in get_args(typing_literal):
        msg = f"Invalid value {value!r}. Valid options are {get_args(typing_literal)!r}"
        raise ValueError(msg)


# Assertion functions to clean user input
def assert_valid_interp_method(value: Any):
    _validate_against_pure_literal(value, InterpMethodOption)


def assert_valid_mesh(value: Any):
    _validate_against_pure_literal(value, Mesh)


def assert_valid_gridindexingtype(value: Any):
    _validate_against_pure_literal(value, GridIndexingType)
