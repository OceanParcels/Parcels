"""
Typing support for Parcels.

This module contains type aliases used throughout Parcels as well as functions that are
used for runtime parameter validation (to ensure users are only using the right params).

"""

import ast
import datetime
import os
from collections.abc import Callable
from typing import Literal


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
    "nearest",
    "bgrid_tracer",
    "cgrid_tracer",
]  # corresponds with `tracer_interp_method`
InterpMethod = (
    InterpMethodOption | dict[str, InterpMethodOption]
)  # corresponds with `interp_method` (which can also be dict mapping field names to method)
PathLike = str | os.PathLike
Mesh = Literal["spherical", "flat"]  # corresponds with `mesh`
VectorType = Literal["3D", "2D"] | None  # corresponds with `vector_type`
ChunkMode = Literal["auto", "specific", "failsafe"]  # corresponds with `chunk_mode`
GridIndexingType = Literal["pop", "mom5", "mitgcm", "nemo"]  # corresponds with `grid_indexing_type`
UpdateStatus = Literal["not_updated", "first_updated", "updated"]  # corresponds with `update_status`
TimePeriodic = float | datetime.timedelta | Literal[False]  # corresponds with `update_status`
NetcdfEngine = Literal["netcdf4", "xarray", "scipy"]


KernelFunction = Callable[..., None]
