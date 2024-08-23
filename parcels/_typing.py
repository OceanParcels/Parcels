"""
Typing support for Parcels.

This module contains type aliases used throughout Parcels as well as functions that are
used for runtime parameter validation (to ensure users are only using the right params).

"""

import ast
import datetime
import os
from typing import Any, Callable, Literal, get_args


class ParcelsAST(ast.AST):
    ccode: str


# InterpMethod = InterpMethodOption | dict[str, InterpMethodOption] # (can also be a dict, search for `if type(interp_method) is dict`)
# InterpMethodOption = Literal[
#     "nearest",
#     "freeslip",
#     "partialslip",
#     "bgrid_velocity",
#     "bgrid_w_velocity",
#     "cgrid_velocity",
#     "linear_invdist_land_tracer",
#     "nearest",
#     "cgrid_tracer",
# ] # mostly corresponds with `interp_method` # TODO: This should be narrowed. Unlikely applies to every context
PathLike = str | os.PathLike
Mesh = Literal["spherical", "flat"]  # mostly corresponds with `mesh`
VectorType = Literal["3D", "2D"] | None  # mostly corresponds with `vector_type`
ChunkMode = Literal["auto", "specific", "failsafe"]  # mostly corresponds with `chunk_mode`
GridIndexingType = Literal["pop", "mom5", "mitgcm", "nemo"]  # mostly corresponds with `grid_indexing_type`
UpdateStatus = Literal["not_updated", "first_updated", "updated"]  # mostly corresponds with `update_status`
TimePeriodic = float | datetime.timedelta | Literal[False]  # mostly corresponds with `update_status`

KernelFunction = Callable[..., None]


def ensure_is_literal_value(value: Any, literal: Any) -> None:
    """Ensures that a value is a valid option for the provided Literal type annotation."""
    valid_options = get_args(literal)
    if value not in valid_options:
        raise ValueError(f"{value!r} is not a valid option. Valid options are {valid_options}")
