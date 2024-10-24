"""Internal helpers for Parcels."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from textwrap import dedent
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parcels import Field, ParticleSet
PACKAGE = "Parcels"


def deprecated(msg: str = "") -> Callable:
    """Decorator marking a function as being deprecated

    Parameters
    ----------
    msg : str, optional
        Custom message to append to the deprecation warning.

    Examples
    --------
    ```
    @deprecated("Please use `another_function` instead")
    def some_old_function(x, y):
        return x + y

    @deprecated()
    def some_other_old_function(x, y):
        return x + y
    ```
    """
    if msg:
        msg = " " + msg

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg_formatted = (
                f"`{func.__qualname__}` is deprecated and will be removed in a future release of {PACKAGE}.{msg}"
            )

            warnings.warn(msg_formatted, category=DeprecationWarning, stacklevel=3)
            return func(*args, **kwargs)

        patch_docstring(wrapper, f"\n\n.. deprecated:: {msg}")
        return wrapper

    return decorator


def deprecated_made_private(func: Callable) -> Callable:
    return deprecated(
        "It has moved to the internal API as it is not expected to be directly used by "
        "the end-user. If you feel that you use this code directly in your scripts, please "
        "comment on our tracking issue at https://github.com/OceanParcels/Parcels/issues/1695.",
    )(func)


def patch_docstring(obj: Callable, extra: str) -> None:
    obj.__doc__ = f"{obj.__doc__ or ''}{extra}".strip()


def pretty_field(field: Field) -> str:
    """Return a pretty repr for Field"""
    out = f"""<{type(field).__name__}>
                grid            : {field.grid!r                    }
                extrapolate time: {field.allow_time_extrapolation!r}
                time_periodic   : {field.time_periodic!r           }
                gridindexingtype: {field.gridindexingtype!r        }
                to_write        : {field.to_write!r                }
            """
    return dedent(out)


def pretty_particleset(pset: ParticleSet) -> str:
    """Return a pretty repr for ParticleSet"""
    if len(pset) < 10:
        lst = [repr(p) for p in pset]
    else:
        lst = [repr(p) for p in pset[:7]] + ["..."]

    out = f"""<{type(pset).__name__}>
                fieldset:    {pset.fieldset}
                pclass:      {pset.pclass}
                repeatdt:    {pset.repeatdt}
                # particles: {len(pset)}
                particles:   {lst}"""
    return dedent(out)


def default_repr(obj: Any):
    return object.__repr__(obj)
