"""Internal helpers for Parcels."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from datetime import timedelta

import numpy as np

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


def timedelta_to_float(dt: float | timedelta | np.timedelta64) -> float:
    """Convert a timedelta to a float in seconds."""
    if isinstance(dt, timedelta):
        return dt.total_seconds()
    if isinstance(dt, np.timedelta64):
        return float(dt / np.timedelta64(1, "s"))
    return float(dt)


def should_calculate_next_ti(ti: int, tau: float, tdim: int):
    """Check if the time is beyond the last time in the field"""
    return np.greater(tau, 0) and ti < tdim - 1


def _assert_same_function_signature(f: Callable, *, ref: Callable, context: str) -> None:
    """Ensures a function `f` has the same signature as the reference function `ref`."""
    sig_ref = inspect.signature(ref)
    sig = inspect.signature(f)

    if len(sig_ref.parameters) != len(sig.parameters):
        raise ValueError(
            f"{context} function must have {len(sig_ref.parameters)} parameters, got {len(sig.parameters)}"
        )

    for (_name1, param1), (_name2, param2) in zip(sig_ref.parameters.items(), sig.parameters.items(), strict=False):
        if param1.kind != param2.kind:
            raise ValueError(
                f"Parameter '{_name2}' has incorrect parameter kind. Expected {param1.kind}, got {param2.kind}"
            )
        if param1.name != param2.name:
            raise ValueError(f"Parameter '{_name2}' has incorrect name. Expected '{param1.name}', got '{param2.name}'")
