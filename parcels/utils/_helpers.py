"""Internal helpers for Parcels."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import timedelta

import numpy as np

PACKAGE = "Parcels"


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

    for param1, param2 in zip(sig_ref.parameters.values(), sig.parameters.values(), strict=False):
        if param1.kind != param2.kind:
            raise ValueError(
                f"Parameter '{param2.name}' has incorrect parameter kind. Expected {param1.kind}, got {param2.kind}"
            )
        if param1.name != param2.name:
            raise ValueError(
                f"Parameter '{param2.name}' has incorrect name. Expected '{param1.name}', got '{param2.name}'"
            )
