"""Import helpers for compatability between installations."""

__all__ = ["MPI", "KMeans"]

from typing import Any

MPI: Any | None = None
KMeans: Any | None = None

try:
    from mpi4py import MPI  # type: ignore[no-redef]
except ModuleNotFoundError:
    pass

# KMeans is used in MPI. sklearn not installed by default
try:
    from sklearn.cluster import KMeans  # type: ignore[no-redef]
except ModuleNotFoundError:
    pass


def add_note(e: Exception, note: str, *, before=False) -> Exception:  # TODO: Remove once py3.10 support is dropped
    """Implements something similar to PEP 678 but for python <3.11.

    https://stackoverflow.com/a/75549200/15545258
    """
    args = e.args
    if not args:
        arg0 = note
    else:
        arg0 = f"{note}\n{args[0]}" if before else f"{args[0]}\n{note}"
    e.args = (arg0,) + args[1:]
    return e
