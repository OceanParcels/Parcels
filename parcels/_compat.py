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
