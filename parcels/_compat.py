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


# for compat with v3 of parcels. Not sure if there's a better way to do this in v4...
class _AttrgetterHelper:
    def __getattr__(self, name):
        return name


_attrgetter_helper = _AttrgetterHelper()
