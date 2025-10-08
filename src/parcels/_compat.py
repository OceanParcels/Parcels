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


# for compat with v3 of parcels when users provide `initial=attrgetter("lon")` to a Variable
# so that particle initial state matches another variable
class _AttrgetterHelper:
    """
    Example usage

    >>> _attrgetter_helper = _AttrgetterHelper()
    >>> _attrgetter_helper.some_attribute
    'some_attribute'
    >>> from operator import attrgetter
    >>> attrgetter('some_attribute')(_attrgetter_helper)
    'some_attribute'
    """

    def __getattr__(self, name):
        return name


_attrgetter_helper = _AttrgetterHelper()
