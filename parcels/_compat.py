"""Import helpers for compatability between installations."""

__all__ = ["MPI", "KMeans"]

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

# KMeans is used in MPI. sklearn not installed by default
try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:
    KMeans = None
