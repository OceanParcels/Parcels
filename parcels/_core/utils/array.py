import numpy as np

from parcels._core.constants import LEFT_OUT_OF_BOUNDS, RIGHT_OUT_OF_BOUNDS


def _search_1d_array(
    arr: np.array,
    x: float,
) -> tuple[int, int]:
    """
    Searches for particle locations in a 1D array and returns barycentric coordinate along dimension.

    Assumptions:
    - array is strictly monotonically increasing.

    Parameters
    ----------
    arr : np.array
        1D array to search in.
    x : float
        Position in the 1D array to search for.

    Returns
    -------
    array of int
        Index of the element just before the position x in the array. Note that this index is -2 if the index is left out of bounds and -1 if the index is right out of bounds.
    array of float
        Barycentric coordinate.
    """
    # TODO v4: We probably rework this to deal with 0D arrays before this point (as we already know field dimensionality)
    if len(arr) < 2:
        return np.zeros(shape=x.shape, dtype=np.int32), np.zeros_like(x)
    index = np.searchsorted(arr, x, side="right") - 1
    # Use broadcasting to avoid repeated array access
    arr_index = arr[index]
    arr_next = arr[np.clip(index + 1, 1, len(arr) - 1)]  # Ensure we don't go out of bounds
    bcoord = (x - arr_index) / (arr_next - arr_index)

    # TODO check how we can avoid searchsorted when grid spacing is uniform
    # dx = arr[1] - arr[0]
    # index = ((x - arr[0]) / dx).astype(int)
    # index = np.clip(index, 0, len(arr) - 2)
    # bcoord = (x - arr[index]) / dx

    index = np.where(x < arr[0], LEFT_OUT_OF_BOUNDS, index)
    index = np.where(x >= arr[-1], RIGHT_OUT_OF_BOUNDS, index)

    return np.atleast_1d(index), np.atleast_1d(bcoord)
