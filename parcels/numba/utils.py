import numpy as np
from numba import njit


@njit
def _numba_isclose(a, b):
    return np.absolute(a-b) <= 1e-8 + 1e-5*np.absolute(b)


@njit
def numba_reshape_34(x):
    ndim = len(x.shape)
    s = x.shape
    if ndim == 4:
        s_new = (s[0], s[1], s[2], x.size // (s[0]*s[1]*s[2]))
    else:
        s_new = (1, s[0], s[1], s[2])
    return x.reshape(*s_new)
