import numpy as np

from tests import utils


def test_round_and_hash_float_array():
    decimals = 7
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    h = utils.round_and_hash_float_array(arr, decimals=decimals)
    assert h == 1068792616613

    delta = 10**-decimals
    arr_test = arr + 0.49 * delta
    h2 = utils.round_and_hash_float_array(arr_test, decimals=decimals)
    assert h2 == h

    arr_test = arr + 0.51 * delta
    h3 = utils.round_and_hash_float_array(arr_test, decimals=decimals)
    assert h3 != h
