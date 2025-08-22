import numpy as np

DATATYPES_TO_FILL_VALUES = {
    np.dtype(np.float16): np.nan,
    np.dtype(np.float32): np.nan,
    np.dtype(np.float64): np.nan,
    np.dtype(np.bool_): np.iinfo(np.int8).max,
    np.dtype(np.int8): np.iinfo(np.int8).max,
    np.dtype(np.int16): np.iinfo(np.int16).max,
    np.dtype(np.int32): np.iinfo(np.int32).max,
    np.dtype(np.int64): np.iinfo(np.int64).max,
    np.dtype(np.uint8): np.iinfo(np.uint8).max,
    np.dtype(np.uint16): np.iinfo(np.uint16).max,
    np.dtype(np.uint32): np.iinfo(np.uint32).max,
    np.dtype(np.uint64): np.iinfo(np.uint64).max,
}
