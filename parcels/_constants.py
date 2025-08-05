import numpy as np

DATATYPES_TO_FILL_VALUES = {
    np.float16: np.nan,
    np.float32: np.nan,
    np.float64: np.nan,
    np.bool_: np.iinfo(np.int8).max,
    np.int8: np.iinfo(np.int8).max,
    np.int16: np.iinfo(np.int16).max,
    np.int32: np.iinfo(np.int32).max,
    np.int64: np.iinfo(np.int64).max,
    np.uint8: np.iinfo(np.uint8).max,
    np.uint16: np.iinfo(np.uint16).max,
    np.uint32: np.iinfo(np.uint32).max,
    np.uint64: np.iinfo(np.uint64).max,
}
