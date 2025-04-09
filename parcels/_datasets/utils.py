from typing import Any

import numpy as np
import xarray as xr

from parcels._compat import add_note

_SUPPORTED_ATTR_TYPES = int | float | str | np.ndarray


def _print_mismatched_keys(d1: dict[Any, Any], d2: dict[Any, Any]) -> None:
    k1 = set(d1.keys())
    k2 = set(d2.keys())
    if len(k1 ^ k2) == 0:
        return
    print("Mismatched keys:")
    print(f"L: {k1 - k2!r}")
    print(f"R: {k2 - k1!r}")


def assert_common_attrs_equal(
    xr_attrs_1: dict[str, _SUPPORTED_ATTR_TYPES], xr_attrs_2: dict[str, _SUPPORTED_ATTR_TYPES], *, verbose: bool = True
) -> None:
    d1, d2 = xr_attrs_1, xr_attrs_2

    common_keys = set(d1.keys()) & set(d2.keys())
    if verbose:
        _print_mismatched_keys(d1, d2)

    for key in common_keys:
        try:
            if isinstance(d1[key], np.ndarray):
                np.testing.assert_array_equal(d1[key], d2[key])
            else:
                assert d1[key] == d2[key], f"{d1[key]} != {d2[key]}"
        except AssertionError as e:
            add_note(e, f"error on key {key!r}")
            raise


def assert_common_variables_common_attrs_equal(ds1: xr.Dataset, ds2: xr.Dataset, *, verbose: bool = True) -> None:
    if verbose:
        print("Checking dataset attrs...")

    assert_common_attrs_equal(ds1.attrs, ds2.attrs, verbose=verbose)

    ds1_vars = set(ds1.variables)
    ds2_vars = set(ds2.variables)

    common_variables = ds1_vars & ds2_vars
    if len(ds1_vars ^ ds2_vars) > 0 and verbose:
        print("Mismatched variables:")
        print(f"L: {ds1_vars - ds2_vars}")
        print(f"R: {ds2_vars - ds1_vars}")

    for var in common_variables:
        if verbose:
            print(f"Checking {var!r} attrs")
        assert_common_attrs_equal(ds1[var].attrs, ds2[var].attrs, verbose=verbose)


def dataset_repr_diff(ds1: xr.Dataset, ds2: xr.Dataset) -> str:
    """Return a text diff of two datasets."""
    repr1 = repr(ds1)
    repr2 = repr(ds2)
    import difflib

    diff = difflib.ndiff(repr1.splitlines(keepends=True), repr2.splitlines(keepends=True))
    return "".join(diff)
