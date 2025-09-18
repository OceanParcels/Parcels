import copy
from typing import Any

import numpy as np
import xarray as xr

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
            e.add_note(f"error on key {key!r}")
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


def _dicts_equal(d1, d2):
    # compare two dictionaries, including when their entries are lists or arrays ( == throws an error then)
    if d1.keys() != d2.keys():
        return False
    for k in d1:
        v1, v2 = d1[k], d2[k]
        # Compare lists or arrays element-wise
        if isinstance(v1, (list, np.ndarray)) and isinstance(v2, (list, np.ndarray)):
            if not np.array_equal(np.array(v1), np.array(v2)):
                return False
        else:
            if v1 != v2:
                return False
    return True


def compare_datasets(ds1, ds2, ds1_name="Dataset 1", ds2_name="Dataset 2", verbose=True):
    print(f"Comparing {ds1_name} and {ds2_name}\n")

    def verbose_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    verbose_print("Dataset Attributes Comparison:")
    if ds1.attrs == ds2.attrs:
        verbose_print("  Dataset attributes are identical.")
    else:
        print("  Dataset attributes differ.")
        for attr_name in set(ds1.attrs.keys()) | set(ds2.attrs.keys()):
            if attr_name not in ds1.attrs:
                print(f"    Attribute '{attr_name}' only in {ds2_name}")
            elif attr_name not in ds2.attrs:
                print(f"    Attribute '{attr_name}' only in {ds1_name}")
            elif ds1.attrs[attr_name] != ds2.attrs[attr_name]:
                print(f"    Attribute '{attr_name}' differs:")
                print(f"      {ds1_name}: {ds1.attrs[attr_name]}")
                print(f"      {ds2_name}: {ds2.attrs[attr_name]}")
    verbose_print("-" * 30)

    # Compare dimensions
    verbose_print("Dimensions Comparison:")
    ds1_dims = set(ds1.dims)
    ds2_dims = set(ds2.dims)
    if ds1_dims == ds2_dims:
        verbose_print("  Dimension names are identical.")
    else:
        print("  Dimension names differ:")
        print(f"    {ds1_name} dims: {sorted(list(ds1_dims))}")
        print(f"    {ds2_name} dims: {sorted(list(ds2_dims))}")

    # For common dimensions, compare order (implicit by comparing coordinate values for sortedness)
    # and size (though size is parameterized and expected to be different)
    for dim_name in ds1_dims.intersection(ds2_dims):
        verbose_print(f"  Dimension '{dim_name}':")
        # Sizes will differ due to DIM_SIZE, so we don't strictly compare them.
        verbose_print(f"    {ds1_name} size: {ds1.dims[dim_name]}, {ds2_name} size: {ds2.dims[dim_name]}")
        # Check if coordinates associated with dimensions are sorted (increasing)
        if dim_name in ds1.coords and dim_name in ds2.coords:
            check_val = (
                np.timedelta64(0, "s") if isinstance(ds1[dim_name].values[0], (np.datetime64, np.timedelta64)) else 0.0
            )
            is_ds1_sorted = (
                np.all(np.diff(ds1[dim_name].values) >= check_val) if len(ds1[dim_name].values) > 1 else True
            )
            is_ds2_sorted = (
                np.all(np.diff(ds2[dim_name].values) >= check_val) if len(ds2[dim_name].values) > 1 else True
            )
            if is_ds1_sorted == is_ds2_sorted:
                verbose_print(f"    Order for '{dim_name}' is consistent (both sorted: {is_ds1_sorted})")
            else:
                print(
                    f"    Order for '{dim_name}' differs: {ds1_name} sorted: {is_ds1_sorted}, {ds2_name} sorted: {is_ds2_sorted}"
                )
    verbose_print("-" * 30)

    # Compare variables (name, attributes, dimensions used)
    verbose_print("Variables Comparison:")
    ds1_vars = set(ds1.variables.keys())
    ds2_vars = set(ds2.variables.keys())

    if ds1_vars == ds2_vars:
        verbose_print("  Variable names are identical.")
    else:
        print("  Variable names differ:")
        print(f"    {ds1_name} vars: {sorted(list(ds1_vars - ds2_vars))}")
        print(f"    {ds2_name} vars: {sorted(list(ds2_vars - ds1_vars))}")
        print(f"    Common vars: {sorted(list(ds1_vars.intersection(ds2_vars)))}")

    for var_name in ds1_vars.intersection(ds2_vars):
        verbose_print(f"  Variable '{var_name}':")
        var1 = ds1[var_name]
        var2 = ds2[var_name]

        # Compare attributes
        if _dicts_equal(var1.attrs, var2.attrs):
            verbose_print("    Attributes are identical.")
        else:
            print("    Attributes differ.")
            for attr_name in set(var1.attrs.keys()) | set(var2.attrs.keys()):
                if attr_name not in var1.attrs:
                    print(f"      Attribute '{attr_name}' only in {ds2_name}'s '{var_name}'")
                elif attr_name not in var2.attrs:
                    print(f"      Attribute '{attr_name}' only in {ds1_name}'s '{var_name}'")
                elif var1.attrs[attr_name] != var2.attrs[attr_name]:
                    print(f"      Attribute '{attr_name}' differs for '{var_name}':")
                    print(f"        {ds1_name}: {var1.attrs[attr_name]}")
                    print(f"        {ds2_name}: {var2.attrs[attr_name]}")

        # Compare dimensions used by the variable
        if var1.dims == var2.dims:
            verbose_print(f"    Dimensions used are identical: {var1.dims}")
        else:
            print("    Dimensions used differ:")
            print(f"      {ds1_name}: {var1.dims}")
            print(f"      {ds2_name}: {var2.dims}")
    verbose_print("=" * 30 + " End of Comparison " + "=" * 30)


def from_xarray_dataset_dict(d) -> xr.Dataset:
    """Reconstruct a dataset with zero data from the output of ``xarray.Dataset.to_dict(data=False)``.

    Useful in issues helping users debug fieldsets - sharing dataset schemas with associated metadata
    without sharing the data itself.

    Example
    -------
    >>> import xarray as xr
    >>> from parcels._datasets.structured.generic import datasets
    >>> ds = datasets['ds_2d_left']
    >>> d = ds.to_dict(data=False)
    >>> ds2 = from_xarray_dataset_dict(d)
    """
    return xr.Dataset.from_dict(_fill_with_dummy_data(copy.deepcopy(d)))


def _fill_with_dummy_data(d: dict[str, dict]):
    assert isinstance(d, dict)
    if "dtype" in d:
        d["data"] = np.zeros(d["shape"], dtype=d["dtype"])
        del d["dtype"]
        del d["shape"]

    for k in d:
        if isinstance(d[k], dict):
            d[k] = _fill_with_dummy_data(d[k])

    return d
