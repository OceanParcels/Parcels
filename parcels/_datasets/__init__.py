"""
Datasets compatible with Parcels.

This subpackage uses xarray to generate *idealised* structured and unstructured hydrodynamical datasets that are compatible with Parcels. The goals are three-fold:

1. To provide users with documentation for the types of datasets they can expect Parcels to work with. When reporting bugs, users can use these datasets to reproduce the bug they're experiencing (allowing developers to quickly troubleshoot the problem).
2. To supply our tutorials with hydrodynamical datasets.
3. To offer developers datasets for use in test cases.

Note that this subpackage is part of the private API for Parcels. Users should not rely directly on the functions defined within this module. Instead, if you want to generate your own datasets, copy the functions from this module into your own code.

Developers, note that you should only add functions that create idealised datasets to this subpackage if they are (a) quick to generate, and (b) only use dependencies already shipped with Parcels. No data files should be added to this subpackage. Real world data files should be added to the `OceanParcels/parcels-data` repository on GitHub.

Parcels Dataset Philosophy
-------------------------

When adding datasets, there may be a tension between wanting to add a specific dataset or wanting to add machinery to generate completely parameterised datasets (e.g., with different grid resolutions, with different ranges, with different datetimes etc.). There are trade-offs to both approaches:

Working with specific hardcoded datasets:

* Pros
    * the example is stable and self-contained
    * easy to see exactly what the dataset is, there is little to no dependency on other functions defined in the same module
      * datasets don't "break" due to changes in other functions (e.g., grid edges becoming out of sync with grid centres)
* Cons
    * inflexible for use in tests where you want to test a large range of datasets, or you want to test a specific resolution

Working with generated datasets is the opposite of all the above.

Most of the time we only want a single dataset. For example, for use in a tutorial, or for testing a specific feature of Parcels - such as (in the case of structured grids) checking that the grid from a certain (ocean) circulation model is correctly parsed, or checking that indexing is correctly picked up. As such, one should often opt for hardcoded datasets. These are more stable and easier to see exactly what the dataset is. We may have specific examples that become the default "go to" dataset for testing when we don't care about the details of the dataset.

Sometimes we may want to test Parcels against a whole range of datasets varying in a certain way - to ensure Parcels works as expected. For these, we should add machinery to create generated datasets.

Structure
--------

This subpackage is broken down into structured and unstructured parts. Each of these have common submodules:

* ``circulation_model`` -> hardcoded datasets with the intention of mimicking dataset structure from a certain (ocean) circulation model. If you'd like to see Parcel support a new model, please open an issue in our issue tracker.
    * exposes a dict ``datasets`` mapping dataset names to xarray datasets
* ``generic`` -> hardcoded datasets that are generic, and not tied to a certain (ocean) circulation model. Instead these focus on the fundamental properties of the dataset
    * exposes a dict ``datasets`` mapping dataset names to xarray datasets
* ``generated`` -> functions to generate datasets with varying properties
* ``utils`` -> any utility functions necessary related to either generating or validating datasets

There may be extra submodules than the ones listed above.

"""
