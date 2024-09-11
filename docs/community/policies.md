# Policies

Parcels, as of v3.1.0, has adopted versioning and deprecation policies.

## Versioning

Parcels mostly follows [semantic versioning](https://semver.org/), where the version number (e.g., v2.1.0) is thought of as `MAJOR.MINOR.PATCH`.

> MAJOR version for incompatible API changes<br>
> MINOR version for added functionality in a backward compatible manner<br>
> PATCH version for backward compatible bug fixes<br>

Parcels doesn't implement strict backwards compatibility between minor versions. We may make small changes that deprecate elements of the codebase (e.g., an obscure parameter that is no longer needed). Such deprecations will follow our deprecation policy.

Note when conducting research we highly recommend documenting which version of Parcels (and other packages) you are using. This can be as easy as doing `conda env export > environment.yml` alongside your project code. The Parcels version used to generate an output file is also stored as metadata entry in the `.zarr` output file.

## Deprecation policy

Deprecations in the Parcels codebase between minor releases will be handled using the following 6-month timeline:

- Functionality is marked for deprecation (e.g., in v2.1.0). This will include a warning to the user, instructions on how to update their scripts, and a note about when the feature will be removed. At this point the functionality still works as before.
- One minor release later (e.g., in v2.2.0), or at least 3 months later, the functionality will be replaced with `NotImplementedError`.
- One minor release later (e.g., in v2.3.0), or at least 3 months later, the functionality will be removed entirely.

These changes will be communicated in release notes.

Deprecations of classes or modules between minor releases will be avoided, except in the instance where it is deemed to have little to no impact on the end user (e.g., if the class/module was mistakenly included in the Public API to begin with, and isn't used in any user scripts or tutorial notebooks).
