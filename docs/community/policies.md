# Policies

## Versioning

Parcels follows [Intended Effort Versioning (EffVer)](https://jacobtomlinson.dev/effver/), where the version number (e.g., v2.1.0) is thought of as `MACRO.MESO.MICRO`.

> MACRO version - you will need to dedicate time to upgrading to this version<br>
> MESO version - some small effort may be required for you to upgrade to this version<br>
> MICRO version - no effort is intended for you to upgrade to this version<br>

While making backward incompatible changes, we will make sure these changes and instructions to upgrade are communicated to the user via change logs or migration guides, and (where applicable) informative error messaging.

Note when conducting research we highly recommend documenting which version of Parcels (and other packages) you are using. This can be as easy as doing `conda env export > environment.yml` alongside your project code. The Parcels version used to generate an output file is also stored as metadata entry in the `.zarr` output file.

## Changes in policies

- In v4.0.0 of Parcels, adopted EffVer which formalises this "SemVer-like" variant we were following - and we adjusted our deprecation policy.
- In [v3.1.0](https://docs.oceanparcels.org/en/v3.1.0/community/policies.html) of Parcels, we adopted SemVer-like versioning and deprecation policies
