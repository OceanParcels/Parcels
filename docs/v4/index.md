# Parcels v4 development

Supported by funding from the [WarmWorld](https://www.warmworld.de) [ELPHE](https://www.kooperation-international.de/foerderung/projekte/detail/info/warmworld-elphe-ermoeglichung-von-lagranian-particle-tracking-fuer-hochaufloesende-und-unstrukturierte-gitter) project and an [NWO Vici project](https://www.nwo.nl/en/researchprogrammes/nwo-talent-programme/projects-vici/vici-2022), the Parcels team is working on a major update to the Parcels codebase.

The key goals of this update are

1. to support `Fields` on unstructured grids;
2. to allow for user-defined interpolation methods (somewhat similar to user-defined kernels);
3. to make the codebase more modular, easier to extend, and more maintainable;
4. to align Parcels more with other tools in the [Pangeo ecosystem](https://www.pangeo.io/#ecosystem), particularly by leveraging `xarray` more; and
5. to improve the performance of Parcels.

The timeline for the release of Parcels v4 is not yet fixed, but we are aiming for a release of an 'alpha' version in September 2025. This v4-alpha will have support for unstructured grids and user-defined interpolation methods, but is not yet performance-optimised.

Collaboration on v4 development is happening on the [Parcels v4 Project Board](https://github.com/orgs/Parcels-code/projects/5).

The pages below provide further background on the development of Parcels v4. You can think of this page as a "living" document as we work towards the release of v4.

```{toctree}
installation
api
nojit
TODO
Parcels v4 Project Board <https://github.com/orgs/Parcels-code/projects/5>
Parcels v4 migration guide <../community/v4-migration>
```
