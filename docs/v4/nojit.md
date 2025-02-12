# Rationale for dropping JIT support in v4

Parcels v4 will not support Just-In-Time (JIT) compilation. This means that the `JITParticle` class will be removed from the codebase. This decision was made for the following reasons:

1. We want to leverage the power of `xarray` and `uxarray` for data handling and interpolation. These libraries are not compatible with the JIT compilation in v3.
2. We want to make the codebase more maintainable and easier to understand. The JIT compilation pre-v4 adds complexity to the codebase and makes it harder to debug and maintain.
3. We have quite a few features in Parcels (also v3) that only work in Scipy mode (particle-particle interaction, particle-field interaction, etc.).
4. We want users to write more flexible/complex kernels. JIT doesn't support calling functions, or using methods from `numpy` or `scipy`, while this is possible in Scipy mode.

Essentially, the only advantage of JIT was its speed. But now that the ecosystem for just-in-time compilation with python has matured in the last 10 years, we want to leverage other packages and methods (`cython`, `numba`, `jax`?) and Python internals for speed-up.

Furthermore, we think we have some good ideas how to speed up Parcels itself without JIT compilation, such as relying more on vectorized operations.

In short, we think that the disadvantages of JIT in Parcels v3 outweigh the advantages, and we want to make Parcels v4 a more modern and maintainable codebase.

In our development of v4, we will first focus on making the codebase more modular and easier to extend. Once we have a working codebase, we will release this as `v4-alpha`. After that, we will start working on performance improvements.
