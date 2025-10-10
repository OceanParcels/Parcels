# Parcels v4 migration guide

```{warning}
Version 4 of Parcels is unreleased at the moment. The information in this migration guide is a work in progress, and is subject to change. If you would like to provide feedback on this migration guide (or generally on the development of v4) please [submit an issue](https://github.com/Parcels-code/Parcels/issues/new/choose).
```

## Kernels

- The Kernel loop has been 'vectorized', so that the input of a Kernel is not one particle anymore, but a collection of particles. This means that `if`-statements in Kernels don't work anymore. Replace `if`-statements with `numpy.where` statements.
- `particle.delete()` is no longer valid. Instead, use `particle.state = StatusCode.Delete`.
- Sharing state between kernels must be done via the particle data (as the kernels are not combined under the hood anymore).
- `particl_dlon`, `particle_dlat` etc have been renamed to `particle.dlon` and `particle.dlat`.
- `particle.dt` is a np.timedelta64 object; be careful when multiplying `particle.dt` with a velocity, as its value may be cast to nanoseconds.
- The `time` argument in the Kernel signature has been removed in the Kernel API, so can't be used. Use `particle.time` instead.
- The `particle` argument in the Kernel signature has been renamed to `particles`.
- `math` functions should be replaced with array compatible equivalents (e.g., `math.sin` -> `np.sin`). Instead of `ParcelsRandom` you should use numpy's random functions.
- `particle.depth` has been changed to `particles.z` to be consistent with the [CF conventions for trajectory data](https://cfconventions.org/cf-conventions/cf-conventions.html#trajectory-data), and to make Parcels also generalizable to atmospheric contexts.

## FieldSet

- `interp_method` has to be an Interpolation function, instead of a string.

## Particle

- `Particle.add_variables()` has been replaced by `Particle.add_variable()`, which now also takes a list of `Variables`.

## ParticleSet

- `repeatdt` and `lonlatdepth_dtype` have been removed from the ParticleSet.
- ParticleSet.execute() expects `numpy.datetime64`/`numpy.timedelta.64` for `runtime`, `endtime` and `dt`.
- `ParticleSet.from_field()`, `ParticleSet.from_line()`, `ParticleSet.from_list()` have been removed.

## ParticleFile

- Particlefiles should be created by `ParticleFile(...)` instead of `pset.ParticleFile(...)`
- The `name` argument in `ParticleFile` has been replaced by `store` and can now be a string, a Path or a zarr store.
