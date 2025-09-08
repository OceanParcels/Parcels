## Kernels:

- The Kernel loop has been 'vectorized', so that the input of a Kernel is not one particle anymore, but a list of particles. This means that `if`-statements in Kernels don't work anymore. Replace `if`-statements with `numpy.where` statements.
- `particle.delete()` is no longer valid. Instead, use `particle.state = StatusCode.Delete`.
- Sharing state between kernels must be done via the particle data (as the kernels are not combined under the hood anymore).
- `particl_dlon`, `particle_dlat` etc have been renamed to `particle.dlon` and `particle.dlat`.
- `particle.dt` is a np.timedelta64 object; be careful when multiplying `particle.dt` with a velocity, as its value may be cast to nanoseconds.
- The `time` argument in the Kernel signature is now standard `None` (and may be removed in the Kernel API before release of v4), so can't be used. Use `particle.time` instead.

## FieldSet

- `interp_method` has to be an Interpolation function, instead of a string.

## ParticleSet

- `repeatdt` and `lonlatdepth_dtype` have been removed from the ParticleSet.
- ParticleSet.execute() expects `numpy.datetime64`/`numpy.timedelta.64` for `runtime`, `endtime` and `dt`.
- `ParticleSet.from_field()`, `ParticleSet.from_line()`, `ParticleSet.from_list()` have been removed.

## ParticleFile

- Particlefiles should be created by `ParticleFile(...)` instead of `pset.ParticleFile(...)`
