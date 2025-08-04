Kernels:

- `particle.delete()` is no longer valid. Have to do `particle.state = StatusCode.Delete`
- Sharing state between kernels must be done via the particle data (as now the kernels are not combined under the hood).
- dt is a np.timedelta64 object

FieldSet

- `mesh` is now called `mesh_type`?
- `interp_method` has to be an Interpolation function, instead of a string

ParticleSet

- ParticleSet init had `repeatdt` and `lonlatdepth_dtype` removed
- ParticleSet.execute() expects `numpy.datetime64`/`numpy.timedelta.64` for `runtime`, `endtime` and `dt`
- `ParticleSet.from_field()`, `ParticleSet.from_line()`, `ParticleSet.from_list()` has been removed
