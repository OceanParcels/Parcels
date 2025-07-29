Kernels:

- `particle.delete()` is no longer valid. Have to do `particle.state = StatusCode.Delete`
- Sharing state between kernels must be done via the particle data (as now the kernels are not combined under the hood).
