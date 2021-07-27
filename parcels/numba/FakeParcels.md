# A description of FakeParcels

FakeParcels is mock version of Parcels that implements only a very small subset of the parcels features. This small subset is mainly implemented in Numba to be able to compare performance with the real version of parcels.

Below is a summary of the features that are implemented by FakeParcels.

### Particle

Particles are implemented with coordinates being float64 values. The size of the particle is because of this reason probably bigger than in the real version of parcels.

### ParticleSet

They are not implemented as a structure as such, but can be produced as a typed list of Particles. This structure thus implements something similar to AoS (Array of Structures).

### Grid

2D grids (lat, lon) are implemented as a rectangular setup.

### Field

Fields are defined as a typed list of arrays which is Numba compatible.

### FieldSet

Implemented as a jitclass with both the grid and (U, V) fields.

### Field interpolation

Implemented as a four point interpolation, where the weights are 1/distance to the field points.

## Benchmarks

Run for a stationary eddy (see examples/bench_parcels.ipynb, examples/field.ipynb) show that for 6*3600 iterations (10K particles), we have about:

FakeParcels: 2m30
Parcels (JIT): 1m
Parcels (SciPy, estimate): 1500m
