# Numba: The expected modifications

This document lists all the aspects of the parcels code that will need to be adjusted. Obviously, this document will not be complete, and is liable to change.

There are a few considerations with regards to what we will call Numba compatibility of code. The first are mainly the features that are available within to be jitted code. A lot of Numpy features are available, but not all (notably np.unique). Also creation of arrays is slightly different within JIT than outside.

A temporary playground has been created for experimentation, called FakeParcels. It is used to try different things without the need to convert all of the parcels machinery in one go.

## Parts that should be converted

### Particleset/collection. 
- The particle set/collection is most likely not directly readable/usable within the jitted code.
- One solution might be to write code that converts parcels particlesets to Numba particlesets. The disadvantage would be that it kind of defeats a bit the purpose of having different kinds of particle sets in the first place, but it would require less time for each kind of particle set.
- The other solution might be to bring even more of the particleset/collection into the Numba realm, which might be better for performance and also perhaps give more flexibility in adding new features (such as particle-particle interaction). It might make a lot of code Numba specific though.
- Currently implemented in FakeParcels.

### Grid
- The grid is probably going to have to live in Numba space, because the Field(set) needs it during interpolation
- Inheritance is a little more involved, see https://github.com/numba/numba/issues/1694.
- Currently implemented in FakeParcels.

### Gridset
- Could be simply a typed list of Grids perhaps.

### Field
- All the parts that are needed during kernel computation have to be in the Numba sphere, so that includes the field.
- The accessor `__getitem__` is available in the Numba jitclass.
- Currently implemented in FakeParcels
- It is currently quite a big amount of code in the Parcels code base.
- Perhaps interpolation can be modularized?
- Incomplete loading of the field to save memory is possible using typed lists of xd arrays, and is partly implemented in FakeParcels.

### Fieldset
- The fieldset has to be implemented in Numba Space.
- IO of the fields should (and can) be outside Numba to allow for library connections to xarray/dask.
- Some parts of the fieldset might be split off into the Python sphere.

### Application kernels
- Application kernels should conform to some principles so that they can be compiled with Numba. Most of this is probably already possible, because of the current ctypes constraints.
- Currently the fields can be accessed with `field.UV[particle]`, which might have to be transformed to having always the same number of arguments.
- One application kernel (RK4 2D) was modified for FakeParcels.

### Kernels
- A new kernel has to be created that suits the Numba code. Most of the non-performance critical parts can live in the Python realm.
- Support for exceptions in Numba is rudimentary to say the least, so propagation of errors probably needs to be handled slightly differently.
