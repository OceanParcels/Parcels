# API design document

## Field data input

Here is the proposed API for Field data ingestion into Parcels.

```{mermaid}

classDiagram
    class Fieldset{
        +List[Field] fields
    }
    class Field{
        +xr.Dataset|xr.DataArray|ux.Dataset|ux.DataArray data
        +parcels.Grid|ux.Grid grid
        +Interpolator interpolator
        +eval(t, z, y, x, particle=None, applyConversion=True)
    }

    class Interpolator{
        +assert_is_compatible(Field)
        +interpolate(Field, rid, t, z, y, x )
    }


    <<Protocol>> Interpolator

    Fieldset ..> Field : depends on
    Field ..> Interpolator : depends on
    Interpolator <|.. ScalarInterpolator : Realization of
    Interpolator <|.. VectorInterpolator : Realization of
    Interpolator <|.. etc : Realization of
```

Here, important things to note are:

- Interpolators (which would implement the `Interpolator` protocol) are responsible for the actual interpolation of the data, and performance considerations. There will be interpolation and indexing utilities that can be made available to the interpolators, allowing for code re-use.
  - Interpolators of the data should handle spatial periodicity and, for the case of rectilinear structured grids, without pre-computing a halo for the FieldSet and Grid ([issue](https://github.com/OceanParcels/Parcels/issues/1898)).

- In the `Field` class, not all combinations of `data`, `grid`, and `interpolator` will logically make sense (e.g., a `xr.DataArray` on a `ux.Grid`, or `ux.DataArray` on a `parcels.Grid`). It's up to the `Interpolator.assert_is_compatible(Field)` to define what is and is not compatible, and raise `ValueError` / `TypeError` on incompatible data types. The `.assert_is_compatible()` method also acts as developer documentation, defining clearly for the `.interpolate()` method what assumptions it is working on. The `.assert_is_compatible()` method should be lightweight as it will be called on `Field` initialisation.

- The `grid` object, in the case of unstructured grids, will be the `Grid` class from UXarray. For structured `Grid`s, it will be an object similar to that of `xgcm.Grid` (note that it will be very different from the v3 `Grid` object hierarchy).

- The `Field.eval` method takes as input the t,z,y,x spatio-temporal position as required arguments; the `particle` is optional and defaults to `None` and the `applyConversion` argument is optional and defaults to `True`. Initially, we will calculate the element index for a particle. As a future optimization, we could pass via the `particle` object a "cached" index value that could be used to bypass an index search. This will effectively provide `(ti,zi,yi,xi)` on a structured grid and `(ti,zi,fi)` on an unstructured grid (where `fi` is the lateral face id); within `eval` these indices will be `ravel`'ed to a single index that can be `unravel`'ed in the `interpolate` method. The `ravel`'ed index is referred to as `rid` in the `Field.Interpolator.interpolate` method. In the `interpolate` method, we envision that a user will benefit from knowing the nearest cell/index from the `ravel`'ed index (which can be `unravel`'ed) in addition the exact coordinate that we want to interpolate onto. This can permit calculation of interpolation weights using points in the neighborhood of `(t,z,y,x)`.

## Changes in API

Below a list of changes in the API that are relevant to users:

- `starttime`, `endtime` and `dt` in `ParticleSet.execute()` are now `numpy.timedelta64` or `numpy.datetime64` objects. This allows for more precise time handling and is consistent with the `numpy` time handling.

- `pid_orig` in `ParticleSet` is removed. Instead, `trajectory_ids` is used to provide a list of "trajectory" values (integers) for the particle IDs.
