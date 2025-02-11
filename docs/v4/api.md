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
        +eval(...)
    }

    class Interpolator{
        +assert_is_compatible(Field)
        +interpolate(Field, ...)
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

- In the `Field` class, not all combinations of `data`, `grid`, and `interpolator` will logically make sense (e.g., a `xr.DataArray` on a `ux.Grid`, or `ux.DataArray` on a `parcels.Grid`). It's up to the `Interpolator.assert_is_compatible(Field)` to define what is and is not compatible, and raise `ValueError` / `TypeError` on incompatible data types. The `.assert_is_compatible()` method also acts as developer documentation, defining clearly for the `.interpolate()` method what assumptions it is working on. The `.assert_is_compatible()` method should be lightweight as it will be called on `Field` initialisation.

- The `grid` object, in the case of unstructured grids, will be the `Grid` class from UXarray. For structured `Grid`s, it will be an object similar to that of `xgcm.Grid` (note that it will be very different from the v3 `Grid` object hierarchy).
