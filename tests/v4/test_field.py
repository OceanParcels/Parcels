from __future__ import annotations

import numpy as np
import pytest
import uxarray as ux
import xarray as xr

from parcels import Field, UXPiecewiseConstantFace, UXPiecewiseLinearNode, VectorField
from parcels._datasets.structured.generic import T as T_structured
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.uxgrid import UxGrid
from parcels.xgrid import XGrid


def test_field_init_param_types():
    data = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(data)
    with pytest.raises(ValueError, match="Expected `name` to be a string"):
        Field(name=123, data=data["data_g"], grid=grid)

    with pytest.raises(ValueError, match="Expected `data` to be a uxarray.UxDataArray or xarray.DataArray"):
        Field(name="test", data=123, grid=grid)

    with pytest.raises(ValueError, match="Expected `grid` to be a parcels UxGrid, or parcels XGrid"):
        Field(name="test", data=data["data_g"], grid=123)

    with pytest.raises(ValueError, match="Invalid value 'invalid'. Valid options are.*"):
        Field(name="test", data=data["data_g"], grid=grid, mesh_type="invalid")


@pytest.mark.parametrize(
    "data,grid",
    [
        pytest.param(ux.UxDataArray(), XGrid.from_dataset(datasets_structured["ds_2d_left"]), id="uxdata-grid"),
        pytest.param(
            xr.DataArray(),
            UxGrid(
                datasets_unstructured["stommel_gyre_delaunay"].uxgrid,
                z=datasets_unstructured["stommel_gyre_delaunay"].coords["nz"],
            ),
            id="xarray-uxgrid",
        ),
    ],
)
def test_field_incompatible_combination(data, grid):
    with pytest.raises(ValueError, match="Incompatible data-grid combination."):
        Field(
            name="test_field",
            data=data,
            grid=grid,
        )


@pytest.mark.parametrize(
    "data,grid",
    [
        pytest.param(
            datasets_structured["ds_2d_left"]["data_g"],
            XGrid.from_dataset(datasets_structured["ds_2d_left"]),
            id="ds_2d_left",
        ),  # TODO: Perhaps this test should be expanded to cover more datasets?
    ],
)
def test_field_init_structured_grid(data, grid):
    """Test creating a field."""
    field = Field(
        name="test_field",
        data=data,
        grid=grid,
    )
    assert field.name == "test_field"
    assert field.data.equals(data)
    assert field.grid == grid


def test_field_init_fail_on_float_time_dim():
    """Test field initialisation fails when given float array as time dimension.

    (users are expected to use timedelta64 or datetime).
    """
    ds = datasets_structured["ds_2d_left"].copy()
    ds["time"] = (ds["time"].dims, np.arange(0, T_structured, dtype="float64"), ds["time"].attrs)

    data = ds["data_g"]
    grid = XGrid.from_dataset(ds)
    with pytest.raises(
        ValueError,
        match="Error getting time interval.*. Are you sure that the time dimension on the xarray dataset is stored as timedelta, datetime or cftime datetime objects\?",
    ):
        Field(
            name="test_field",
            data=data,
            grid=grid,
        )


@pytest.mark.parametrize(
    "data,grid",
    [
        pytest.param(
            datasets_structured["ds_2d_left"]["data_g"],
            XGrid.from_dataset(datasets_structured["ds_2d_left"]),
            id="ds_2d_left",
        ),
    ],
)
def test_field_time_interval(data, grid):
    """Test creating a field."""
    field = Field(name="test_field", data=data, grid=grid, mesh_type="flat")
    assert field.time_interval.left == np.datetime64("2000-01-01")
    assert field.time_interval.right == np.datetime64("2001-01-01")


def test_vectorfield_init_different_time_intervals():
    # Tests that a VectorField raises a ValueError if the component fields have different time domains.
    ...


def test_field_invalid_interpolator():
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds)

    def invalid_interpolator_wrong_signature(self, ti, position, tau, t, z, y, invalid):
        return 0.0

    # Test invalid interpolator with wrong signature
    with pytest.raises(ValueError, match=".*incorrect name.*"):
        Field(name="test", data=ds["data_g"], grid=grid, interp_method=invalid_interpolator_wrong_signature)


def test_vectorfield_invalid_interpolator():
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds)

    def invalid_interpolator_wrong_signature(self, ti, position, tau, t, z, y, applyConversion, invalid):
        return 0.0

    # Create component fields
    U = Field(name="U", data=ds["data_g"], grid=grid)
    V = Field(name="V", data=ds["data_g"], grid=grid)

    # Test invalid interpolator with wrong signature
    with pytest.raises(ValueError, match=".*incorrect name.*"):
        VectorField(name="UV", U=U, V=V, vector_interp_method=invalid_interpolator_wrong_signature)


def test_field_unstructured_z_linear():
    """Tests correctness of piecewise constant and piecewise linear interpolation methods on an unstructured grid with a vertical coordinate.
    The example dataset is a FESOM2 square Delaunay grid with uniform z-coordinate. Cell centered and layer registered data are defined to be
    linear functions of the vertical coordinate. This allows for testing of exactness of the interpolation methods.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"].copy(deep=True)

    # Change the pressure values to be linearly dependent on the vertical coordinate
    for k, z in enumerate(ds.coords["nz1"]):
        ds["p"].values[:, k, :] = z

    # Change the vertical velocity values to be linearly dependent on the vertical coordinate
    for k, z in enumerate(ds.coords["nz"]):
        ds["W"].values[:, k, :] = z

    grid = UxGrid(ds.uxgrid, z=ds.coords["nz"])
    # Note that the vertical coordinate is required to be the position of the layer interfaces ("nz"), not the mid-layers ("nz1")
    P = Field(name="p", data=ds.p, grid=grid, interp_method=UXPiecewiseConstantFace)

    # Test above first cell center - for piecewise constant, should return the depth of the first cell center
    assert np.isclose(P.eval(time=ds.time[0].values, z=10.0, y=30.0, x=30.0, applyConversion=False), 55.555557)
    # Test below first cell center, but in the first layer  - for piecewise constant, should return the depth of the first cell center
    assert np.isclose(P.eval(time=ds.time[0].values, z=65.0, y=30.0, x=30.0, applyConversion=False), 55.555557)
    # Test bottom layer  - for piecewise constant, should return the depth of the of the bottom layer cell center
    assert np.isclose(P.eval(time=ds.time[0].values, z=900.0, y=30.0, x=30.0, applyConversion=False), 944.44445801)

    W = Field(name="W", data=ds.W, grid=grid, interp_method=UXPiecewiseLinearNode)
    assert np.isclose(W.eval(time=ds.time[0].values, z=10.0, y=30.0, x=30.0, applyConversion=False), 10.0)
    assert np.isclose(W.eval(time=ds.time[0].values, z=65.0, y=30.0, x=30.0, applyConversion=False), 65.0)
    assert np.isclose(W.eval(time=ds.time[0].values, z=900.0, y=30.0, x=30.0, applyConversion=False), 900.0)


def test_field_constant_in_time():
    """Tests field evaluation for a field with no time interval (i.e., constant in time)."""
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(ds.uxgrid, z=ds.coords["nz"])
    # Note that the vertical coordinate is required to be the position of the layer interfaces ("nz"), not the mid-layers ("nz1")
    P = Field(name="p", data=ds.p, grid=grid, interp_method=UXPiecewiseConstantFace)

    # Assert that the field can be evaluated at any time, and returns the same value
    time = np.datetime64("2000-01-01T00:00:00")
    P1 = P.eval(time=time, z=10.0, y=30.0, x=30.0, applyConversion=False)
    P2 = P.eval(time=time + np.timedelta64(1, "D"), z=10.0, y=30.0, x=30.0, applyConversion=False)
    assert np.isclose(P1, P2)


def test_field_unstructured_grid_creation(): ...


def test_field_interpolation(): ...


def test_field_interpolation_out_of_spatial_bounds(): ...


def test_field_interpolation_out_of_time_bounds(): ...
