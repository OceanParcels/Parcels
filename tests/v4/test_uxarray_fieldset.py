import numpy as np
import pytest
import uxarray as ux

from parcels import (
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    UXPiecewiseConstantFace,
    UXPiecewiseLinearNode,
    VectorField,
    download_example_dataset,
)
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.uxgrid import UxGrid


@pytest.fixture
def ds_fesom_channel() -> ux.UxDataset:
    fesom_path = download_example_dataset("FESOM_periodic_channel")
    grid_path = f"{fesom_path}/fesom_channel.nc"
    data_path = [
        f"{fesom_path}/u.fesom_channel.nc",
        f"{fesom_path}/v.fesom_channel.nc",
        f"{fesom_path}/w.fesom_channel.nc",
    ]
    ds = ux.open_mfdataset(grid_path, data_path).rename_vars({"u": "U", "v": "V", "w": "W"})
    return ds


@pytest.fixture
def uv_fesom_channel(ds_fesom_channel) -> VectorField:
    UV = VectorField(
        name="UV",
        U=Field(
            name="U",
            data=ds_fesom_channel.U,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"]),
            interp_method=UXPiecewiseConstantFace,
        ),
        V=Field(
            name="V",
            data=ds_fesom_channel.V,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"]),
            interp_method=UXPiecewiseConstantFace,
        ),
    )
    return UV


@pytest.fixture
def uvw_fesom_channel(ds_fesom_channel) -> VectorField:
    UVW = VectorField(
        name="UVW",
        U=Field(
            name="U",
            data=ds_fesom_channel.U,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"]),
            interp_method=UXPiecewiseConstantFace,
        ),
        V=Field(
            name="V",
            data=ds_fesom_channel.V,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"]),
            interp_method=UXPiecewiseConstantFace,
        ),
        W=Field(
            name="W",
            data=ds_fesom_channel.W,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"]),
            interp_method=UXPiecewiseLinearNode,
        ),
    )
    return UVW


def test_fesom_fieldset(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel, uv_fesom_channel.U, uv_fesom_channel.V])
    # Check that the fieldset has the expected properties
    assert (fieldset.U.data == ds_fesom_channel.U).all()
    assert (fieldset.V.data == ds_fesom_channel.V).all()


def test_fesom_in_particleset(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel, uv_fesom_channel.U, uv_fesom_channel.V])

    # Check that the fieldset has the expected properties
    assert (fieldset.U.data == ds_fesom_channel.U).all()
    assert (fieldset.V.data == ds_fesom_channel.V).all()
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.fieldset == fieldset


def test_set_interp_methods(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel, uv_fesom_channel.U, uv_fesom_channel.V])
    # Check that the fieldset has the expected properties
    assert (fieldset.U.data == ds_fesom_channel.U).all()
    assert (fieldset.V.data == ds_fesom_channel.V).all()

    # Set the interpolation method for each field
    fieldset.U.interp_method = UXPiecewiseConstantFace
    fieldset.V.interp_method = UXPiecewiseConstantFace


def test_fesom2_square_delaunay_uniform_z_coordinate_eval():
    """
    Test the evaluation of a fieldset with a FESOM2 square Delaunay grid and uniform z-coordinate.
    Ensures that the fieldset can be created and evaluated correctly.
    Since the underlying data is constant, we can check that the values are as expected.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"]
    UVW = VectorField(
        name="UVW",
        U=Field(name="U", data=ds.U, grid=UxGrid(ds.uxgrid, z=ds.coords["nz"]), interp_method=UXPiecewiseConstantFace),
        V=Field(name="V", data=ds.V, grid=UxGrid(ds.uxgrid, z=ds.coords["nz"]), interp_method=UXPiecewiseConstantFace),
        W=Field(name="W", data=ds.W, grid=UxGrid(ds.uxgrid, z=ds.coords["nz"]), interp_method=UXPiecewiseLinearNode),
    )
    P = Field(name="p", data=ds.p, grid=UxGrid(ds.uxgrid, z=ds.coords["nz"]), interp_method=UXPiecewiseLinearNode)
    fieldset = FieldSet([UVW, P, UVW.U, UVW.V, UVW.W])

    assert fieldset.U.eval(time=ds.time[0].values, z=1.0, y=30.0, x=30.0, applyConversion=False) == 1.0
    assert fieldset.V.eval(time=ds.time[0].values, z=1.0, y=30.0, x=30.0, applyConversion=False) == 1.0
    assert fieldset.W.eval(time=ds.time[0].values, z=1.0, y=30.0, x=30.0, applyConversion=False) == 0.0
    assert fieldset.p.eval(time=ds.time[0].values, z=1.0, y=30.0, x=30.0, applyConversion=False) == 1.0


def test_fesom2_square_delaunay_antimeridian_eval():
    """
    Test the evaluation of a fieldset with a FESOM2 square Delaunay grid that crosses the antimeridian.
    Ensures that the fieldset can be created and evaluated correctly.
    Since the underlying data is constant, we can check that the values are as expected.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_antimeridian"]
    P = Field(name="p", data=ds.p, grid=UxGrid(ds.uxgrid, z=ds.coords["nz"]), interp_method=UXPiecewiseLinearNode)
    fieldset = FieldSet([P])

    assert np.isclose(fieldset.p.eval(time=ds.time[0].values, z=1.0, y=30.0, x=-170.0, applyConversion=False), 1.0)
    assert np.isclose(fieldset.p.eval(time=ds.time[0].values, z=1.0, y=30.0, x=-180.0, applyConversion=False), 1.0)
    assert np.isclose(fieldset.p.eval(time=ds.time[0].values, z=1.0, y=30.0, x=180.0, applyConversion=False), 1.0)
    assert np.isclose(fieldset.p.eval(time=ds.time[0].values, z=1.0, y=30.0, x=170.0, applyConversion=False), 1.0)
