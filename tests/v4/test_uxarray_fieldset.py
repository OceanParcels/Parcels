from datetime import timedelta

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
        U=Field(name="U", data=ds_fesom_channel.U, grid=ds_fesom_channel.uxgrid, interp_method=UXPiecewiseConstantFace),
        V=Field(name="V", data=ds_fesom_channel.V, grid=ds_fesom_channel.uxgrid, interp_method=UXPiecewiseConstantFace),
    )
    return UV


@pytest.fixture
def uvw_fesom_channel(ds_fesom_channel) -> VectorField:
    UVW = VectorField(
        name="UVW",
        U=Field(name="U", data=ds_fesom_channel.U, grid=ds_fesom_channel.uxgrid, interp_method=UXPiecewiseConstantFace),
        V=Field(name="V", data=ds_fesom_channel.V, grid=ds_fesom_channel.uxgrid, interp_method=UXPiecewiseConstantFace),
        W=Field(name="W", data=ds_fesom_channel.W, grid=ds_fesom_channel.uxgrid, interp_method=UXPiecewiseLinearNode),
    )
    return UVW


def test_fesom_fieldset(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel])
    # Check that the fieldset has the expected properties
    assert (fieldset.fields["U"] == ds_fesom_channel.U).all()
    assert (fieldset.fields["V"] == ds_fesom_channel.V).all()


def test_fesom_in_particleset(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel])
    # Check that the fieldset has the expected properties
    assert (fieldset.fields["U"] == ds_fesom_channel.U).all()
    assert (fieldset.fields["V"] == ds_fesom_channel.V).all()
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.fieldset == fieldset


def test_set_interp_methods(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel])
    # Check that the fieldset has the expected properties
    assert (fieldset.fields["U"] == ds_fesom_channel.U).all()
    assert (fieldset.fields["V"] == ds_fesom_channel.V).all()

    # Set the interpolation method for each field
    fieldset.fields["U"].interp_method = UXPiecewiseConstantFace
    fieldset.fields["V"].interp_method = UXPiecewiseConstantFace


def test_fesom_channel(ds_fesom_channel, uvw_fesom_channel):
    fieldset = FieldSet([uvw_fesom_channel])

    # Check that the fieldset has the expected properties
    assert (fieldset.fields["U"] == ds_fesom_channel.U).all()
    assert (fieldset.fields["V"] == ds_fesom_channel.V).all()
    assert (fieldset.fields["W"] == ds_fesom_channel.W).all()

    pset = ParticleSet(fieldset, pclass=Particle)
    pset.execute(endtime=timedelta(days=1), dt=timedelta(hours=1))
