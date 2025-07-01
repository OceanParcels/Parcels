from datetime import timedelta

import pytest

from parcels import (
    AdvectionEE,
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    UXPiecewiseConstantFace,
    VectorField,
)
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.uxgrid import UxGrid


@pytest.mark.parametrize("verbose_progress", [True, False])
def test_uxstommelgyre_pset_execute(verbose_progress):
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"])
    U = Field(
        name="U",
        data=ds.U,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    V = Field(
        name="V",
        data=ds.V,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    P = Field(
        name="P",
        data=ds.p,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    UV = VectorField(name="UV", U=U, V=V)
    fieldset = FieldSet([UV, UV.U, UV.V, P])
    pset = ParticleSet(
        fieldset,
        lon=[30.0],
        lat=[5.0],
        depth=[50.0],
        time=[0.0],
        pclass=Particle,
    )
    pset.execute(
        runtime=timedelta(minutes=10), dt=timedelta(seconds=60), pyfunc=AdvectionEE, verbose_progress=verbose_progress
    )


@pytest.mark.xfail(reason="Output file not implemented yet")
def test_uxstommelgyre_pset_execute_output():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"])
    U = Field(
        name="U",
        data=ds.U,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    V = Field(
        name="V",
        data=ds.V,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    P = Field(
        name="P",
        data=ds.p,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    UV = VectorField(name="UV", U=U, V=V)
    fieldset = FieldSet([UV, UV.U, UV.V, P])
    pset = ParticleSet(
        fieldset,
        lon=[30.0],
        lat=[5.0],
        depth=[50.0],
        time=[0.0],
        pclass=Particle,
    )
    output_file = pset.ParticleFile(
        name="stommel_uxarray_particles.zarr",  # the file name
        outputdt=timedelta(minutes=5),  # the time step of the outputs
    )
    pset.execute(runtime=timedelta(minutes=10), dt=timedelta(seconds=60), pyfunc=AdvectionEE, output_file=output_file)
