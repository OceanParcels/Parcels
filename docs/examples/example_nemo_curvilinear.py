"""Example script that runs a set of particles in a NEMO curvilinear grid."""

from datetime import timedelta
from glob import glob

import numpy as np
import pytest

import parcels

advection = {"RK4": parcels.AdvectionRK4, "AA": parcels.AdvectionAnalytical}


def run_nemo_curvilinear(outfile, advtype="RK4"):
    """Run parcels on the NEMO curvilinear grid."""
    data_folder = parcels.download_example_dataset("NemoCurvilinear_data")

    filenames = {
        "U": {
            "lon": f"{data_folder}/mesh_mask.nc4",
            "lat": f"{data_folder}/mesh_mask.nc4",
            "data": f"{data_folder}/U_purely_zonal-ORCA025_grid_U.nc4",
        },
        "V": {
            "lon": f"{data_folder}/mesh_mask.nc4",
            "lat": f"{data_folder}/mesh_mask.nc4",
            "data": f"{data_folder}/V_purely_zonal-ORCA025_grid_V.nc4",
        },
    }
    variables = {"U": "U", "V": "V"}
    dimensions = {"lon": "glamf", "lat": "gphif"}
    fieldset = parcels.FieldSet.from_nemo(filenames, variables, dimensions)

    # Now run particles as normal
    npart = 20
    lonp = 30 * np.ones(npart)
    if advtype == "RK4":
        latp = np.linspace(-70, 88, npart)
        runtime = timedelta(days=160)
    else:
        latp = np.linspace(-70, 70, npart)
        runtime = timedelta(days=15)

    def periodicBC(particle, fieldSet, time):  # pragma: no cover
        if particle.lon > 180:
            particle.dlon -= 360

    pset = parcels.ParticleSet.from_list(fieldset, parcels.Particle, lon=lonp, lat=latp)
    pfile = parcels.ParticleFile(outfile, pset, outputdt=timedelta(days=1))
    kernels = pset.Kernel(advection[advtype]) + periodicBC
    pset.execute(kernels, runtime=runtime, dt=timedelta(hours=6), output_file=pfile)
    assert np.allclose(pset.lat - latp, 0, atol=1e-1)


def test_nemo_curvilinear(tmpdir):
    """Test the NEMO curvilinear example."""
    outfile = tmpdir.join("nemo_particles")
    run_nemo_curvilinear(outfile)


def test_nemo_curvilinear_AA(tmpdir):
    """Test the NEMO curvilinear example with analytical advection."""
    outfile = tmpdir.join("nemo_particlesAA")
    run_nemo_curvilinear(outfile, "AA")


@pytest.mark.v4alpha
@pytest.mark.xfail(
    reason="The method for checking whether fields are on the same grid is going to change in v4 (i.e., not by looking at the dataFiles attribute)."
)
def test_nemo_3D_samegrid():
    """Test that the same grid is used for U and V in 3D NEMO fields."""
    data_folder = parcels.download_example_dataset("NemoNorthSeaORCA025-N006_data")
    ufiles = sorted(glob(f"{data_folder}/ORCA*U.nc"))
    vfiles = sorted(glob(f"{data_folder}/ORCA*V.nc"))
    wfiles = sorted(glob(f"{data_folder}/ORCA*W.nc"))
    mesh_mask = f"{data_folder}/coordinates.nc"

    filenames = {
        "U": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": ufiles},
        "V": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": vfiles},
        "W": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": wfiles},
    }

    variables = {"U": "uo", "V": "vo", "W": "wo"}
    dimensions = {
        "U": {
            "lon": "glamf",
            "lat": "gphif",
            "depth": "depthw",
            "time": "time_counter",
        },
        "V": {
            "lon": "glamf",
            "lat": "gphif",
            "depth": "depthw",
            "time": "time_counter",
        },
        "W": {
            "lon": "glamf",
            "lat": "gphif",
            "depth": "depthw",
            "time": "time_counter",
        },
    }

    fieldset = parcels.FieldSet.from_nemo(filenames, variables, dimensions)

    assert fieldset.U._dataFiles is not fieldset.W._dataFiles


def main():
    run_nemo_curvilinear("nemo_particles")


if __name__ == "__main__":
    main()
