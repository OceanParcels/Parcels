"""Example script that runs a set of particles in a NEMO curvilinear grid."""
from argparse import ArgumentParser
from datetime import timedelta as delta
from glob import glob

import numpy as np
import pytest

from parcels import (
    AdvectionAnalytical,
    AdvectionRK4,
    FieldSet,
    JITParticle,
    ParticleFile,
    ParticleSet,
    ScipyParticle,
    download_example_dataset,
)

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
advection = {'RK4': AdvectionRK4, 'AA': AdvectionAnalytical}


def run_nemo_curvilinear(mode, outfile, advtype='RK4'):
    """Run parcels on the NEMO curvilinear grid."""
    data_folder = download_example_dataset('NemoCurvilinear_data')

    filenames = {'U': {'lon': f'{data_folder}/mesh_mask.nc4',
                       'lat': f'{data_folder}/mesh_mask.nc4',
                       'data': f'{data_folder}/U_purely_zonal-ORCA025_grid_U.nc4'},
                 'V': {'lon': f'{data_folder}/mesh_mask.nc4',
                       'lat': f'{data_folder}/mesh_mask.nc4',
                       'data': f'{data_folder}/V_purely_zonal-ORCA025_grid_V.nc4'}}
    variables = {'U': 'U', 'V': 'V'}
    dimensions = {'lon': 'glamf', 'lat': 'gphif'}
    chunksize = {'lat': ('y', 256), 'lon': ('x', 512)}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=chunksize)
    assert fieldset.U.chunksize == chunksize

    # Now run particles as normal
    npart = 20
    lonp = 30 * np.ones(npart)
    if advtype == 'RK4':
        latp = np.linspace(-70, 88, npart)
        runtime = delta(days=160)
    else:
        latp = np.linspace(-70, 70, npart)
        runtime = delta(days=15)

    def periodicBC(particle, fieldSet, time):
        if particle.lon > 180:
            particle_dlon -= 360  # noqa

    pset = ParticleSet.from_list(fieldset, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile(outfile, pset, outputdt=delta(days=1))
    kernels = pset.Kernel(advection[advtype]) + periodicBC
    pset.execute(kernels, runtime=runtime, dt=delta(hours=6),
                 output_file=pfile)
    assert np.allclose(pset.lat - latp, 0, atol=2e-2)


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
def test_nemo_curvilinear(mode, tmpdir):
    """Test the NEMO curvilinear example."""
    outfile = tmpdir.join('nemo_particles')
    run_nemo_curvilinear(mode, outfile)


def test_nemo_curvilinear_AA(tmpdir):
    """Test the NEMO curvilinear example with analytical advection."""
    outfile = tmpdir.join('nemo_particlesAA')
    run_nemo_curvilinear('scipy', outfile, 'AA')


def test_nemo_3D_samegrid():
    """Test that the same grid is used for U and V in 3D NEMO fields."""
    data_folder = download_example_dataset('NemoNorthSeaORCA025-N006_data')
    ufiles = sorted(glob(f'{data_folder}/ORCA*U.nc'))
    vfiles = sorted(glob(f'{data_folder}/ORCA*V.nc'))
    wfiles = sorted(glob(f'{data_folder}/ORCA*W.nc'))
    mesh_mask = f'{data_folder}/coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles}}

    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

    assert fieldset.U.dataFiles is not fieldset.W.dataFiles


def main(args=None):
    """Run the example with given arguments."""
    p = ArgumentParser(description="""Chose the mode using mode option""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
    args = p.parse_args(args)

    outfile = "nemo_particles"

    run_nemo_curvilinear(args.mode, outfile)


if __name__ == "__main__":
    main()
