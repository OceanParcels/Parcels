from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4, ParticleFile
from parcels import compute_curvilinearGrid_rotationAngles
from argparse import ArgumentParser
import numpy as np
import pytest
from datetime import timedelta as delta
from os import path

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def run_nemo_curvilinear(mode, outfile):
    """Function that shows how to read in curvilinear grids, in this case from NEMO"""
    data_path = path.join(path.dirname(__file__), 'NemoCurvilinear_data/')

    # First, create a file with the rotation angles using the compute_curvilinearGrid_rotationAngles script
    mesh_filename = data_path + 'mesh_mask.nc4'
    rotation_angles_filename = data_path + 'rotation_angles.nc'
    compute_curvilinearGrid_rotationAngles(mesh_filename, rotation_angles_filename)

    # Now define the variables and dimensions of both the zonal (U) and meridional (V)
    # velocity, as well as the rotation angles just created in the rotation_angles.nc file
    filenames = {'U': data_path + 'U_purely_zonal-ORCA025_grid_U.nc4',
                 'V': data_path + 'V_purely_zonal-ORCA025_grid_V.nc4',
                 'cosU': rotation_angles_filename,
                 'sinU': rotation_angles_filename,
                 'cosV': rotation_angles_filename,
                 'sinV': rotation_angles_filename}
    variables = {'U': 'U',
                 'V': 'V',
                 'cosU': 'cosU',
                 'sinU': 'sinU',
                 'cosV': 'cosV',
                 'sinV': 'sinV'}
    dimensions = {'U': {'lon': 'nav_lon_u', 'lat': 'nav_lat_u'},
                  'V': {'lon': 'nav_lon_v', 'lat': 'nav_lat_v'},
                  'cosU': {'lon': 'glamu', 'lat': 'gphiu'},
                  'sinU': {'lon': 'glamu', 'lat': 'gphiu'},
                  'cosV': {'lon': 'glamv', 'lat': 'gphiv'},
                  'sinV': {'lon': 'glamv', 'lat': 'gphiv'}}
    field_set = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='spherical', allow_time_extrapolation=True)

    # Now run particles as normal
    npart = 20
    lonp = 30 * np.ones(npart)
    latp = [i for i in np.linspace(-70, 88, npart)]

    def periodicBC(particle, pieldSet, time, dt):
        if particle.lon > 180:
            particle.lon -= 360

    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile(outfile, pset)
    kernels = pset.Kernel(AdvectionRK4) + periodicBC
    pset.execute(kernels, runtime=delta(days=1)*160, dt=delta(hours=6),
                 interval=delta(days=1), output_file=pfile)
    assert np.allclose([pset[i].lat - latp[i] for i in range(len(pset))], 0, atol=1e-3)


def make_plot(trajfile):
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    class ParticleData(object):
        def __init__(self):
            self.id = []

    def load_particles_file(fname, varnames):
        T = ParticleData()
        pfile = Dataset(fname, 'r')
        T.id = pfile.variables['trajectory'][:]
        for v in varnames:
            setattr(T, v, pfile.variables[v][:])
        return T

    T = load_particles_file(trajfile, ['lon', 'lat', 'time'])
    m = Basemap(projection='cyl')
    m.drawparallels(np.arange(-90, 91, 30), labels=[True, False, False, False])
    m.drawmeridians(np.arange(-180, 181, 60), labels=[False, False, False, True])

    T.lon[T.lon > 180] -= 360

    xs, ys = m(T.lon, T.lat)
    m.scatter(xs, ys, c=T.time, s=5)
    plt.show()


@pytest.mark.parametrize('mode', ['jit'])  # Only testing jit as scipy is very slow
def test_nemo_curvilinear(mode):
    outfile = 'nemo_particles'
    run_nemo_curvilinear(mode, outfile)


if __name__ == "__main__":
    p = ArgumentParser(description="""Chose the mode using mode option""")
    p.add_argument('--mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
    args = p.parse_args()

    outfile = "nemo_particles"

    run_nemo_curvilinear(args.mode, outfile)
    make_plot(outfile+'.nc')
