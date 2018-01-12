from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4, ParticleFile
from parcels import compute_curvilinearGrid_rotationAngles
from argparse import ArgumentParser
import numpy as np
import pytest
from os import path

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def run_nemo_curvilinear(mode, outfile):
    data_path = path.join(path.dirname(__file__), 'NemoCurvilinear_data/')

    mesh_filename = data_path + 'mesh_mask.nc4'
    rotation_angles_filename = data_path + 'rotation_angles.nc'
    variables = {'cosU': 'cosU',
                 'sinU': 'sinU',
                 'cosV': 'cosV',
                 'sinV': 'sinV'}
    dimensions = {'U': {'lon': 'glamu', 'lat': 'gphiu'},
                  'V': {'lon': 'glamv', 'lat': 'gphiv'},
                  'F': {'lon': 'glamf', 'lat': 'gphif'}}
    compute_curvilinearGrid_rotationAngles(mesh_filename, rotation_angles_filename, variables, dimensions)

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
    field_set.add_periodic_halo(zonal=True)

    def periodicBC(particle, fieldset, time, dt):
        if particle.lon > 432:
            particle.lon -= 360

    lonp = [30 for i in range(-70, 41, 10)]
    latp = [i for i in range(-70, 41, 10)]
    timep = [0 for i in range(-70, 41, 10)]
    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp, time=timep)
    kernel = AdvectionRK4 + pset.Kernel(periodicBC)
    pfile = ParticleFile(outfile, pset, type="indexed")
    pfile.write(pset, pset[0].time)
    for _ in range(160):
        pset.execute(kernel, runtime=86400*1, dt=3600*6)
        pfile.write(pset, pset[0].time)
    assert abs(pset[0].lat - latp[0]) < 1e-3


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
    m = Basemap(projection='merc', llcrnrlat=-85, urcrnrlat=85, llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90, 91, 30), labels=[True, False, False, False])
    m.drawmeridians(np.arange(-180, 181, 60), labels=[False, False, False, True])

    T.lon[T.lon > 180] -= 360

    xs, ys = m(T.lon, T.lat)
    m.scatter(xs, ys, c=T.time, s=5)

    plt.show()


@pytest.mark.parametrize('mode', ['jit'])
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
