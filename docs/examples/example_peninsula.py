import math  # NOQA
from argparse import ArgumentParser
from datetime import timedelta

import numpy as np
import pytest

import parcels

method = {
    "RK4": parcels.AdvectionRK4,
    "EE": parcels.AdvectionEE,
    "RK45": parcels.AdvectionRK45,
}


def peninsula_fieldset(xdim, ydim, mesh="flat", grid_type="A"):
    """Construct a fieldset encapsulating the flow field around an idealised peninsula.

    Parameters
    ----------
    xdim :
        Horizontal dimension of the generated fieldset
    xdim :
        Vertical dimension of the generated fieldset
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical: Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat  (default): No conversion, lat/lon are assumed to be in m.
    grid_type :
        Option whether grid is either Arakawa A (default) or C

        The original test description can be found in Fig. 2.2.3 in:
        North, E. W., Gallego, A., Petitgas, P. (Eds). 2009. Manual of
        recommended practices for modelling physical - biological
        interactions during fish early life.
        ICES Cooperative Research Report No. 295. 111 pp.
        http://archimer.ifremer.fr/doc/00157/26792/24888.pdf
    ydim :


    """
    # Set Parcels FieldSet variables

    # Generate the original test setup on A-grid in m
    domainsizeX, domainsizeY = (1.0e5, 5.0e4)
    La = np.linspace(1e3, domainsizeX, xdim, dtype=np.float32)
    Wa = np.linspace(1e3, domainsizeY, ydim, dtype=np.float32)

    u0 = 1
    x0 = domainsizeX / 2
    R = 0.32 * domainsizeX / 2

    # Create the fields
    x, y = np.meshgrid(La, Wa, sparse=True, indexing="xy")
    P = u0 * R**2 * y / ((x - x0) ** 2 + y**2) - u0 * y

    # Set land points to zero
    landpoints = P >= 0.0
    P[landpoints] = 0.0

    if grid_type == "A":
        U = u0 - u0 * R**2 * ((x - x0) ** 2 - y**2) / (((x - x0) ** 2 + y**2) ** 2)
        V = -2 * u0 * R**2 * ((x - x0) * y) / (((x - x0) ** 2 + y**2) ** 2)
        U[landpoints] = 0.0
        V[landpoints] = 0.0
    elif grid_type == "C":
        U = np.zeros(P.shape)
        V = np.zeros(P.shape)
        V[:, 1:] = (P[:, 1:] - P[:, :-1]) / (La[1] - La[0])
        U[1:, :] = -(P[1:, :] - P[:-1, :]) / (Wa[1] - Wa[0])
    else:
        raise RuntimeError(f"Grid_type {grid_type} is not a valid option")

    # Convert from m to lat/lon for spherical meshes
    lon = La / 1852.0 / 60.0 if mesh == "spherical" else La
    lat = Wa / 1852.0 / 60.0 if mesh == "spherical" else Wa

    data = {"U": U, "V": V, "P": P}
    dimensions = {"lon": lon, "lat": lat}

    fieldset = parcels.FieldSet.from_data(data, dimensions, mesh=mesh)
    if grid_type == "C":
        fieldset.U.interp_method = "cgrid_velocity"
        fieldset.V.interp_method = "cgrid_velocity"
    return fieldset


def UpdateP(particle, fieldset, time):  # pragma: no cover
    if time == 0:
        particle.p_start = fieldset.P[time, particle.depth, particle.lat, particle.lon]
    particle.p = fieldset.P[time, particle.depth, particle.lat, particle.lon]


def peninsula_example(
    fieldset,
    outfile,
    npart,
    degree=1,
    verbose=False,
    output=True,
    method=parcels.AdvectionRK4,
):
    """Example configuration of particle flow around an idealised Peninsula

    Parameters
    ----------
    fieldset :

    outfile : str
        Basename of the input fieldset.
    npart : int
        Number of particles to intialise.
    degree :
         (Default value = 1)
    verbose :
         (Default value = False)
    output :
         (Default value = True)
    method :
         (Default value = AdvectionRK4)

    """
    # First, we define a custom Particle class to which we add a
    # custom variable, the initial stream function value p.
    MyParticle = parcels.Particle.add_variable(
        [
            parcels.Variable("p", dtype=np.float32, initial=0.0),
            parcels.Variable("p_start", dtype=np.float32, initial=0),
        ]
    )

    # Initialise particles
    if fieldset.U.grid.mesh == "flat":
        x = 3000  # 3 km offset from boundary
    else:
        x = 3.0 * (1.0 / 1.852 / 60)  # 3 km offset from boundary
    y = (
        fieldset.U.lat[0] + x,
        fieldset.U.lat[-1] - x,
    )  # latitude range, including offsets
    pset = parcels.ParticleSet.from_line(
        fieldset,
        size=npart,
        pclass=MyParticle,
        start=(x, y[0]),
        finish=(x, y[1]),
        time=0,
    )

    if verbose:
        print(f"Initial particle positions:\n{pset}")

    # Advect the particles for 24h
    time = timedelta(hours=24)
    dt = timedelta(minutes=5)
    k_adv = pset.Kernel(method)
    k_p = pset.Kernel(UpdateP)
    out = (
        pset.ParticleFile(name=outfile, outputdt=timedelta(hours=1)) if output else None
    )
    print(f"Peninsula: Advecting {npart} particles for {time}")
    pset.execute(k_adv + k_p, runtime=time, dt=dt, output_file=out)

    if verbose:
        print(f"Final particle positions:\n{pset}")

    return pset


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
def test_peninsula_fieldset(mesh, tmpdir):
    """Execute peninsula test from fieldset generated in memory."""
    fieldset = peninsula_fieldset(100, 50, mesh)
    outfile = tmpdir.join("Peninsula")
    pset = peninsula_example(fieldset, outfile, 5, degree=1)
    # Test advection accuracy by comparing streamline values
    err_adv = np.abs(pset.p_start - pset.p)
    assert (err_adv <= 1.0).all()
    # Test Field sampling accuracy by comparing kernel against Field sampling
    err_smpl = np.array(
        [
            abs(
                pset.p[i]
                - pset.fieldset.P[0.0, pset.depth[i], pset.lat[i], pset.lon[i]]
            )
            for i in range(pset.size)
        ]
    )
    assert (err_smpl <= 1.0).all()


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
def test_peninsula_fieldset_AnalyticalAdvection(mesh, tmpdir):
    """Execute peninsula test using Analytical Advection on C grid."""
    fieldset = peninsula_fieldset(101, 51, "flat", grid_type="C")
    outfile = tmpdir.join("PeninsulaAA")
    pset = peninsula_example(
        fieldset, outfile, npart=10, method=parcels.AdvectionAnalytical
    )
    # Test advection accuracy by comparing streamline values
    err_adv = np.array([abs(p.p_start - p.p) for p in pset])

    assert (err_adv <= 3.0e2).all()


def fieldsetfile(mesh, tmpdir):
    """Generate fieldset files for peninsula test."""
    filename = tmpdir.join("peninsula")
    fieldset = peninsula_fieldset(100, 50, mesh=mesh)
    fieldset.write(filename)
    return filename


def test_peninsula_file(tmpdir):
    """Open fieldset files and execute."""
    data_folder = parcels.download_example_dataset("Peninsula_data")
    filenames = {
        "U": str(data_folder / "peninsulaU.nc"),
        "V": str(data_folder / "peninsulaV.nc"),
        "P": str(data_folder / "peninsulaP.nc"),
    }
    variables = {"U": "vozocrtx", "V": "vomecrty", "P": "P"}
    dimensions = {"lon": "nav_lon", "lat": "nav_lat", "time": "time_counter"}
    fieldset = parcels.FieldSet.from_netcdf(
        filenames, variables, dimensions, allow_time_extrapolation=True
    )
    outfile = tmpdir.join("Peninsula")
    pset = peninsula_example(fieldset, outfile, 5, degree=1)
    # Test advection accuracy by comparing streamline values
    err_adv = np.abs(pset.p_start - pset.p)
    assert (err_adv <= 1.0).all()
    # Test Field sampling accuracy by comparing kernel against Field sampling
    err_smpl = np.array(
        [
            abs(
                pset.p[i]
                - pset.fieldset.P[0.0, pset.depth[i], pset.lat[i], pset.lon[i]]
            )
            for i in range(pset.size)
        ]
    )
    assert (err_smpl <= 1.0).all()


def main(args=None):
    p = ArgumentParser(
        description="""
Example of particle advection around an idealised peninsula"""
    )
    p.add_argument(
        "-p", "--particles", type=int, default=20, help="Number of particles to advect"
    )
    p.add_argument(
        "-d", "--degree", type=int, default=1, help="Degree of spatial interpolation"
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print particle information before and after execution",
    )
    p.add_argument(
        "-o",
        "--nooutput",
        action="store_true",
        default=False,
        help="Suppress trajectory output",
    )
    p.add_argument(
        "--profiling",
        action="store_true",
        default=False,
        help="Print profiling information after run",
    )
    p.add_argument(
        "-f",
        "--fieldset",
        type=int,
        nargs=2,
        default=None,
        help="Generate fieldset file with given dimensions",
    )
    p.add_argument(
        "-m",
        "--method",
        choices=("RK4", "EE", "RK45"),
        default="RK4",
        help="Numerical method used for advection",
    )
    args = p.parse_args(args)

    filename = "peninsula"
    if args.fieldset is not None:
        fieldset = peninsula_fieldset(args.fieldset[0], args.fieldset[1], mesh="flat")
    else:
        fieldset = peninsula_fieldset(100, 50, mesh="flat")
    fieldset.write(filename)

    # Open fieldset file set
    filenames = {
        "U": f"{filename}U.nc",
        "V": f"{filename}V.nc",
        "P": f"{filename}P.nc",
    }
    variables = {"U": "vozocrtx", "V": "vomecrty", "P": "P"}
    dimensions = {"lon": "nav_lon", "lat": "nav_lat", "time": "time_counter"}
    fieldset = parcels.FieldSet.from_netcdf(
        filenames, variables, dimensions, allow_time_extrapolation=True
    )

    outfile = "Peninsula"

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats

        runctx(
            "peninsula_example(fieldset, outfile, args.particles,\
                                   degree=args.degree, verbose=args.verbose,\
                                   output=not args.nooutput, method=method[args.method])",
            globals(),
            locals(),
            "Profile.prof",
        )
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        peninsula_example(
            fieldset,
            outfile,
            args.particles,
            degree=args.degree,
            verbose=args.verbose,
            output=not args.nooutput,
            method=method[args.method],
        )


if __name__ == "__main__":
    main()
