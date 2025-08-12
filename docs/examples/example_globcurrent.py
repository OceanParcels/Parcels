from datetime import timedelta
from glob import glob

import numpy as np
import pytest
import xarray as xr

import parcels


def set_globcurrent_fieldset(
    filename=None,
):
    if filename is None:
        data_folder = parcels.download_example_dataset("GlobCurrent_example_data")
        filename = str(
            data_folder / "2002*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"
        )
    variables = {
        "U": "eastward_eulerian_current_velocity",
        "V": "northward_eulerian_current_velocity",
    }
    dimensions = {"lat": "lat", "lon": "lon", "time": "time"}
    ds = xr.open_mfdataset(filename, combine="by_coords")

    return parcels.FieldSet.from_xarray_dataset(
        ds,
        variables,
        dimensions,
    )


@pytest.mark.parametrize(
    "dt, lonstart, latstart", [(3600.0, 25, -35), (-3600.0, 20, -39)]
)
def test_globcurrent_fieldset_advancetime(dt, lonstart, latstart):
    data_folder = parcels.download_example_dataset("GlobCurrent_example_data")
    basepath = str(data_folder / "20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc")
    files = sorted(glob(str(basepath)))

    fieldsetsub = set_globcurrent_fieldset(files[0:10])
    psetsub = parcels.ParticleSet.from_list(
        fieldset=fieldsetsub,
        pclass=parcels.Particle,
        lon=[lonstart],
        lat=[latstart],
    )

    fieldsetall = set_globcurrent_fieldset(files[0:10])
    psetall = parcels.ParticleSet.from_list(
        fieldset=fieldsetall,
        pclass=parcels.Particle,
        lon=[lonstart],
        lat=[latstart],
    )
    if dt < 0:
        psetsub[0].time_nextloop = fieldsetsub.U.grid.time[-1]
        psetall[0].time_nextloop = fieldsetall.U.grid.time[-1]

    psetsub.execute(parcels.AdvectionRK4, runtime=timedelta(days=7), dt=dt)
    psetall.execute(parcels.AdvectionRK4, runtime=timedelta(days=7), dt=dt)

    assert abs(psetsub[0].lon - psetall[0].lon) < 1e-4


def test_globcurrent_particles():
    fieldset = set_globcurrent_fieldset()

    lonstart = [25]
    latstart = [-35]

    pset = parcels.ParticleSet(
        fieldset, pclass=parcels.Particle, lon=lonstart, lat=latstart
    )

    pset.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=1), dt=timedelta(minutes=5)
    )

    assert abs(pset[0].lon - 23.8) < 1
    assert abs(pset[0].lat - -35.3) < 1


def test__particles_init_time():
    fieldset = set_globcurrent_fieldset()

    lonstart = [25]
    latstart = [-35]

    # tests the different ways of initialising the time of a particle
    pset = parcels.ParticleSet(
        fieldset,
        pclass=parcels.Particle,
        lon=lonstart,
        lat=latstart,
        time=np.datetime64("2002-01-15"),
    )
    pset2 = parcels.ParticleSet(
        fieldset,
        pclass=parcels.Particle,
        lon=lonstart,
        lat=latstart,
        time=14 * 86400,
    )
    pset3 = parcels.ParticleSet(
        fieldset,
        pclass=parcels.Particle,
        lon=lonstart,
        lat=latstart,
        time=np.array([np.datetime64("2002-01-15")]),
    )
    pset4 = parcels.ParticleSet(
        fieldset,
        pclass=parcels.Particle,
        lon=lonstart,
        lat=latstart,
        time=[np.datetime64("2002-01-15")],
    )
    assert pset[0].time - pset2[0].time == 0
    assert pset[0].time - pset3[0].time == 0
    assert pset[0].time - pset4[0].time == 0


def test_globcurrent_time_extrapolation_error():
    fieldset = set_globcurrent_fieldset()
    pset = parcels.ParticleSet(
        fieldset,
        pclass=parcels.Particle,
        lon=[25],
        lat=[-35],
        time=fieldset.U.grid.time[0] - timedelta(days=1).total_seconds(),
    )
    with pytest.raises(parcels.TimeExtrapolationError):
        pset.execute(
            parcels.AdvectionRK4, runtime=timedelta(days=1), dt=timedelta(minutes=5)
        )


@pytest.mark.v4alpha
@pytest.mark.skip(
    reason="This was always broken when using eager loading `deferred_load=False` for the P field. Needs to be fixed."
)
@pytest.mark.parametrize("dt", [-300, 300])
@pytest.mark.parametrize("with_starttime", [True, False])
def test_globcurrent_startparticles_between_time_arrays(dt, with_starttime):
    """Test for correctly initialising particle start times.

    When using Fields with different temporal domains, its important to intialise particles
    at the beginning of the time period where all Fields have available data (i.e., the
    intersection of the temporal domains)
    """
    fieldset = set_globcurrent_fieldset()

    data_folder = parcels.download_example_dataset("GlobCurrent_example_data")
    fnamesFeb = sorted(glob(f"{data_folder}/200202*.nc"))
    fieldset.add_field(
        parcels.Field.from_netcdf(
            fnamesFeb,
            ("P", "eastward_eulerian_current_velocity"),
            {"lat": "lat", "lon": "lon", "time": "time"},
        )
    )

    MyParticle = parcels.Particle.add_variable("sample_var", initial=0.0)

    def SampleP(particle, fieldset, time):  # pragma: no cover
        particle.sample_var += fieldset.P[
            time, particle.depth, particle.lat, particle.lon
        ]

    if with_starttime:
        time = fieldset.U.grid.time[0] if dt > 0 else fieldset.U.grid.time[-1]
        pset = parcels.ParticleSet(
            fieldset, pclass=MyParticle, lon=[25], lat=[-35], time=time
        )
    else:
        pset = parcels.ParticleSet(fieldset, pclass=MyParticle, lon=[25], lat=[-35])

    if with_starttime:
        with pytest.raises(parcels.TimeExtrapolationError):
            pset.execute(
                pset.Kernel(parcels.AdvectionRK4) + SampleP,
                runtime=timedelta(days=1),
                dt=dt,
            )
    else:
        pset.execute(
            pset.Kernel(parcels.AdvectionRK4) + SampleP,
            runtime=timedelta(days=1),
            dt=dt,
        )


def test_globcurrent_particle_independence(rundays=5):
    fieldset = set_globcurrent_fieldset()
    time0 = fieldset.U.grid.time[0]

    def DeleteP0(particle, fieldset, time):  # pragma: no cover
        if particle.trajectory == 0:
            particle.delete()

    pset0 = parcels.ParticleSet(
        fieldset, pclass=parcels.Particle, lon=[25, 25], lat=[-35, -35], time=time0
    )

    pset0.execute(
        pset0.Kernel(DeleteP0) + parcels.AdvectionRK4,
        runtime=timedelta(days=rundays),
        dt=timedelta(minutes=5),
    )

    pset1 = parcels.ParticleSet(
        fieldset, pclass=parcels.Particle, lon=[25, 25], lat=[-35, -35], time=time0
    )

    pset1.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=rundays), dt=timedelta(minutes=5)
    )

    assert np.allclose([pset0[-1].lon, pset0[-1].lat], [pset1[-1].lon, pset1[-1].lat])


@pytest.mark.parametrize("dt", [-300, 300])
@pytest.mark.parametrize("pid_offset", [0, 20])
def test_globcurrent_pset_fromfile(dt, pid_offset, tmpdir):
    filename = tmpdir.join("pset_fromparticlefile.zarr")
    fieldset = set_globcurrent_fieldset()

    parcels.Particle.setLastID(pid_offset)
    pset = parcels.ParticleSet(fieldset, pclass=parcels.Particle, lon=25, lat=-35)
    pfile = pset.ParticleFile(filename, outputdt=timedelta(hours=6))
    pset.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=1), dt=dt, output_file=pfile
    )
    pfile.write_latest_locations(pset, max(pset.time_nextloop))

    restarttime = np.nanmax if dt > 0 else np.nanmin
    pset_new = parcels.ParticleSet.from_particlefile(
        fieldset,
        pclass=parcels.Particle,
        filename=filename,
        restarttime=restarttime,
    )
    pset.execute(parcels.AdvectionRK4, runtime=timedelta(days=1), dt=dt)
    pset_new.execute(parcels.AdvectionRK4, runtime=timedelta(days=1), dt=dt)

    for var in ["lon", "lat", "depth", "time", "trajectory"]:
        assert np.allclose(
            [getattr(p, var) for p in pset], [getattr(p, var) for p in pset_new]
        )


def test_error_outputdt_not_multiple_dt(tmpdir):
    # Test that outputdt is a multiple of dt
    fieldset = set_globcurrent_fieldset()

    filepath = tmpdir.join("pfile_error_outputdt_not_multiple_dt.zarr")

    dt = 81.2584344538292  # number for which output writing fails

    pset = parcels.ParticleSet(fieldset, pclass=parcels.Particle, lon=[0], lat=[0])
    ofile = pset.ParticleFile(name=filepath, outputdt=timedelta(days=1))

    def DoNothing(particle, fieldset, time):  # pragma: no cover
        pass

    with pytest.raises(ValueError):
        pset.execute(DoNothing, runtime=timedelta(days=10), dt=dt, output_file=ofile)
