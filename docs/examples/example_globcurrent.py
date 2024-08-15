from datetime import timedelta
from glob import glob

import numpy as np
import parcels
import pytest
import xarray as xr

ptype = {"scipy": parcels.ScipyParticle, "jit": parcels.JITParticle}


def set_globcurrent_fieldset(
    filename=None,
    indices=None,
    deferred_load=True,
    use_xarray=False,
    time_periodic=False,
    timestamps=None,
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
    if timestamps is None:
        dimensions = {"lat": "lat", "lon": "lon", "time": "time"}
    else:
        dimensions = {"lat": "lat", "lon": "lon"}
    if use_xarray:
        ds = xr.open_mfdataset(filename, combine="by_coords")
        return parcels.FieldSet.from_xarray_dataset(
            ds, variables, dimensions, time_periodic=time_periodic
        )
    else:
        return parcels.FieldSet.from_netcdf(
            filename,
            variables,
            dimensions,
            indices,
            deferred_load=deferred_load,
            time_periodic=time_periodic,
            timestamps=timestamps,
        )


@pytest.mark.parametrize("use_xarray", [True, False])
def test_globcurrent_fieldset(use_xarray):
    fieldset = set_globcurrent_fieldset(use_xarray=use_xarray)
    assert fieldset.U.lon.size == 81
    assert fieldset.U.lat.size == 41
    assert fieldset.V.lon.size == 81
    assert fieldset.V.lat.size == 41

    if not use_xarray:
        indices = {"lon": [5], "lat": range(20, 30)}
        fieldsetsub = set_globcurrent_fieldset(indices=indices, use_xarray=use_xarray)
        assert np.allclose(fieldsetsub.U.lon, fieldset.U.lon[indices["lon"]])
        assert np.allclose(fieldsetsub.U.lat, fieldset.U.lat[indices["lat"]])
        assert np.allclose(fieldsetsub.V.lon, fieldset.V.lon[indices["lon"]])
        assert np.allclose(fieldsetsub.V.lat, fieldset.V.lat[indices["lat"]])


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize(
    "dt, lonstart, latstart", [(3600.0, 25, -35), (-3600.0, 20, -39)]
)
@pytest.mark.parametrize("use_xarray", [True, False])
def test_globcurrent_fieldset_advancetime(mode, dt, lonstart, latstart, use_xarray):
    data_folder = parcels.download_example_dataset("GlobCurrent_example_data")
    basepath = str(data_folder / "20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc")
    files = sorted(glob(str(basepath)))

    fieldsetsub = set_globcurrent_fieldset(files[0:10], use_xarray=use_xarray)
    psetsub = parcels.ParticleSet.from_list(
        fieldset=fieldsetsub, pclass=ptype[mode], lon=[lonstart], lat=[latstart]
    )

    fieldsetall = set_globcurrent_fieldset(
        files[0:10], deferred_load=False, use_xarray=use_xarray
    )
    psetall = parcels.ParticleSet.from_list(
        fieldset=fieldsetall, pclass=ptype[mode], lon=[lonstart], lat=[latstart]
    )
    if dt < 0:
        psetsub[0].time_nextloop = fieldsetsub.U.grid.time[-1]
        psetall[0].time_nextloop = fieldsetall.U.grid.time[-1]

    psetsub.execute(parcels.AdvectionRK4, runtime=timedelta(days=7), dt=dt)
    psetall.execute(parcels.AdvectionRK4, runtime=timedelta(days=7), dt=dt)

    assert abs(psetsub[0].lon - psetall[0].lon) < 1e-4


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("use_xarray", [True, False])
def test_globcurrent_particles(mode, use_xarray):
    fieldset = set_globcurrent_fieldset(use_xarray=use_xarray)

    lonstart = [25]
    latstart = [-35]

    pset = parcels.ParticleSet(fieldset, pclass=ptype[mode], lon=lonstart, lat=latstart)

    pset.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=1), dt=timedelta(minutes=5)
    )

    assert abs(pset[0].lon - 23.8) < 1
    assert abs(pset[0].lat - -35.3) < 1


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("rundays", [300, 900])
def test_globcurrent_time_periodic(mode, rundays):
    sample_var = []
    for deferred_load in [True, False]:
        fieldset = set_globcurrent_fieldset(
            time_periodic=timedelta(days=365), deferred_load=deferred_load
        )

        MyParticle = ptype[mode].add_variable("sample_var", initial=0.0)

        pset = parcels.ParticleSet(
            fieldset, pclass=MyParticle, lon=25, lat=-35, time=fieldset.U.grid.time[0]
        )

        def SampleU(particle, fieldset, time):
            u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
            particle.sample_var += u

        pset.execute(SampleU, runtime=timedelta(days=rundays), dt=timedelta(days=1))
        sample_var.append(pset[0].sample_var)

    assert np.allclose(sample_var[0], sample_var[1])


@pytest.mark.parametrize("dt", [-300, 300])
def test_globcurrent_xarray_vs_netcdf(dt):
    fieldsetNetcdf = set_globcurrent_fieldset(use_xarray=False)
    fieldsetxarray = set_globcurrent_fieldset(use_xarray=True)
    lonstart, latstart, runtime = (25, -35, timedelta(days=7))

    psetN = parcels.ParticleSet(
        fieldsetNetcdf, pclass=parcels.JITParticle, lon=lonstart, lat=latstart
    )
    psetN.execute(parcels.AdvectionRK4, runtime=runtime, dt=dt)

    psetX = parcels.ParticleSet(
        fieldsetxarray, pclass=parcels.JITParticle, lon=lonstart, lat=latstart
    )
    psetX.execute(parcels.AdvectionRK4, runtime=runtime, dt=dt)

    assert np.allclose(psetN[0].lon, psetX[0].lon)
    assert np.allclose(psetN[0].lat, psetX[0].lat)


@pytest.mark.parametrize("dt", [-300, 300])
def test_globcurrent_netcdf_timestamps(dt):
    fieldsetNetcdf = set_globcurrent_fieldset()
    timestamps = fieldsetNetcdf.U.grid.timeslices
    fieldsetTimestamps = set_globcurrent_fieldset(timestamps=timestamps)
    lonstart, latstart, runtime = (25, -35, timedelta(days=7))

    psetN = parcels.ParticleSet(
        fieldsetNetcdf, pclass=parcels.JITParticle, lon=lonstart, lat=latstart
    )
    psetN.execute(parcels.AdvectionRK4, runtime=runtime, dt=dt)

    psetT = parcels.ParticleSet(
        fieldsetTimestamps, pclass=parcels.JITParticle, lon=lonstart, lat=latstart
    )
    psetT.execute(parcels.AdvectionRK4, runtime=runtime, dt=dt)

    assert np.allclose(psetN.lon[0], psetT.lon[0])
    assert np.allclose(psetN.lat[0], psetT.lat[0])


def test__particles_init_time():
    fieldset = set_globcurrent_fieldset()

    lonstart = [25]
    latstart = [-35]

    # tests the different ways of initialising the time of a particle
    pset = parcels.ParticleSet(
        fieldset,
        pclass=parcels.JITParticle,
        lon=lonstart,
        lat=latstart,
        time=np.datetime64("2002-01-15"),
    )
    pset2 = parcels.ParticleSet(
        fieldset,
        pclass=parcels.JITParticle,
        lon=lonstart,
        lat=latstart,
        time=14 * 86400,
    )
    pset3 = parcels.ParticleSet(
        fieldset,
        pclass=parcels.JITParticle,
        lon=lonstart,
        lat=latstart,
        time=np.array([np.datetime64("2002-01-15")]),
    )
    pset4 = parcels.ParticleSet(
        fieldset,
        pclass=parcels.JITParticle,
        lon=lonstart,
        lat=latstart,
        time=[np.datetime64("2002-01-15")],
    )
    assert pset[0].time - pset2[0].time == 0
    assert pset[0].time - pset3[0].time == 0
    assert pset[0].time - pset4[0].time == 0


@pytest.mark.xfail(reason="Time extrapolation error expected to be thrown", strict=True)
@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("use_xarray", [True, False])
def test_globcurrent_time_extrapolation_error(mode, use_xarray):
    fieldset = set_globcurrent_fieldset(use_xarray=use_xarray)

    pset = parcels.ParticleSet(
        fieldset,
        pclass=ptype[mode],
        lon=[25],
        lat=[-35],
        time=fieldset.U.time[0] - timedelta(days=1).total_seconds(),
    )

    pset.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=1), dt=timedelta(minutes=5)
    )


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("dt", [-300, 300])
@pytest.mark.parametrize("with_starttime", [True, False])
def test_globcurrent_startparticles_between_time_arrays(mode, dt, with_starttime):
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

    MyParticle = ptype[mode].add_variable("sample_var", initial=0.0)

    def SampleP(particle, fieldset, time):
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


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_globcurrent_particle_independence(mode, rundays=5):
    fieldset = set_globcurrent_fieldset()
    time0 = fieldset.U.grid.time[0]

    def DeleteP0(particle, fieldset, time):
        if particle.id == 0:
            particle.delete()

    pset0 = parcels.ParticleSet(
        fieldset, pclass=ptype[mode], lon=[25, 25], lat=[-35, -35], time=time0
    )

    pset0.execute(
        pset0.Kernel(DeleteP0) + parcels.AdvectionRK4,
        runtime=timedelta(days=rundays),
        dt=timedelta(minutes=5),
    )

    pset1 = parcels.ParticleSet(
        fieldset, pclass=ptype[mode], lon=[25, 25], lat=[-35, -35], time=time0
    )

    pset1.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=rundays), dt=timedelta(minutes=5)
    )

    assert np.allclose([pset0[-1].lon, pset0[-1].lat], [pset1[-1].lon, pset1[-1].lat])


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("dt", [-300, 300])
@pytest.mark.parametrize("pid_offset", [0, 20])
def test_globcurrent_pset_fromfile(mode, dt, pid_offset, tmpdir):
    filename = tmpdir.join("pset_fromparticlefile.zarr")
    fieldset = set_globcurrent_fieldset()

    ptype[mode].setLastID(pid_offset)
    pset = parcels.ParticleSet(fieldset, pclass=ptype[mode], lon=25, lat=-35)
    pfile = pset.ParticleFile(filename, outputdt=timedelta(hours=6))
    pset.execute(
        parcels.AdvectionRK4, runtime=timedelta(days=1), dt=dt, output_file=pfile
    )
    pfile.write_latest_locations(pset, max(pset.time_nextloop))

    restarttime = np.nanmax if dt > 0 else np.nanmin
    pset_new = parcels.ParticleSet.from_particlefile(
        fieldset, pclass=ptype[mode], filename=filename, restarttime=restarttime
    )
    pset.execute(parcels.AdvectionRK4, runtime=timedelta(days=1), dt=dt)
    pset_new.execute(parcels.AdvectionRK4, runtime=timedelta(days=1), dt=dt)

    for var in ["lon", "lat", "depth", "time", "id"]:
        assert np.allclose(
            [getattr(p, var) for p in pset], [getattr(p, var) for p in pset_new]
        )


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_error_outputdt_not_multiple_dt(mode, tmpdir):
    # Test that outputdt is a multiple of dt
    fieldset = set_globcurrent_fieldset()

    filepath = tmpdir.join("pfile_error_outputdt_not_multiple_dt.zarr")

    dt = 81.2584344538292  # number for which output writing fails

    pset = parcels.ParticleSet(fieldset, pclass=ptype[mode], lon=[0], lat=[0])
    ofile = pset.ParticleFile(name=filepath, outputdt=timedelta(days=1))

    def DoNothing(particle, fieldset, time):
        pass

    with pytest.raises(ValueError):
        pset.execute(DoNothing, runtime=timedelta(days=10), dt=dt, output_file=ofile)
