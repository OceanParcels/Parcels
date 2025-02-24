import math
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from parcels import (
    AdvectionRK4,
    AdvectionRK4_3D,
    CurvilinearZGrid,
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    RectilinearSGrid,
    RectilinearZGrid,
    StatusCode,
    UnitConverter,
    Variable,
)
from parcels.grid import Grid, _calc_cell_edge_sizes
from parcels.tools.converters import TimeConverter
from tests.utils import TEST_DATA


def test_multi_structured_grids():
    def temp_func(lon, lat):
        return 20 + lat / 1000.0 + 2 * np.sin(lon * 2 * np.pi / 5000.0)

    a = 10000
    b = 10000

    # Grid 0
    xdim_g0 = 201
    ydim_g0 = 201
    # Coordinates of the test fieldset (on A-grid in deg)
    lon_g0 = np.linspace(0, a, xdim_g0, dtype=np.float32)
    lat_g0 = np.linspace(0, b, ydim_g0, dtype=np.float32)
    time_g0 = np.linspace(0.0, 1000.0, 2, dtype=np.float64)
    grid_0 = RectilinearZGrid(lon_g0, lat_g0, time=time_g0)

    # Grid 1
    xdim_g1 = 51
    ydim_g1 = 51
    # Coordinates of the test fieldset (on A-grid in deg)
    lon_g1 = np.linspace(0, a, xdim_g1, dtype=np.float32)
    lat_g1 = np.linspace(0, b, ydim_g1, dtype=np.float32)
    time_g1 = np.linspace(0.0, 1000.0, 2, dtype=np.float64)
    grid_1 = RectilinearZGrid(lon_g1, lat_g1, time=time_g1)

    u_data = np.ones((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    u_data = 2 * u_data
    u_field = Field("U", u_data, grid=grid_0, transpose=True)

    temp0_data = np.empty((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    for i in range(lon_g0.size):
        for j in range(lat_g0.size):
            temp0_data[i, j, :] = temp_func(lon_g0[i], lat_g0[j])
    temp0_field = Field("temp0", temp0_data, grid=grid_0, transpose=True)

    v_data = np.zeros((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    v_field = Field("V", v_data, grid=grid_1, transpose=True)

    temp1_data = np.empty((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    for i in range(lon_g1.size):
        for j in range(lat_g1.size):
            temp1_data[i, j, :] = temp_func(lon_g1[i], lat_g1[j])
    temp1_field = Field("temp1", temp1_data, grid=grid_1, transpose=True)

    other_fields = {}
    other_fields["temp0"] = temp0_field
    other_fields["temp1"] = temp1_field

    fieldset = FieldSet(u_field, v_field, fields=other_fields)

    def sampleTemp(particle, fieldset, time):  # pragma: no cover
        # Note that fieldset.temp is interpolated at time=time+dt.
        # Indeed, sampleTemp is called at time=time, but the result is written
        # at time=time+dt, after the Kernel update
        particle.temp0 = fieldset.temp0[time + particle.dt, particle.depth, particle.lat, particle.lon]
        particle.temp1 = fieldset.temp1[time + particle.dt, particle.depth, particle.lat, particle.lon]

    MyParticle = Particle.add_variables(
        [Variable("temp0", dtype=np.float32, initial=20.0), Variable("temp1", dtype=np.float32, initial=20.0)]
    )

    pset = ParticleSet.from_list(fieldset, MyParticle, lon=[3001], lat=[5001], repeatdt=1)

    pset.execute(AdvectionRK4 + pset.Kernel(sampleTemp), runtime=3, dt=1)

    # check if particle xi and yi are different for the two grids
    # assert np.all([pset.xi[i, 0] != pset.xi[i, 1] for i in range(3)])
    # assert np.all([pset.yi[i, 0] != pset.yi[i, 1] for i in range(3)])
    yi = []
    xi = []
    for p in pset:
        for e in p.ei:
            k,j,i = p.fieldset.U.unravel_index(p.ei)

    assert np.all([pset[i].xi[0] != pset[i].xi[1] for i in range(3)])
    assert np.all([pset[i].yi[0] != pset[i].yi[1] for i in range(3)])

    # advect without updating temperature to test particle deletion
    pset.remove_indices(np.array([1]))
    pset.execute(AdvectionRK4, runtime=1, dt=1)

    assert np.all([np.isclose(p.temp0, p.temp1, atol=1e-3) for p in pset])


def test_time_format_in_grid():
    lon = np.linspace(0, 1, 2, dtype=np.float32)
    lat = np.linspace(0, 1, 2, dtype=np.float32)
    time = np.array([np.datetime64("2000-01-01")] * 2)
    with pytest.raises(AssertionError, match="Time vector"):
        RectilinearZGrid(lon, lat, time=time)


def test_negate_depth():
    depth = np.linspace(0, 5, 10, dtype=np.float32)
    fieldset = FieldSet.from_data(
        {"U": np.zeros((10, 1, 1)), "V": np.zeros((10, 1, 1))}, {"lon": [0], "lat": [0], "depth": depth}
    )
    assert np.all(fieldset.gridset.grids[0].depth == depth)
    fieldset.U.grid.negate_depth()
    assert np.all(fieldset.gridset.grids[0].depth == -depth)


def test_avoid_repeated_grids():
    lon_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    time_g0 = np.linspace(0, 1000, 2, dtype=np.float64)
    grid_0 = RectilinearZGrid(lon_g0, lat_g0, time=time_g0)

    lon_g1 = np.linspace(0, 1000, 21, dtype=np.float32)
    lat_g1 = np.linspace(0, 1000, 21, dtype=np.float32)
    time_g1 = np.linspace(0, 1000, 2, dtype=np.float64)
    grid_1 = RectilinearZGrid(lon_g1, lat_g1, time=time_g1)

    u_data = np.zeros((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    u_field = Field("U", u_data, grid=grid_0, transpose=True)

    v_data = np.zeros((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    v_field = Field("V", v_data, grid=grid_1, transpose=True)

    temp0_field = Field("temp", u_data, lon=lon_g0, lat=lat_g0, time=time_g0, transpose=True)

    other_fields = {}
    other_fields["temp"] = temp0_field

    fieldset = FieldSet(u_field, v_field, fields=other_fields)
    assert fieldset.gridset.size == 2
    assert fieldset.U.grid is fieldset.temp.grid
    assert fieldset.V.grid is not fieldset.U.grid


def test_multigrids_pointer():
    lon_g0 = np.linspace(0, 1e4, 21, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    depth_g0 = np.zeros((5, lat_g0.size, lon_g0.size), dtype=np.float32)

    def bath_func(lon):
        return lon / 1000.0 + 10

    bath = bath_func(lon_g0)

    zdim = depth_g0.shape[0]
    for i in range(lon_g0.size):
        for k in range(zdim):
            depth_g0[k, :, i] = bath[i] * k / (zdim - 1)

    grid_0 = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)
    grid_1 = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)

    u_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    v_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    w_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)

    u_field = Field("U", u_data, grid=grid_0)
    v_field = Field("V", v_data, grid=grid_0)
    w_field = Field("W", w_data, grid=grid_1)

    fieldset = FieldSet(u_field, v_field, fields={"W": w_field})
    fieldset.add_periodic_halo(zonal=3, meridional=2)  # unit test of halo for SGrid

    assert u_field.grid == v_field.grid
    assert u_field.grid == w_field.grid  # w_field.grid is now supposed to be grid_1

    pset = ParticleSet.from_list(fieldset, Particle, lon=[0], lat=[0], depth=[1])

    for _ in range(10):
        pset.execute(AdvectionRK4_3D, runtime=1000, dt=500)


@pytest.mark.parametrize("z4d", ["True", "False"])
def test_rectilinear_s_grid_sampling(z4d):
    lon_g0 = np.linspace(-3e4, 3e4, 61, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    time_g0 = np.linspace(0, 1000, 2, dtype=np.float64)
    if z4d:
        depth_g0 = np.zeros((time_g0.size, 5, lat_g0.size, lon_g0.size), dtype=np.float32)
    else:
        depth_g0 = np.zeros((5, lat_g0.size, lon_g0.size), dtype=np.float32)

    def bath_func(lon):
        bath = (lon <= -2e4) * 20.0
        bath += (lon > -2e4) * (lon < 2e4) * (110.0 + 90 * np.sin(lon / 2e4 * np.pi / 2.0))
        bath += (lon >= 2e4) * 200.0
        return bath

    bath = bath_func(lon_g0)

    zdim = depth_g0.shape[-3]
    for i in range(depth_g0.shape[-1]):
        for k in range(zdim):
            if z4d:
                depth_g0[:, k, :, i] = bath[i] * k / (zdim - 1)
            else:
                depth_g0[k, :, i] = bath[i] * k / (zdim - 1)

    grid = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0, time=time_g0)

    u_data = np.zeros((grid.tdim, grid.zdim, grid.ydim, grid.xdim), dtype=np.float32)
    v_data = np.zeros((grid.tdim, grid.zdim, grid.ydim, grid.xdim), dtype=np.float32)
    temp_data = np.zeros((grid.tdim, grid.zdim, grid.ydim, grid.xdim), dtype=np.float32)
    for k in range(1, zdim):
        temp_data[:, k, :, :] = k / (zdim - 1.0)
    u_field = Field("U", u_data, grid=grid)
    v_field = Field("V", v_data, grid=grid)
    temp_field = Field("temp", temp_data, grid=grid)

    other_fields = {}
    other_fields["temp"] = temp_field
    fieldset = FieldSet(u_field, v_field, fields=other_fields)

    def sampleTemp(particle, fieldset, time):  # pragma: no cover
        particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]

    MyParticle = Particle.add_variable("temp", dtype=np.float32, initial=20.0)

    lon = 400
    lat = 0
    ratio = 0.3
    pset = ParticleSet.from_list(fieldset, MyParticle, lon=[lon], lat=[lat], depth=[bath_func(lon) * ratio])

    pset.execute(pset.Kernel(sampleTemp), runtime=1)
    assert np.allclose(pset.temp[0], ratio, atol=1e-4)


def test_rectilinear_s_grids_advect1():
    # Constant water transport towards the east. check that the particle stays at the same relative depth (z/bath)
    lon_g0 = np.linspace(0, 1e4, 21, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    depth_g0 = np.zeros((lon_g0.size, lat_g0.size, 5), dtype=np.float32)

    def bath_func(lon):
        return lon / 1000.0 + 10

    bath = bath_func(lon_g0)

    for i in range(depth_g0.shape[0]):
        for k in range(depth_g0.shape[2]):
            depth_g0[i, :, k] = bath[i] * k / (depth_g0.shape[2] - 1)
    depth_g0 = depth_g0.transpose()

    grid = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)

    zdim = depth_g0.shape[0]
    u_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    v_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    w_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    for i in range(lon_g0.size):
        u_data[:, :, i] = 1 * 10 / bath[i]
        for k in range(zdim):
            w_data[k, :, i] = u_data[k, :, i] * depth_g0[k, :, i] / bath[i] * 1e-3

    u_field = Field("U", u_data, grid=grid)
    v_field = Field("V", v_data, grid=grid)
    w_field = Field("W", w_data, grid=grid)

    fieldset = FieldSet(u_field, v_field, fields={"W": w_field})

    lon = np.zeros(11)
    lat = np.zeros(11)
    ratio = [min(i / 10.0, 0.99) for i in range(11)]
    depth = bath_func(lon) * ratio
    pset = ParticleSet.from_list(fieldset, Particle, lon=lon, lat=lat, depth=depth)

    pset.execute(AdvectionRK4_3D, runtime=10000, dt=500)
    assert np.allclose(pset.depth / bath_func(pset.lon), ratio)


def test_rectilinear_s_grids_advect2():
    # Move particle towards the east, check relative depth evolution
    lon_g0 = np.linspace(0, 1e4, 21, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    depth_g0 = np.zeros((5, lat_g0.size, lon_g0.size), dtype=np.float32)

    def bath_func(lon):
        return lon / 1000.0 + 10

    bath = bath_func(lon_g0)

    zdim = depth_g0.shape[0]
    for i in range(lon_g0.size):
        for k in range(zdim):
            depth_g0[k, :, i] = bath[i] * k / (zdim - 1)

    grid = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)

    u_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    v_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    rel_depth_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    for k in range(1, zdim):
        rel_depth_data[k, :, :] = k / (zdim - 1.0)

    u_field = Field("U", u_data, grid=grid)
    v_field = Field("V", v_data, grid=grid)
    rel_depth_field = Field("relDepth", rel_depth_data, grid=grid)
    fieldset = FieldSet(u_field, v_field, fields={"relDepth": rel_depth_field})

    MyParticle = Particle.add_variable("relDepth", dtype=np.float32, initial=20.0)

    def moveEast(particle, fieldset, time):  # pragma: no cover
        particle_dlon += 5 * particle.dt  # noqa
        particle.relDepth = fieldset.relDepth[time, particle.depth, particle.lat, particle.lon]

    depth = 0.9
    pset = ParticleSet.from_list(fieldset, MyParticle, lon=[0], lat=[0], depth=[depth])

    kernel = pset.Kernel(moveEast)
    for _ in range(10):
        pset.execute(kernel, runtime=100, dt=50)
        assert np.allclose(pset.relDepth[0], depth / bath_func(pset.lon[0]))


def test_curvilinear_grids():
    x = np.linspace(0, 1e3, 7, dtype=np.float32)
    y = np.linspace(0, 1e3, 5, dtype=np.float32)
    (xx, yy) = np.meshgrid(x, y)

    r = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    theta = theta + np.pi / 6.0

    lon = r * np.cos(theta)
    lat = r * np.sin(theta)
    time = np.array([0, 86400], dtype=np.float64)
    grid = CurvilinearZGrid(lon, lat, time=time)

    u_data = np.ones((2, y.size, x.size), dtype=np.float32)
    v_data = np.zeros((2, y.size, x.size), dtype=np.float32)
    u_data[0, :, :] = lon[:, :] + lat[:, :]
    u_field = Field("U", u_data, grid=grid, transpose=False)
    v_field = Field("V", v_data, grid=grid, transpose=False)
    fieldset = FieldSet(u_field, v_field)

    def sampleSpeed(particle, fieldset, time):  # pragma: no cover
        u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        particle.speed = math.sqrt(u * u + v * v)

    MyParticle = Particle.add_variable("speed", dtype=np.float32, initial=0.0)

    pset = ParticleSet.from_list(fieldset, MyParticle, lon=[400, -200], lat=[600, 600])
    pset.execute(pset.Kernel(sampleSpeed), runtime=1)
    assert np.allclose(pset.speed[0], 1000)


def test_nemo_grid():
    filenames = {
        "U": {
            "lon": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "lat": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "data": str(TEST_DATA / "Uu_eastward_nemo_cross_180lon.nc"),
        },
        "V": {
            "lon": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "lat": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "data": str(TEST_DATA / "Vv_eastward_nemo_cross_180lon.nc"),
        },
    }
    variables = {"U": "U", "V": "V"}
    dimensions = {"lon": "glamf", "lat": "gphif"}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

    # test ParticleSet.from_field on curvilinear grids
    ParticleSet.from_field(fieldset, Particle, start_field=fieldset.U, size=5)

    def sampleVel(particle, fieldset, time):  # pragma: no cover
        (particle.zonal, particle.meridional) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    MyParticle = Particle.add_variables(
        [Variable("zonal", dtype=np.float32, initial=0.0), Variable("meridional", dtype=np.float32, initial=0.0)]
    )

    lonp = 175.5
    latp = 81.5
    pset = ParticleSet.from_list(fieldset, MyParticle, lon=[lonp], lat=[latp])
    pset.execute(pset.Kernel(sampleVel), runtime=1)
    u = fieldset.U.units.to_source(pset.zonal[0], 0, latp, lonp)
    v = fieldset.V.units.to_source(pset.meridional[0], 0, latp, lonp)
    assert abs(u - 1) < 1e-4
    assert abs(v) < 1e-4


def test_advect_nemo():
    filenames = {
        "U": {
            "lon": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "lat": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "data": str(TEST_DATA / "Uu_eastward_nemo_cross_180lon.nc"),
        },
        "V": {
            "lon": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "lat": str(TEST_DATA / "mask_nemo_cross_180lon.nc"),
            "data": str(TEST_DATA / "Vv_eastward_nemo_cross_180lon.nc"),
        },
    }
    variables = {"U": "U", "V": "V"}
    dimensions = {"lon": "glamf", "lat": "gphif"}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

    lonp = 175.5
    latp = 81.5
    pset = ParticleSet.from_list(fieldset, Particle, lon=[lonp], lat=[latp])
    pset.execute(AdvectionRK4, runtime=timedelta(days=2), dt=timedelta(hours=6))
    assert abs(pset.lat[0] - latp) < 1e-3


@pytest.mark.parametrize("time", [True, False])
def test_cgrid_uniform_2dvel(time):
    lon = np.array([[0, 2], [0.4, 1.5]])
    lat = np.array([[0, -0.5], [0.8, 0.5]])
    U = np.array([[-99, -99], [4.4721359549995793e-01, 1.3416407864998738e00]])
    V = np.array([[-99, 1.2126781251816650e00], [-99, 1.2278812270298409e00]])

    if time:
        U = np.stack((U, U))
        V = np.stack((V, V))
        dimensions = {"lat": lat, "lon": lon, "time": np.array([0, 10])}
    else:
        dimensions = {"lat": lat, "lon": lon}
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"

    def sampleVel(particle, fieldset, time):  # pragma: no cover
        (particle.zonal, particle.meridional) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    MyParticle = Particle.add_variables(
        [Variable("zonal", dtype=np.float32, initial=0.0), Variable("meridional", dtype=np.float32, initial=0.0)]
    )

    pset = ParticleSet.from_list(fieldset, MyParticle, lon=0.7, lat=0.3)
    pset.execute(pset.Kernel(sampleVel), runtime=1)
    assert (pset[0].zonal - 1) < 1e-6
    assert (pset[0].meridional - 1) < 1e-6


@pytest.mark.parametrize("vert_mode", ["zlev", "slev1", "slev2"])
@pytest.mark.parametrize("time", [True, False])
def test_cgrid_uniform_3dvel(vert_mode, time):
    lon = np.array([[0, 2], [0.4, 1.5]])
    lat = np.array([[0, -0.5], [0.8, 0.5]])

    u0 = 4.4721359549995793e-01
    u1 = 1.3416407864998738e00
    v0 = 1.2126781251816650e00
    v1 = 1.2278812270298409e00
    w0 = 1
    w1 = 1

    if vert_mode == "zlev":
        depth = np.array([0, 1])
    elif vert_mode == "slev1":
        depth = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
    elif vert_mode == "slev2":
        depth = np.array([[[-1, -0.6], [-1.1257142857142859, -0.9]], [[1, 1.5], [0.50857142857142845, 0.8]]])
        w0 = 1.0483007922296661e00
        w1 = 1.3098951476312375e00

    U = np.array([[[-99, -99], [u0, u1]], [[-99, -99], [-99, -99]]])
    V = np.array([[[-99, v0], [-99, v1]], [[-99, -99], [-99, -99]]])
    W = np.array([[[-99, -99], [-99, w0]], [[-99, -99], [-99, w1]]])

    if time:
        U = np.stack((U, U))
        V = np.stack((V, V))
        W = np.stack((W, W))
        dimensions = {"lat": lat, "lon": lon, "depth": depth, "time": np.array([0, 10])}
    else:
        dimensions = {"lat": lat, "lon": lon, "depth": depth}
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32), "W": np.array(W, dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"
    fieldset.W.interp_method = "cgrid_velocity"

    def sampleVel(particle, fieldset, time):  # pragma: no cover
        (particle.zonal, particle.meridional, particle.vertical) = fieldset.UVW[
            time, particle.depth, particle.lat, particle.lon
        ]

    MyParticle = Particle.add_variables(
        [
            Variable("zonal", dtype=np.float32, initial=0.0),
            Variable("meridional", dtype=np.float32, initial=0.0),
            Variable("vertical", dtype=np.float32, initial=0.0),
        ]
    )

    pset = ParticleSet.from_list(fieldset, MyParticle, lon=0.7, lat=0.3, depth=0.2)
    pset.execute(pset.Kernel(sampleVel), runtime=1)
    assert abs(pset[0].zonal - 1) < 1e-6
    assert abs(pset[0].meridional - 1) < 1e-6
    assert abs(pset[0].vertical - 1) < 1e-6


@pytest.mark.parametrize("vert_mode", ["zlev", "slev1"])
@pytest.mark.parametrize("time", [True, False])
def test_cgrid_uniform_3dvel_spherical(vert_mode, time):
    dim_file = xr.open_dataset(TEST_DATA / "mask_nemo_cross_180lon.nc")
    u_file = xr.open_dataset(TEST_DATA / "Uu_eastward_nemo_cross_180lon.nc")
    v_file = xr.open_dataset(TEST_DATA / "Vv_eastward_nemo_cross_180lon.nc")
    j = 4
    i = 11
    lon = np.array(dim_file.glamf[0, j : j + 2, i : i + 2])
    lat = np.array(dim_file.gphif[0, j : j + 2, i : i + 2])
    U = np.array(u_file.U[0, j : j + 2, i : i + 2])
    V = np.array(v_file.V[0, j : j + 2, i : i + 2])
    trash = np.zeros((2, 2))
    U = np.stack((U, trash))
    V = np.stack((V, trash))
    w0 = 1
    w1 = 1
    W = np.array([[[-99, -99], [-99, w0]], [[-99, -99], [-99, w1]]])

    if vert_mode == "zlev":
        depth = np.array([0, 1])
    elif vert_mode == "slev1":
        depth = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])

    if time:
        U = np.stack((U, U))
        V = np.stack((V, V))
        W = np.stack((W, W))
        dimensions = {"lat": lat, "lon": lon, "depth": depth, "time": np.array([0, 10])}
    else:
        dimensions = {"lat": lat, "lon": lon, "depth": depth}
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32), "W": np.array(W, dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh="spherical")
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"
    fieldset.W.interp_method = "cgrid_velocity"

    def sampleVel(particle, fieldset, time):  # pragma: no cover
        (particle.zonal, particle.meridional, particle.vertical) = fieldset.UVW[
            time, particle.depth, particle.lat, particle.lon
        ]

    MyParticle = Particle.add_variables(
        [
            Variable("zonal", dtype=np.float32, initial=0.0),
            Variable("meridional", dtype=np.float32, initial=0.0),
            Variable("vertical", dtype=np.float32, initial=0.0),
        ]
    )

    lonp = 179.8
    latp = 81.35
    pset = ParticleSet.from_list(fieldset, MyParticle, lon=lonp, lat=latp, depth=0.2)
    pset.execute(pset.Kernel(sampleVel), runtime=1)
    pset.zonal[0] = fieldset.U.units.to_source(pset.zonal[0], 0, latp, lonp)
    pset.meridional[0] = fieldset.V.units.to_source(pset.meridional[0], 0, latp, lonp)
    assert abs(pset[0].zonal - 1) < 1e-3
    assert abs(pset[0].meridional) < 1e-3
    assert abs(pset[0].vertical - 1) < 1e-3


@pytest.mark.parametrize("vert_discretisation", ["zlevel", "slevel", "slevel2"])
@pytest.mark.parametrize("deferred_load", [True, False])
def test_popgrid(vert_discretisation, deferred_load):
    if vert_discretisation == "zlevel":
        w_dep = "w_dep"
    elif vert_discretisation == "slevel":
        w_dep = "w_deps"  # same as zlevel, but defined as slevel
    elif vert_discretisation == "slevel2":
        w_dep = "w_deps2"  # contains shaved cells

    filenames = str(TEST_DATA / "POPtestdata_time.nc")
    variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
    dimensions = {"lon": "lon", "lat": "lat", "depth": w_dep, "time": "time"}

    fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat", deferred_load=deferred_load)

    def sampleVel(particle, fieldset, time):  # pragma: no cover
        (particle.zonal, particle.meridional, particle.vert) = fieldset.UVW[particle]
        particle.tracer = fieldset.T[particle]

    def OutBoundsError(particle, fieldset, time):  # pragma: no cover
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.out_of_bounds = 1
            particle_ddepth -= 3  # noqa
            particle.state = StatusCode.Success

    MyParticle = Particle.add_variables(
        [
            Variable("zonal", dtype=np.float32, initial=0.0),
            Variable("meridional", dtype=np.float32, initial=0.0),
            Variable("vert", dtype=np.float32, initial=0.0),
            Variable("tracer", dtype=np.float32, initial=0.0),
            Variable("out_of_bounds", dtype=np.float32, initial=0.0),
        ]
    )

    pset = ParticleSet.from_list(fieldset, MyParticle, lon=[3, 5, 1], lat=[3, 5, 1], depth=[3, 7, 11])
    pset.execute(pset.Kernel(sampleVel) + OutBoundsError, runtime=1)
    if vert_discretisation == "slevel2":
        assert np.isclose(pset.vert[0], 0.0)
        assert np.isclose(pset.zonal[0], 0.0)
        assert np.isclose(pset.tracer[0], 99.0)
        assert np.isclose(pset.vert[1], -0.0066666666)
        assert np.isclose(pset.zonal[1], 0.015)
        assert np.isclose(pset.tracer[1], 1.0)
        assert pset.out_of_bounds[0] == 0
        assert pset.out_of_bounds[1] == 0
        assert pset.out_of_bounds[2] == 1
    else:
        assert np.allclose(pset.zonal, 0.015)
        assert np.allclose(pset.meridional, 0.01)
        assert np.allclose(pset.vert, -0.01)
        assert np.allclose(pset.tracer, 1)


@pytest.mark.parametrize("gridindexingtype", ["mitgcm", "nemo"])
@pytest.mark.parametrize("coordtype", ["rectilinear", "curvilinear"])
def test_cgrid_indexing(gridindexingtype, coordtype):
    xdim, ydim = 151, 201
    a = b = 20000  # domain size
    lon = np.linspace(-a / 2, a / 2, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    dx, dy = lon[2] - lon[1], lat[2] - lat[1]
    omega = 2 * np.pi / timedelta(days=1).total_seconds()

    index_signs = {"nemo": -1, "mitgcm": 1}
    isign = index_signs[gridindexingtype]

    def rotate_coords(lon, lat, alpha=0):
        rotmat = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        lons, lats = np.meshgrid(lon, lat)
        rotated = np.einsum("ji, mni -> jmn", rotmat, np.dstack([lons, lats]))
        return rotated[0], rotated[1]

    if coordtype == "rectilinear":
        alpha = 0
    elif coordtype == "curvilinear":
        alpha = 15 * np.pi / 180
        lon, lat = rotate_coords(lon, lat, alpha)

    def calc_r_phi(ln, lt):
        return np.sqrt(ln**2 + lt**2), np.arctan2(ln, lt)

    if coordtype == "rectilinear":

        def calculate_UVR(lat, lon, dx, dy, omega, alpha):
            U = np.zeros((lat.size, lon.size), dtype=np.float32)
            V = np.zeros((lat.size, lon.size), dtype=np.float32)
            R = np.zeros((lat.size, lon.size), dtype=np.float32)
            for i in range(lon.size):
                for j in range(lat.size):
                    r, phi = calc_r_phi(lon[i], lat[j])
                    R[j, i] = r
                    r, phi = calc_r_phi(lon[i] + isign * dx / 2, lat[j])
                    V[j, i] = -omega * r * np.sin(phi)
                    r, phi = calc_r_phi(lon[i], lat[j] + isign * dy / 2)
                    U[j, i] = omega * r * np.cos(phi)
            return U, V, R
    elif coordtype == "curvilinear":

        def calculate_UVR(lat, lon, dx, dy, omega, alpha):
            U = np.zeros(lat.shape, dtype=np.float32)
            V = np.zeros(lat.shape, dtype=np.float32)
            R = np.zeros(lat.shape, dtype=np.float32)
            for i in range(lat.shape[1]):
                for j in range(lat.shape[0]):
                    r, phi = calc_r_phi(lon[j, i], lat[j, i])
                    R[j, i] = r
                    r, phi = calc_r_phi(
                        lon[j, i] + isign * (dx / 2) * np.cos(alpha), lat[j, i] - isign * (dx / 2) * np.sin(alpha)
                    )
                    V[j, i] = np.sin(alpha) * (omega * r * np.cos(phi)) + np.cos(alpha) * (-omega * r * np.sin(phi))
                    r, phi = calc_r_phi(
                        lon[j, i] + isign * (dy / 2) * np.sin(alpha), lat[j, i] + isign * (dy / 2) * np.cos(alpha)
                    )
                    U[j, i] = np.cos(alpha) * (omega * r * np.cos(phi)) - np.sin(alpha) * (-omega * r * np.sin(phi))
            return U, V, R

    U, V, R = calculate_UVR(lat, lon, dx, dy, omega, alpha)

    data = {"U": U, "V": V, "R": R}
    dimensions = {"lon": lon, "lat": lat}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", gridindexingtype=gridindexingtype)
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"

    def UpdateR(particle, fieldset, time):  # pragma: no cover
        if time == 0:
            particle.radius_start = fieldset.R[time, particle.depth, particle.lat, particle.lon]
        particle.radius = fieldset.R[time, particle.depth, particle.lat, particle.lon]

    MyParticle = Particle.add_variables(
        [Variable("radius", dtype=np.float32, initial=0.0), Variable("radius_start", dtype=np.float32, initial=0.0)]
    )

    pset = ParticleSet(fieldset, pclass=MyParticle, lon=0, lat=4e3, time=0)

    pset.execute(pset.Kernel(UpdateR) + AdvectionRK4, runtime=timedelta(hours=14), dt=timedelta(minutes=5))
    assert np.allclose(pset.radius, pset.radius_start, atol=10)


@pytest.mark.parametrize("gridindexingtype", ["mitgcm", "nemo"])
@pytest.mark.parametrize("withtime", [False, True])
def test_cgrid_indexing_3D(gridindexingtype, withtime):
    xdim = zdim = 201
    ydim = 2
    a = c = 20000  # domain size
    b = 2
    lon = np.linspace(-a / 2, a / 2, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    depth = np.linspace(-c / 2, c / 2, zdim, dtype=np.float32)
    dx, dz = lon[1] - lon[0], depth[1] - depth[0]
    omega = 2 * np.pi / timedelta(days=1).total_seconds()
    if withtime:
        time = np.linspace(0, 24 * 60 * 60, 10)
        dimensions = {"lon": lon, "lat": lat, "depth": depth, "time": time}
        dsize = (time.size, depth.size, lat.size, lon.size)
    else:
        dimensions = {"lon": lon, "lat": lat, "depth": depth}
        dsize = (depth.size, lat.size, lon.size)

    hindex_signs = {"nemo": -1, "mitgcm": 1}
    hsign = hindex_signs[gridindexingtype]

    def calc_r_phi(ln, dp):
        # r = np.sqrt(ln ** 2 + dp ** 2)
        # phi = np.arcsin(dp/r) if r > 0 else 0
        return np.sqrt(ln**2 + dp**2), np.arctan2(ln, dp)

    def populate_UVWR(lat, lon, depth, dx, dz, omega):
        U = np.zeros(dsize, dtype=np.float32)
        V = np.zeros(dsize, dtype=np.float32)
        W = np.zeros(dsize, dtype=np.float32)
        R = np.zeros(dsize, dtype=np.float32)

        for i in range(lon.size):
            for j in range(lat.size):
                for k in range(depth.size):
                    r, phi = calc_r_phi(lon[i], depth[k])
                    if withtime:
                        R[:, k, j, i] = r
                    else:
                        R[k, j, i] = r
                    r, phi = calc_r_phi(lon[i] + hsign * dx / 2, depth[k])
                    if withtime:
                        W[:, k, j, i] = -omega * r * np.sin(phi)
                    else:
                        W[k, j, i] = -omega * r * np.sin(phi)
                    r, phi = calc_r_phi(lon[i], depth[k] + dz / 2)
                    if withtime:
                        U[:, k, j, i] = omega * r * np.cos(phi)
                    else:
                        U[k, j, i] = omega * r * np.cos(phi)
        return U, V, W, R

    U, V, W, R = populate_UVWR(lat, lon, depth, dx, dz, omega)
    data = {"U": U, "V": V, "W": W, "R": R}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", gridindexingtype=gridindexingtype)
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"
    fieldset.W.interp_method = "cgrid_velocity"

    def UpdateR(particle, fieldset, time):  # pragma: no cover
        if time == 0:
            particle.radius_start = fieldset.R[time, particle.depth, particle.lat, particle.lon]
        particle.radius = fieldset.R[time, particle.depth, particle.lat, particle.lon]

    MyParticle = Particle.add_variables(
        [Variable("radius", dtype=np.float32, initial=0.0), Variable("radius_start", dtype=np.float32, initial=0.0)]
    )

    pset = ParticleSet(fieldset, pclass=MyParticle, depth=4e3, lon=0, lat=0, time=0)

    pset.execute(pset.Kernel(UpdateR) + AdvectionRK4_3D, runtime=timedelta(hours=14), dt=timedelta(minutes=5))
    assert np.allclose(pset.radius, pset.radius_start, atol=10)


@pytest.mark.parametrize("gridindexingtype", ["pop", "mom5"])
@pytest.mark.parametrize("withtime", [False, True])
def test_bgrid_indexing_3D(gridindexingtype, withtime):
    xdim = zdim = 201
    ydim = 2
    a = c = 20000  # domain size
    b = 2
    lon = np.linspace(-a / 2, a / 2, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    depth = np.linspace(-c / 2, c / 2, zdim, dtype=np.float32)
    dx, dz = lon[1] - lon[0], depth[1] - depth[0]
    omega = 2 * np.pi / timedelta(days=1).total_seconds()
    if withtime:
        time = np.linspace(0, 24 * 60 * 60, 10)
        dimensions = {"lon": lon, "lat": lat, "depth": depth, "time": time}
        dsize = (time.size, depth.size, lat.size, lon.size)
    else:
        dimensions = {"lon": lon, "lat": lat, "depth": depth}
        dsize = (depth.size, lat.size, lon.size)

    vindex_signs = {"pop": 1, "mom5": -1}
    vsign = vindex_signs[gridindexingtype]

    def calc_r_phi(ln, dp):
        return np.sqrt(ln**2 + dp**2), np.arctan2(ln, dp)

    def populate_UVWR(lat, lon, depth, dx, dz, omega):
        U = np.zeros(dsize, dtype=np.float32)
        V = np.zeros(dsize, dtype=np.float32)
        W = np.zeros(dsize, dtype=np.float32)
        R = np.zeros(dsize, dtype=np.float32)

        for i in range(lon.size):
            for j in range(lat.size):
                for k in range(depth.size):
                    r, phi = calc_r_phi(lon[i], depth[k])
                    if withtime:
                        R[:, k, j, i] = r
                    else:
                        R[k, j, i] = r
                    r, phi = calc_r_phi(lon[i] - dx / 2, depth[k])
                    if withtime:
                        W[:, k, j, i] = -omega * r * np.sin(phi)
                    else:
                        W[k, j, i] = -omega * r * np.sin(phi)
                    # Since Parcels loads as dimensions only the depth of W-points
                    # and lon/lat of UV-points, W-points are similarly interpolated
                    # in MOM5 and POP. Indexing is shifted for UV-points.
                    r, phi = calc_r_phi(lon[i], depth[k] + vsign * dz / 2)
                    if withtime:
                        U[:, k, j, i] = omega * r * np.cos(phi)
                    else:
                        U[k, j, i] = omega * r * np.cos(phi)
        return U, V, W, R

    U, V, W, R = populate_UVWR(lat, lon, depth, dx, dz, omega)
    data = {"U": U, "V": V, "W": W, "R": R}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", gridindexingtype=gridindexingtype)
    fieldset.U.interp_method = "bgrid_velocity"
    fieldset.V.interp_method = "bgrid_velocity"
    fieldset.W.interp_method = "bgrid_w_velocity"

    def UpdateR(particle, fieldset, time):  # pragma: no cover
        if time == 0:
            particle.radius_start = fieldset.R[time, particle.depth, particle.lat, particle.lon]
        particle.radius = fieldset.R[time, particle.depth, particle.lat, particle.lon]

    MyParticle = Particle.add_variables(
        [Variable("radius", dtype=np.float32, initial=0.0), Variable("radius_start", dtype=np.float32, initial=0.0)]
    )

    pset = ParticleSet(fieldset, pclass=MyParticle, depth=-9.995e3, lon=0, lat=0, time=0)

    pset.execute(pset.Kernel(UpdateR) + AdvectionRK4_3D, runtime=timedelta(hours=14), dt=timedelta(minutes=5))
    assert np.allclose(pset.radius, pset.radius_start, atol=10)


@pytest.mark.parametrize("gridindexingtype", ["pop", "mom5"])
@pytest.mark.parametrize("extrapolation", [True, False])
def test_bgrid_interpolation(gridindexingtype, extrapolation):
    xi, yi = 3, 2
    if extrapolation:
        zi = 0 if gridindexingtype == "mom5" else -1
    else:
        zi = 2
    if gridindexingtype == "mom5":
        ufile = str(TEST_DATA / "access-om2-01_u.nc")
        vfile = str(TEST_DATA / "access-om2-01_v.nc")
        wfile = str(TEST_DATA / "access-om2-01_wt.nc")

        filenames = {
            "U": {"lon": ufile, "lat": ufile, "depth": wfile, "data": ufile},
            "V": {"lon": ufile, "lat": ufile, "depth": wfile, "data": vfile},
            "W": {"lon": ufile, "lat": ufile, "depth": wfile, "data": wfile},
        }

        variables = {"U": "u", "V": "v", "W": "wt"}

        dimensions = {
            "U": {"lon": "xu_ocean", "lat": "yu_ocean", "depth": "sw_ocean", "time": "time"},
            "V": {"lon": "xu_ocean", "lat": "yu_ocean", "depth": "sw_ocean", "time": "time"},
            "W": {"lon": "xu_ocean", "lat": "yu_ocean", "depth": "sw_ocean", "time": "time"},
        }

        fieldset = FieldSet.from_mom5(filenames, variables, dimensions)
        ds_u = xr.open_dataset(ufile)
        ds_v = xr.open_dataset(vfile)
        ds_w = xr.open_dataset(wfile)
        u = ds_u.u.isel(time=0, st_ocean=zi, yu_ocean=yi, xu_ocean=xi)
        v = ds_v.v.isel(time=0, st_ocean=zi, yu_ocean=yi, xu_ocean=xi)
        w = ds_w.wt.isel(time=0, sw_ocean=zi, yt_ocean=yi, xt_ocean=xi)

    elif gridindexingtype == "pop":
        datafname = str(TEST_DATA / "popdata.nc")
        coordfname = str(TEST_DATA / "popcoordinates.nc")
        filenames = {
            "U": {"lon": coordfname, "lat": coordfname, "depth": coordfname, "data": datafname},
            "V": {"lon": coordfname, "lat": coordfname, "depth": coordfname, "data": datafname},
            "W": {"lon": coordfname, "lat": coordfname, "depth": coordfname, "data": datafname},
        }

        variables = {"U": "UVEL", "V": "VVEL", "W": "WVEL"}
        dimensions = {"lon": "U_LON_2D", "lat": "U_LAT_2D", "depth": "w_dep"}

        fieldset = FieldSet.from_pop(filenames, variables, dimensions)
        dsc = xr.open_dataset(coordfname)
        dsd = xr.open_dataset(datafname)
        u = dsd.UVEL.isel(k=zi, j=yi, i=xi)
        v = dsd.VVEL.isel(k=zi, j=yi, i=xi)
        w = dsd.WVEL.isel(k=zi, j=yi, i=xi)

    fieldset.U.units = UnitConverter()
    fieldset.V.units = UnitConverter()

    def VelocityInterpolator(particle, fieldset, time):  # pragma: no cover
        particle.Uvel = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        particle.Vvel = fieldset.V[time, particle.depth, particle.lat, particle.lon]
        particle.Wvel = fieldset.W[time, particle.depth, particle.lat, particle.lon]

    myParticle = Particle.add_variables(
        [
            Variable("Uvel", dtype=np.float32, initial=0.0),
            Variable("Vvel", dtype=np.float32, initial=0.0),
            Variable("Wvel", dtype=np.float32, initial=0.0),
        ]
    )

    for pointtype in ["U", "V", "W"]:
        if gridindexingtype == "pop":
            if pointtype in ["U", "V"]:
                lons = dsc.U_LON_2D[yi, xi].values
                lats = dsc.U_LAT_2D[yi, xi].values
                deps = dsc.depth_t[zi].values
            elif pointtype == "W":
                lons = dsc.T_LON_2D[yi, xi].values
                lats = dsc.T_LAT_2D[yi, xi].values
                deps = dsc.w_dep[zi].values
            if extrapolation:
                deps = 5499.0
        elif gridindexingtype == "mom5":
            if pointtype in ["U", "V"]:
                lons = u.xu_ocean.data.reshape(1)
                lats = u.yu_ocean.data.reshape(1)
                deps = u.st_ocean.data.reshape(1)
            elif pointtype == "W":
                lons = w.xt_ocean.data.reshape(1)
                lats = w.yt_ocean.data.reshape(1)
                deps = w.sw_ocean.data.reshape(1)
            if extrapolation:
                deps = 0

        pset = ParticleSet.from_list(fieldset=fieldset, pclass=myParticle, lon=lons, lat=lats, depth=deps)
        pset.execute(VelocityInterpolator, runtime=1)

        convfactor = 0.01 if gridindexingtype == "pop" else 1.0
        if pointtype in ["U", "V"]:
            assert np.allclose(pset.Uvel[0], u * convfactor)
            assert np.allclose(pset.Vvel[0], v * convfactor)
        elif pointtype == "W":
            if extrapolation:
                assert np.allclose(pset.Wvel[0], 0, atol=1e-9)
            else:
                assert np.allclose(pset.Wvel[0], -w * convfactor)


@pytest.mark.parametrize(
    "lon, lat",
    [
        (np.arange(0.0, 20.0, 1.0), np.arange(0.0, 10.0, 1.0)),
    ],
)
@pytest.mark.parametrize("mesh", ["flat", "spherical"])
def test_grid_celledgesizes(lon, lat, mesh):
    grid = Grid.create_grid(
        lon=lon, lat=lat, depth=np.array([0]), time=np.array([0]), time_origin=TimeConverter(0), mesh=mesh
    )

    _calc_cell_edge_sizes(grid)
    D_meridional = grid.cell_edge_sizes["y"]
    D_zonal = grid.cell_edge_sizes["x"]
    assert np.allclose(
        D_meridional.flatten(), D_meridional[0, 0]
    )  # all meridional distances should be the same in either mesh
    if mesh == "flat":
        assert np.allclose(D_zonal.flatten(), D_zonal[0, 0])  # all zonal distances should be the same in flat mesh
    else:
        assert all((np.gradient(D_zonal, axis=0) < 0).flatten())  # zonal distances should decrease in spherical mesh
