from parcels import FieldSet, Field, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4
from parcels import StructuredGrid, StructuredSGrid
import numpy as np
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_structured_grids(mode):

    def temp_func(lon, lat):
        return 20 + lat/1000. + 2 * np.sin(lon*2*np.pi/5000.)

    a = 10000
    b = 10000

    # Grid 0
    xdim_g0 = 201
    ydim_g0 = 201
    # Coordinates of the test fieldset (on A-grid in deg)
    lon_g0 = np.linspace(0, a, xdim_g0, dtype=np.float32)
    lat_g0 = np.linspace(0, b, ydim_g0, dtype=np.float32)
    time_g0 = np.linspace(0., 1000., 2, dtype=np.float64)
    grid_0 = StructuredGrid('grid0py', lon_g0, lat_g0, time=time_g0)

    # Grid 1
    xdim_g1 = 51
    ydim_g1 = 51
    # Coordinates of the test fieldset (on A-grid in deg)
    lon_g1 = np.linspace(0, a, xdim_g1, dtype=np.float32)
    lat_g1 = np.linspace(0, b, ydim_g1, dtype=np.float32)
    time_g1 = np.linspace(0., 1000., 2, dtype=np.float64)
    grid_1 = StructuredGrid('grid1py', lon_g1, lat_g1, time=time_g1)

    u_data = np.ones((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    u_data = 2*u_data
    u_field = Field('U', u_data, grid=grid_0, transpose=True)

    temp0_data = np.empty((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    for i in range(lon_g0.size):
        for j in range(lat_g0.size):
            temp0_data[i, j, :] = temp_func(lon_g0[i], lat_g0[j])
    temp0_field = Field('temp0', temp0_data, grid=grid_0, transpose=True)

    v_data = np.zeros((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    v_field = Field('V', v_data, grid=grid_1, transpose=True)

    temp1_data = np.empty((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    for i in range(lon_g1.size):
        for j in range(lat_g1.size):
            temp1_data[i, j, :] = temp_func(lon_g1[i], lat_g1[j])
    temp1_field = Field('temp1', temp1_data, grid=grid_1, transpose=True)

    other_fields = {}
    other_fields['temp0'] = temp0_field
    other_fields['temp1'] = temp1_field

    field_set = FieldSet(u_field, v_field, fields=other_fields)

    def sampleTemp(particle, fieldset, time, dt):
        # Note that fieldset.temp is interpolated at time=time+dt.
        # Indeed, sampleTemp is called at time=time, but the result is written
        # at time=time+dt, after the Kernel update
        particle.temp0 = fieldset.temp0[time+dt, particle.lon, particle.lat, particle.depth]
        particle.temp1 = fieldset.temp1[time+dt, particle.lon, particle.lat, particle.depth]

    class MyParticle(ptype[mode]):
        temp0 = Variable('temp0', dtype=np.float32, initial=20.)
        temp1 = Variable('temp1', dtype=np.float32, initial=20.)

    pset = ParticleSet.from_list(field_set, MyParticle, lon=[3001], lat=[5001])

    pset.execute(AdvectionRK4 + pset.Kernel(sampleTemp), runtime=1, dt=1)

    assert np.allclose(pset.particles[0].temp0, pset.particles[0].temp1, atol=1e-3)


def test_avoid_repeated_grids():

    lon_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    time_g0 = np.linspace(0, 1000, 2, dtype=np.float64)
    grid_0 = StructuredGrid('grid0py', lon_g0, lat_g0, time=time_g0)

    lon_g1 = np.linspace(0, 1000, 21, dtype=np.float32)
    lat_g1 = np.linspace(0, 1000, 21, dtype=np.float32)
    time_g1 = np.linspace(0, 1000, 2, dtype=np.float64)
    grid_1 = StructuredGrid('grid1py', lon_g1, lat_g1, time=time_g1)

    u_data = np.zeros((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    u_field = Field('U', u_data, grid=grid_0, transpose=True)

    v_data = np.zeros((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    v_field = Field('V', v_data, grid=grid_1, transpose=True)

    temp0_field = Field('temp', u_data, lon=lon_g0, lat=lat_g0, time=time_g0, transpose=True)

    other_fields = {}
    other_fields['temp0'] = temp0_field

    field_set = FieldSet(u_field, v_field, fields=other_fields)
    assert field_set.gridset.size == 2
    assert field_set.U.grid.name == 'grid0py'
    assert field_set.V.grid.name == 'grid1py'
    assert field_set.temp.grid.name == 'grid0py'


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('z4d', ['True', 'False'])
def test_s_grids(mode, z4d):
    lon_g0 = np.linspace(-3e4, 3e4, 61, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    time_g0 = np.linspace(0, 1000, 2, dtype=np.float64)
    if z4d:
        depth_g0 = np.zeros((lon_g0.size, lat_g0.size, 5, time_g0.size), dtype=np.float32)
    else:
        depth_g0 = np.zeros((lon_g0.size, lat_g0.size, 5), dtype=np.float32)

    def bath_func(lon):
        bath = (lon <= -2e4) * 20.
        bath += (lon > -2e4) * (lon < 2e4) * (110. + 90 * np.sin(lon/2e4 * np.pi/2.))
        bath += (lon >= 2e4) * 200.
        return bath
    bath = bath_func(lon_g0)

    for i in range(depth_g0.shape[0]):
        for k in range(depth_g0.shape[2]):
            if z4d:
                depth_g0[i, :, k, :] = bath[i] * k / (depth_g0.shape[2]-1)
            else:
                depth_g0[i, :, k] = bath[i] * k / (depth_g0.shape[2]-1)

    grid_0 = StructuredSGrid('grid0py', lon_g0, lat_g0, depth=depth_g0, time=time_g0)

    u_data = np.zeros((lon_g0.size, lat_g0.size, depth_g0.shape[2], time_g0.size), dtype=np.float32)
    v_data = np.zeros((lon_g0.size, lat_g0.size, depth_g0.shape[2], time_g0.size), dtype=np.float32)
    temp_data = np.zeros((lon_g0.size, lat_g0.size, depth_g0.shape[2], time_g0.size), dtype=np.float32)
    for k in range(1, depth_g0.shape[2]):
        temp_data[:, :, k, :] = k / (depth_g0.shape[2]-1.)
    u_field = Field('U', u_data, grid=grid_0, transpose=True)
    v_field = Field('V', v_data, grid=grid_0, transpose=True)
    temp_field = Field('temp', temp_data, grid=grid_0, transpose=True)

    other_fields = {}
    other_fields['temp'] = temp_field
    field_set = FieldSet(u_field, v_field, fields=other_fields)

    def sampleTemp(particle, fieldset, time, dt):
        particle.temp = fieldset.temp[time, particle.lon, particle.lat, particle.depth]

    class MyParticle(ptype[mode]):
        temp = Variable('temp', dtype=np.float32, initial=20.)

    lon = 400
    lat = 0
    ratio = .3
    pset = ParticleSet.from_list(field_set, MyParticle, lon=[lon], lat=[lat], depth=[bath_func(lon)*ratio])

    pset.execute(pset.Kernel(sampleTemp), runtime=1, dt=1)
    assert np.allclose(pset.particles[0].temp, ratio, atol=1e-4)
