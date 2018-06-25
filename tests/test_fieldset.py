from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, AdvectionRK4_3D, RectilinearZGrid
from parcels.field import Field, VectorField
from datetime import timedelta as delta
import datetime
import numpy as np
import math
import pytest
from os import path


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def generate_fieldset(xdim, ydim, zdim=1, tdim=1):
    lon = np.linspace(0., 10., xdim, dtype=np.float32)
    lat = np.linspace(0., 10., ydim, dtype=np.float32)
    depth = np.zeros(zdim, dtype=np.float32)
    time = np.zeros(tdim, dtype=np.float64)
    if zdim == 1 and tdim == 1:
        U, V = np.meshgrid(lon, lat)
        dimensions = {'lat': lat, 'lon': lon}
    else:
        U = np.ones((tdim, zdim, ydim, xdim))
        V = np.ones((tdim, zdim, ydim, xdim))
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}

    return (data, dimensions)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_fieldset_from_data(xdim, ydim):
    """ Simple test for fieldset initialisation from data. """
    data, dimensions = generate_fieldset(xdim, ydim)
    fieldset = FieldSet.from_data(data, dimensions)
    assert len(fieldset.U.data.shape) == 3
    assert len(fieldset.V.data.shape) == 3
    assert np.allclose(fieldset.U.data[0, :], data['U'], rtol=1e-12)
    assert np.allclose(fieldset.V.data[0, :], data['V'], rtol=1e-12)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 50])
def test_fieldset_from_data_different_dimensions(xdim, ydim, zdim=4, tdim=2):
    """ Test for fieldset initialisation from data using
    dict-of-dict for dimensions. """

    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.zeros(zdim, dtype=np.float32)
    time = np.zeros(tdim, dtype=np.float64)
    U = np.zeros((xdim, ydim), dtype=np.float32)
    V = np.ones((xdim, ydim), dtype=np.float32)
    P = 2 * np.ones((int(xdim/2), int(ydim/2), zdim, tdim), dtype=np.float32)
    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'U': {'lat': lat, 'lon': lon},
                  'V': {'lat': lat, 'lon': lon},
                  'P': {'lat': lat[0::2], 'lon': lon[0::2], 'depth': depth, 'time': time}}

    fieldset = FieldSet.from_data(data, dimensions, transpose=True)
    assert len(fieldset.U.data.shape) == 3
    assert len(fieldset.V.data.shape) == 3
    assert len(fieldset.P.data.shape) == 4
    assert fieldset.P.data.shape == (tdim, zdim, ydim/2, xdim/2)
    assert np.allclose(fieldset.U.data, 0., rtol=1e-12)
    assert np.allclose(fieldset.V.data, 1., rtol=1e-12)
    assert np.allclose(fieldset.P.data, 2., rtol=1e-12)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_fieldset_from_parcels(xdim, ydim, tmpdir, filename='test_parcels'):
    """ Simple test for fieldset initialisation from Parcels FieldSet file format. """
    filepath = tmpdir.join(filename)
    data, dimensions = generate_fieldset(xdim, ydim)
    fieldset_out = FieldSet.from_data(data, dimensions)
    fieldset_out.write(filepath)
    fieldset = FieldSet.from_parcels(filepath)
    assert len(fieldset.U.data.shape) == 3  # Will be 4 once we use depth
    assert len(fieldset.V.data.shape) == 3
    assert np.allclose(fieldset.U.data[0, :], data['U'], rtol=1e-12)
    assert np.allclose(fieldset.V.data[0, :], data['V'], rtol=1e-12)


@pytest.mark.parametrize('indslon', [range(10, 20), [1]])
@pytest.mark.parametrize('indslat', [range(30, 60), [22]])
def test_fieldset_from_file_subsets(indslon, indslat, tmpdir, filename='test_subsets'):
    """ Test for subsetting fieldset from file using indices dict. """
    data, dimensions = generate_fieldset(100, 100)
    filepath = tmpdir.join(filename)
    fieldsetfull = FieldSet.from_data(data, dimensions)
    fieldsetfull.write(filepath)
    indices = {'lon': indslon, 'lat': indslat}
    fieldsetsub = FieldSet.from_parcels(filepath, indices=indices)
    assert np.allclose(fieldsetsub.U.lon, fieldsetfull.U.grid.lon[indices['lon']])
    assert np.allclose(fieldsetsub.U.lat, fieldsetfull.U.grid.lat[indices['lat']])
    assert np.allclose(fieldsetsub.V.lon, fieldsetfull.V.grid.lon[indices['lon']])
    assert np.allclose(fieldsetsub.V.lat, fieldsetfull.V.grid.lat[indices['lat']])

    ixgrid = np.ix_([0], indices['lat'], indices['lon'])
    assert np.allclose(fieldsetsub.U.data, fieldsetfull.U.data[ixgrid])
    assert np.allclose(fieldsetsub.V.data, fieldsetfull.V.data[ixgrid])


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_add_field(xdim, ydim, tmpdir, filename='test_add'):
    filepath = tmpdir.join(filename)
    data, dimensions = generate_fieldset(xdim, ydim)
    fieldset = FieldSet.from_data(data, dimensions)
    field = Field('newfld', fieldset.U.data, lon=fieldset.U.lon, lat=fieldset.U.lat)
    fieldset.add_field(field)
    assert fieldset.newfld.data.shape == fieldset.U.data.shape
    fieldset.write(filepath)


@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_fieldset_celledgesizes(mesh):
    data, dimensions = generate_fieldset(10, 7)
    fieldset = FieldSet.from_data(data, dimensions, mesh=mesh)

    fieldset.U.calc_cell_edge_sizes()
    D_meridional = fieldset.U.cell_edge_sizes['y']
    D_zonal = fieldset.U.cell_edge_sizes['x']

    assert np.allclose(D_meridional.flatten(), D_meridional[0, 0])  # all meridional distances should be the same in either mesh
    if mesh == 'flat':
        assert np.allclose(D_zonal.flatten(), D_zonal[0, 0])  # all zonal distances should be the same in flat mesh
    else:
        assert all((np.gradient(D_zonal, axis=0) < 0).flatten())  # zonal distances should decrease in spherical mesh


@pytest.mark.parametrize('dx, dy', [('e1u', 'e2u'), ('e1v', 'e2v')])
def test_fieldset_celledgesizes_curvilinear(dx, dy):
    fname = path.join(path.dirname(__file__), 'test_data', 'mask_nemo_cross_180lon.nc')
    filenames = {'dx': fname, 'dy': fname, 'mesh_mask': fname}
    variables = {'dx': dx, 'dy': dy}
    dimensions = {'dx': {'lon': 'glamu', 'lat': 'gphiu'},
                  'dy': {'lon': 'glamu', 'lat': 'gphiu'}}
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

    # explicitly setting cell_edge_sizes from e1u and e2u etc
    fieldset.dx.grid.cell_edge_sizes['x'] = fieldset.dx.data
    fieldset.dx.grid.cell_edge_sizes['y'] = fieldset.dy.data

    A = fieldset.dx.cell_areas()
    assert np.allclose(A, fieldset.dx.data * fieldset.dy.data)


@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_fieldset_cellareas(mesh):
    data, dimensions = generate_fieldset(10, 7)
    fieldset = FieldSet.from_data(data, dimensions, mesh=mesh)
    cell_areas = fieldset.V.cell_areas()
    if mesh == 'flat':
        assert np.allclose(cell_areas.flatten(), cell_areas[0, 0], rtol=1e-3)
    else:
        assert all((np.gradient(cell_areas, axis=0) < 0).flatten())  # areas should decrease with latitude in spherical mesh
        for y in range(cell_areas.shape[0]):
            assert np.allclose(cell_areas[y, :], cell_areas[y, 0], rtol=1e-3)


@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_fieldset_gradient(mesh):
    data, dimensions = generate_fieldset(5, 3)
    fieldset = FieldSet.from_data(data, dimensions, mesh=mesh)

    # Calculate field gradients for testing against numpy gradients.
    dFdx, dFdy = fieldset.V.gradient()

    # Create numpy fields.
    conv_factor = 6.371e6 * np.pi / 180. if mesh == 'spherical' else 1.
    np_dFdx = np.gradient(fieldset.V.data[0, :, :], (np.diff(fieldset.V.lon) * conv_factor)[0], axis=1)
    np_dFdy = np.gradient(fieldset.V.data[0, :, :], (np.diff(fieldset.V.lat) * conv_factor)[0], axis=0)
    if mesh == 'spherical':
        for y in range(np_dFdx.shape[0]):
            np_dFdx[:, y] /= math.cos(fieldset.V.grid.lat[y] * math.pi / 180.)

    assert np.allclose(dFdx.data, np_dFdx, rtol=5e-2)  # Field gradient dx.
    assert np.allclose(dFdy.data, np_dFdy, rtol=5e-2)  # Field gradient dy.


def addConst(particle, fieldset, time, dt):
    particle.lon = particle.lon + fieldset.movewest + fieldset.moveeast


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_constant(mode):
    data, dimensions = generate_fieldset(100, 100)
    fieldset = FieldSet.from_data(data, dimensions)
    westval = -0.2
    eastval = 0.3
    fieldset.add_constant('movewest', westval)
    fieldset.add_constant('moveeast', eastval)
    assert fieldset.movewest == westval

    pset = ParticleSet.from_line(fieldset, size=1, pclass=ptype[mode],
                                 start=(0.5, 0.5), finish=(0.5, 0.5))
    pset.execute(pset.Kernel(addConst), dt=1, runtime=1)
    assert abs(pset[0].lon - (0.5 + westval + eastval)) < 1e-4


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('swapUV', [False, True])
def test_vector_fields(mode, swapUV):
    lon = np.linspace(0., 10., 12, dtype=np.float32)
    lat = np.linspace(0., 10., 10, dtype=np.float32)
    U = np.ones((10, 12), dtype=np.float32)
    V = np.zeros((10, 12), dtype=np.float32)
    data = {'U': U, 'V': V}
    dimensions = {'U': {'lat': lat, 'lon': lon},
                  'V': {'lat': lat, 'lon': lon}}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    if swapUV:  # we test that we can freely edit whatever UV field
        UV = VectorField('UV', fieldset.V, fieldset.U)
        fieldset.add_vector_field(UV)

    pset = ParticleSet.from_line(fieldset, size=1, pclass=ptype[mode],
                                 start=(0.5, 0.5), finish=(0.5, 0.5))
    pset.execute(AdvectionRK4, dt=1, runtime=1)
    if swapUV:
        assert pset[0].lon == .5
        assert pset[0].lat == 1.5
    else:
        assert pset[0].lon == 1.5
        assert pset[0].lat == .5


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('time_periodic', [True, False])
@pytest.mark.parametrize('dt_sign', [-1, 1])
def test_periodic(mode, time_periodic, dt_sign):
    lon = np.array([0, 1], dtype=np.float32)
    lat = np.array([0, 1], dtype=np.float32)
    depth = np.array([0, 1], dtype=np.float32)
    tsize = 24*60+1
    period = 86400
    time = np.linspace(0, period, tsize, dtype=np.float64)

    def temp_func(time):
        return 20 + 2 * np.sin(time*2*np.pi/period)
    temp_vec = temp_func(time)

    U = np.zeros((2, 2, 2, tsize), dtype=np.float32)
    V = np.zeros((2, 2, 2, tsize), dtype=np.float32)
    W = np.zeros((2, 2, 2, tsize), dtype=np.float32)
    temp = np.zeros((2, 2, 2, tsize), dtype=np.float32)
    temp[:, :, :, :] = temp_vec

    data = {'U': U, 'V': V, 'W': W, 'temp': temp}
    dimensions = {'lon': lon, 'lat': lat, 'depth': depth, 'time': time}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', time_periodic=time_periodic, transpose=True, allow_time_extrapolation=True)

    def sampleTemp(particle, fieldset, time, dt):
        # Note that fieldset.temp is interpolated at time=time+dt.
        # Indeed, sampleTemp is called at time=time, but the result is written
        # at time=time+dt, after the Kernel update
        particle.temp = fieldset.temp[time+dt, particle.lon, particle.lat, particle.depth]

    class MyParticle(ptype[mode]):
        temp = Variable('temp', dtype=np.float32, initial=20.)

    dt_sign = -1
    pset = ParticleSet.from_list(fieldset, pclass=MyParticle,
                                 lon=[0.5], lat=[0.5], depth=[0.5])
    pset.execute(AdvectionRK4_3D + pset.Kernel(sampleTemp),
                 runtime=delta(hours=51), dt=delta(hours=dt_sign*1))

    if time_periodic:
        t = pset.particles[0].time
        temp_theo = temp_func(t)
    elif dt_sign == 1:
        temp_theo = temp_vec[-1]
    elif dt_sign == -1:
        temp_theo = temp_vec[0]
    assert np.allclose(temp_theo, pset.particles[0].temp, atol=1e-5)


@pytest.mark.parametrize('fail', [False, pytest.mark.xfail(strict=True)(True)])
def test_fieldset_defer_loading_with_diff_time_origin(tmpdir, fail, filename='test_parcels_defer_loading'):
    filepath = tmpdir.join(filename)
    data0, dims0 = generate_fieldset(10, 10, 1, 10)
    dims0['time'] = np.arange(0, 10, 1) * 3600
    fieldset_out = FieldSet.from_data(data0, dims0)
    fieldset_out.U.grid.time_origin = np.datetime64('2018-04-20')
    data1, dims1 = generate_fieldset(10, 10, 1, 10)
    if fail:
        dims1['time'] = np.arange(0, 10, 1) * 3600
    else:
        dims1['time'] = np.arange(0, 10, 1) * 1800 + (24+25)*3600
    gridW = RectilinearZGrid(dims1['lon'], dims1['lat'], dims1['depth'], dims1['time'])
    fieldW = Field('W', np.zeros(data1['U'].shape), grid=gridW)
    if fail:
        fieldW.grid.time_origin = np.datetime64('2018-04-22')
    else:
        fieldW.grid.time_origin = np.datetime64('2018-04-18')
    fieldset_out.add_field(fieldW)
    fieldset_out.write(filepath)
    fieldset = FieldSet.from_parcels(filepath, extra_fields={'W': 'W'})
    pset = ParticleSet.from_list(fieldset, pclass=JITParticle, lon=[0.5], lat=[0.5], depth=[0.5],
                                 time=[datetime.datetime(2018, 4, 20, 1)])
    pset.execute(AdvectionRK4_3D, runtime=delta(hours=4), dt=delta(hours=1))
