from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4_3D
from parcels.field import Field
from datetime import timedelta as delta
import numpy as np
import pytest
from os import path, pardir


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def generate_fieldset(xdim, ydim, zdim=1, tdim=1):
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.zeros(zdim, dtype=np.float32)
    time = np.zeros(tdim, dtype=np.float64)
    U, V = np.meshgrid(lon, lat)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
    return (data, dimensions)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_fieldset_from_data(xdim, ydim):
    """ Simple test for fieldset initialisation from data. """
    data, dimensions = generate_fieldset(xdim, ydim)
    fieldset = FieldSet.from_data(data, dimensions)
    u_t = np.transpose(data['U']).reshape((dimensions['lat'].size, dimensions['lon'].size))
    v_t = np.transpose(data['V']).reshape((dimensions['lat'].size, dimensions['lon'].size))
    assert len(fieldset.U.data.shape) == 3
    assert len(fieldset.V.data.shape) == 3
    assert np.allclose(fieldset.U.data[0, :], u_t, rtol=1e-12)
    assert np.allclose(fieldset.V.data[0, :], v_t, rtol=1e-12)


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
    P = 2 * np.ones((xdim/2, ydim/2, zdim, tdim), dtype=np.float32)
    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'U': {'lat': lat, 'lon': lon},
                  'V': {'lat': lat, 'lon': lon},
                  'P': {'lat': lat[0::2], 'lon': lon[0::2], 'depth': depth, 'time': time}}

    fieldset = FieldSet.from_data(data, dimensions)
    assert len(fieldset.U.data.shape) == 3
    assert len(fieldset.V.data.shape) == 3
    assert len(fieldset.P.data.shape) == 4
    assert fieldset.P.data.shape == (tdim, zdim, ydim/2, xdim/2)
    assert np.allclose(fieldset.U.data, 0., rtol=1e-12)
    assert np.allclose(fieldset.V.data, 1., rtol=1e-12)
    assert np.allclose(fieldset.P.data, 2., rtol=1e-12)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_fieldset_from_nemo(xdim, ydim, tmpdir, filename='test_nemo'):
    """ Simple test for fieldset initialisation from NEMO file format. """
    filepath = tmpdir.join(filename)
    data, dimensions = generate_fieldset(xdim, ydim)
    fieldset_out = FieldSet.from_data(data, dimensions)
    fieldset_out.write(filepath)
    fieldset = FieldSet.from_nemo(filepath)
    u_t = np.transpose(data['U']).reshape((dimensions['lat'].size, dimensions['lon'].size))
    v_t = np.transpose(data['V']).reshape((dimensions['lat'].size, dimensions['lon'].size))
    assert len(fieldset.U.data.shape) == 3  # Will be 4 once we use depth
    assert len(fieldset.V.data.shape) == 3
    assert np.allclose(fieldset.U.data[0, :], u_t, rtol=1e-12)
    assert np.allclose(fieldset.V.data[0, :], v_t, rtol=1e-12)


@pytest.mark.parametrize('indslon', [range(10, 20), [1]])
@pytest.mark.parametrize('indslat', [range(30, 60), [22]])
def test_fieldset_from_file_subsets(indslon, indslat, tmpdir, filename='test_subsets'):
    """ Test for subsetting fieldset from file using indices dict. """
    data, dimensions = generate_fieldset(100, 100)
    filepath = tmpdir.join(filename)
    fieldsetfull = FieldSet.from_data(data, dimensions)
    fieldsetfull.write(filepath)
    indices = {'lon': indslon, 'lat': indslat}
    fieldsetsub = FieldSet.from_nemo(filepath, indices=indices)
    assert np.allclose(fieldsetsub.U.lon, fieldsetfull.U.lon[indices['lon']])
    assert np.allclose(fieldsetsub.U.lat, fieldsetfull.U.lat[indices['lat']])
    assert np.allclose(fieldsetsub.V.lon, fieldsetfull.V.lon[indices['lon']])
    assert np.allclose(fieldsetsub.V.lat, fieldsetfull.V.lat[indices['lat']])

    ixgrid = np.ix_([0], indices['lat'], indices['lon'])
    assert np.allclose(fieldsetsub.U.data, fieldsetfull.U.data[ixgrid])
    assert np.allclose(fieldsetsub.V.data, fieldsetfull.V.data[ixgrid])


@pytest.mark.parametrize('indstime', [range(2, 8), [4]])
def test_moving_eddies_file_subsettime(indstime):
    fieldsetfile = path.join(path.dirname(__file__), pardir, 'examples', 'MovingEddies_data', 'moving_eddies')
    fieldsetfull = FieldSet.from_nemo(fieldsetfile, extra_fields={'P': 'P'})
    fieldsetsub = FieldSet.from_nemo(fieldsetfile, extra_fields={'P': 'P'}, indices={'time': indstime})
    assert np.allclose(fieldsetsub.P.time, fieldsetfull.P.time[indstime])
    assert np.allclose(fieldsetsub.P.data, fieldsetfull.P.data[indstime, :, :])


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_add_field(xdim, ydim, tmpdir, filename='test_add'):
    filepath = tmpdir.join(filename)
    data, dimensions = generate_fieldset(xdim, ydim)
    fieldset = FieldSet.from_data(data, dimensions)
    field = Field('newfld', fieldset.U.data, fieldset.U.lon, fieldset.U.lat)
    fieldset.add_field(field)
    assert fieldset.newfld.data.shape == fieldset.U.data.shape
    fieldset.write(filepath)


def create_simple_fieldset(x, y, time):
    field = np.zeros((time.size, x, y), dtype=np.float32)
    ltri = np.triu_indices(n=x, m=y)
    for t in time:
        temp = np.zeros((x, y), dtype=np.float32)
        temp[ltri] = 1
        field[t, :, :] = np.reshape(temp.T, np.shape(field[t, :, :]))
    return field


def test_fieldset_gradient():
    x = 4
    y = 6
    time = np.linspace(0, 2, 3, dtype=np.int)
    field = Field("Test", data=create_simple_fieldset(x, y, time), time=time,
                  lon=np.linspace(0, x-1, x, dtype=np.float32),
                  lat=np.linspace(-y/2, y/2-1, y, dtype=np.float32))

    # Calculate field gradients for testing against numpy gradients.
    grad_fields = field.gradient()

    # Create numpy fields.
    r = 6.371e6
    deg2rd = np.pi / 180.
    numpy_grad_fields = np.gradient(np.transpose(field.data[0, :, :]), (r * np.diff(field.lat) * deg2rd)[0])

    # Arbitrarily set relative tolerance to 1%.
    assert np.allclose(grad_fields[0].data[0, :, :], np.array(np.transpose(numpy_grad_fields[0])),
                       rtol=1e-2)  # Field gradient dx.
    assert np.allclose(grad_fields[1].data[0, :, :], np.array(np.transpose(numpy_grad_fields[1])),
                       rtol=1e-2)  # Field gradient dy.


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
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', time_periodic=time_periodic)

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
