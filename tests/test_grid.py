from parcels import Grid, ParticleSet, ScipyParticle, JITParticle
from parcels.field import Field
import numpy as np
import pytest
from os import path


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def generate_grid(xdim, ydim, zdim=1, tdim=1):
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.zeros(zdim, dtype=np.float32)
    time = np.zeros(tdim, dtype=np.float64)
    U, V = np.meshgrid(lon, lat)
    return (np.array(U, dtype=np.float32),
            np.array(V, dtype=np.float32),
            lon, lat, depth, time)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_grid_from_data(xdim, ydim):
    """ Simple test for grid initialisation from data. """
    u, v, lon, lat, depth, time = generate_grid(xdim, ydim)
    grid = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    u_t = np.transpose(u).reshape((lat.size, lon.size))
    v_t = np.transpose(v).reshape((lat.size, lon.size))
    assert len(grid.U.data.shape) == 3  # Will be 4 once we use depth
    assert len(grid.V.data.shape) == 3
    assert np.allclose(grid.U.data[0, :], u_t, rtol=1e-12)
    assert np.allclose(grid.V.data[0, :], v_t, rtol=1e-12)


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_grid_from_nemo(xdim, ydim, tmpdir, filename='test_nemo'):
    """ Simple test for grid initialisation from NEMO file format. """
    filepath = tmpdir.join(filename)
    u, v, lon, lat, depth, time = generate_grid(xdim, ydim)
    grid_out = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    grid_out.write(filepath)
    grid = Grid.from_nemo(filepath)
    u_t = np.transpose(u).reshape((lat.size, lon.size))
    v_t = np.transpose(v).reshape((lat.size, lon.size))
    assert len(grid.U.data.shape) == 3  # Will be 4 once we use depth
    assert len(grid.V.data.shape) == 3
    assert np.allclose(grid.U.data[0, :], u_t, rtol=1e-12)
    assert np.allclose(grid.V.data[0, :], v_t, rtol=1e-12)


@pytest.mark.parametrize('indslon', [range(10, 20), [1]])
@pytest.mark.parametrize('indslat', [range(30, 60), [22]])
def test_grid_from_file_subsets(indslon, indslat, tmpdir, filename='test_subsets'):
    """ Test for subsetting grid from file using indices dict. """
    u, v, lon, lat, depth, time = generate_grid(100, 100)
    filepath = tmpdir.join(filename)
    gridfull = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    gridfull.write(filepath)
    indices = {'lon': indslon, 'lat': indslat}
    gridsub = Grid.from_nemo(filepath, indices=indices)
    assert np.allclose(gridsub.U.lon, gridfull.U.lon[indices['lon']])
    assert np.allclose(gridsub.U.lat, gridfull.U.lat[indices['lat']])
    assert np.allclose(gridsub.V.lon, gridfull.V.lon[indices['lon']])
    assert np.allclose(gridsub.V.lat, gridfull.V.lat[indices['lat']])

    ixgrid = np.ix_([0], indices['lat'], indices['lon'])
    assert np.allclose(gridsub.U.data, gridfull.U.data[ixgrid])
    assert np.allclose(gridsub.V.data, gridfull.V.data[ixgrid])


@pytest.mark.parametrize('indstime', [range(10, 20), [4]])
def test_moving_eddies_file_subsettime(indstime):
    gridfile = path.join(path.dirname(__file__), '..', 'examples', 'MovingEddies_data', 'moving_eddies')
    gridfull = Grid.from_nemo(gridfile, extra_vars={'P': 'P'})
    gridsub = Grid.from_nemo(gridfile, extra_vars={'P': 'P'}, indices={'time': indstime})
    assert np.allclose(gridsub.P.time, gridfull.P.time[indstime])
    assert np.allclose(gridsub.P.data, gridfull.P.data[indstime, :, :])


@pytest.mark.parametrize('xdim', [100, 200])
@pytest.mark.parametrize('ydim', [100, 200])
def test_add_field(xdim, ydim, tmpdir, filename='test_add'):
    filepath = tmpdir.join(filename)
    u, v, lon, lat, depth, time = generate_grid(xdim, ydim)
    grid = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    field = Field('newfld', grid.U.data, grid.U.lon, grid.U.lat)
    grid.add_field(field)
    assert grid.newfld.data.shape == grid.U.data.shape
    grid.write(filepath)


def createSimpleGrid(x, y, time):
    field = np.zeros((time.size, x, y), dtype=np.float32)
    ltri = np.triu_indices(n=x, m=y)
    for t in time:
        temp = np.zeros((x, y), dtype=np.float32)
        temp[ltri] = 1
        field[t, :, :] = np.reshape(temp.T, np.shape(field[t, :, :]))
    return field


def test_grid_gradient():
    x = 4
    y = 6
    time = np.linspace(0, 2, 3, dtype=np.int)
    field = Field("Test", data=createSimpleGrid(x, y, time), time=time,
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


def addConst(particle, grid, time, dt):
    particle.lon = particle.lon + grid.movewest + grid.moveeast


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_grid_constant(mode):
    u, v, lon, lat, depth, time = generate_grid(100, 100)
    grid = Grid.from_data(u, lon, lat, v, lon, lat, depth, time)
    westval = -0.2
    eastval = 0.3
    grid.add_constant('movewest', westval)
    grid.add_constant('moveeast', eastval)
    assert grid.movewest == westval

    pset = ParticleSet.from_line(grid, size=1, pclass=ptype[mode],
                                 start=(0.5, 0.5), finish=(0.5, 0.5))
    pset.execute(pset.Kernel(addConst), dt=1, runtime=1)
    assert abs(pset[0].lon - (0.5 + westval + eastval)) < 1e-4
