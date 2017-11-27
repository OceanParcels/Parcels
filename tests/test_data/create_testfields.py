from parcels import FieldSet
import numpy as np


def generate_testfieldset(xdim, ydim, zdim, tdim):
    lon = np.linspace(0., 2., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    depth = np.linspace(0., 0.5, zdim, dtype=np.float32)
    time = np.linspace(0., tdim, tdim, dtype=np.float64)
    U = np.ones((xdim, ydim, zdim, tdim), dtype=np.float32)
    V = np.zeros((xdim, ydim, zdim, tdim), dtype=np.float32)
    P = 2.*np.ones((xdim, ydim, zdim, tdim), dtype=np.float32)
    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
    fieldset = FieldSet.from_data(data, dimensions)
    fieldset.write('testfields')


if __name__ == "__main__":
    generate_testfieldset(xdim=5, ydim=3, zdim=2, tdim=15)
