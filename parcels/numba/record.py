import numpy as np


def create_record_pset(n_particle):
    dtype = [
        ('dt', 'f8'),
        ('lat', 'f8'),
        ('lon', 'f8'),
        ('depth', 'f8'),
        ('status', 'i4'),
        ('request_chunk', 'i4')
    ]
    pset = np.empty(n_particle, dtype=dtype).view(np.recarray)
    pset.lat = np.random.randn(n_particle)
    pset.lon = np.random.randn(n_particle)
    pset.dt = 0.01
    pset.depth = np.random.randn(n_particle)
    pset.status = 0
    pset.request_chunk = 0
    return pset
