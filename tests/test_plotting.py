from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, ErrorCode
from parcels.field import Field, VectorField
from datetime import timedelta as delta
import numpy as np
import dask.array as da
#import pytest
from os import path
from matplotlib import pyplot as plt

def periodicBC(particle, fieldset, time):
    if particle.lon > 180:
        particle.lon -= 360

def test_field_from_netcdf():
    data_path = path.join(path.dirname(__file__), 'test_data/')

    filenames = { 'U': {'lon': data_path + 'mask_nemo_cross_180lon.nc',
                 'lat': data_path + 'mask_nemo_cross_180lon.nc',
                 'data': data_path + 'Uu_eastward_nemo_cross_180lon.nc'},
                  'V': {'lon': data_path + 'mask_nemo_cross_180lon.nc',
                 'lat': data_path + 'mask_nemo_cross_180lon.nc',
                 'data': data_path + 'Vv_eastward_nemo_cross_180lon.nc'}
                  }
    variables = {'U': 'U',
                 'V': 'V'}
    dimensions = {'lon': 'glamf', 'lat': 'gphif'}
    return FieldSet.from_netcdf(filenames, variables, dimensions, interp_method='cgrid_velocity', field_chunksize='auto', allow_time_extrapolation=True)

def test_pset_create_field(fieldset, npart=100):
    lonp = -180 * np.ones(npart)
    latp = [i for i in np.linspace(-70, 88, npart)]
    pset = ParticleSet.from_list(fieldset, JITParticle, lon=lonp, lat=latp)
    return pset

def DeleteParticle(particle, fieldset, time):
    particle.delete()

if __name__=='__main__':
    fset = test_field_from_netcdf()
    print(fset)
    pset = test_pset_create_field(fset)
    kernels = pset.Kernel(AdvectionRK4) + periodicBC
    pset.execute(kernels, dt=delta(hours=1), output_file=None,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
    pset.show(field=fset.U)
