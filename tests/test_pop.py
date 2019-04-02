
#import fieldset
from parcels import FieldSet, Field, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, AdvectionRK4_3D
from parcels import RectilinearZGrid, RectilinearSGrid, CurvilinearZGrid
import numpy as np
import xarray as xr
import math
import pytest
from os import path
import os
from datetime import timedelta as delta
import parcels

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
#mode = 'scipy'; vert_mode='slayer'; location = (11,1,1)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('vert_mode', ['normal','slayer'])
@pytest.mark.parametrize('location', [(3,3,3), (11,1,1)])
def test_popgrid(mode, vert_mode, location):
    data_path = os.getcwd()+'/test_data/'
    if('normal'==vert_mode):
        mesh = data_path + 'POPtestdata_normal.nc'
    if('slayer'==vert_mode):
        mesh = data_path + 'POPtestdata_slayer.nc'
    
    filenames = { 'U': {'lon': mesh,
                        'lat': mesh,
                        'depth': mesh,
                        'data':mesh},
                'V' : {'lon': mesh,
                        'lat': mesh,
                        'depth': mesh,
                        'data':mesh},
                'W' : {'lon': mesh,
                        'lat': mesh,
                        'depth': mesh,
                        'data':mesh}  ,
                'T' : {'lon': mesh,
                        'lat': mesh,
                        'depth': mesh,
                        'data':mesh} 
                }
    
    variables = {'U': 'U',
                 'V': 'V',
                 'W': 'W',
                 'T': 'T'}
    if('normal'==vert_mode):
        dimensions = {'U':{'lon': 'lon', 'lat': 'lat', 'depth': 'w_dep', 'time': 'time'},
                  'V': {'lon': 'lon', 'lat': 'lat', 'depth': 'w_dep', 'time': 'time'},
                    'W': {'lon': 'lon', 'lat': 'lat', 'depth': 'w_dep', 'time': 'time'},
                    'T': {'lon': 'lon', 'lat': 'lat', 'depth': 'w_dep', 'time': 'time'}  }  
    if('slayer'==vert_mode):
        dimensions = {'U':{'lon': 'lon', 'lat': 'lat', 'depth': 'w_deps', 'time': 'time'},
                  'V': {'lon': 'lon', 'lat': 'lat', 'depth': 'w_deps', 'time': 'time'},
                    'W': {'lon': 'lon', 'lat': 'lat', 'depth': 'w_deps', 'time': 'time'},
                    'T': {'lon': 'lon', 'lat': 'lat', 'depth': 'w_deps', 'time': 'time'} }  
     
    
    field_set = FieldSet.from_pop(filenames, variables, dimensions, mesh='flat')
    
    def sampleVel(particle, fieldset, time):
        particle.zonal =  fieldset.U[time, particle.depth, particle.lat, particle.lon]
        particle.vert =  fieldset.W[time, particle.depth, particle.lat, particle.lon]
        particle.tracer =  fieldset.T[time, particle.depth, particle.lat, particle.lon]
        particle.meridional =  fieldset.V[time, particle.depth, particle.lat, particle.lon]    
    
    class MyParticle(ptype[mode]):
        zonal = Variable('zonal', dtype=np.float32, initial=0.)
        meridional = Variable('meridional', dtype=np.float32, initial=0.)
        vert = Variable('vert', dtype=np.float32, initial=0.)
        tracer = Variable('tracer', dtype=np.float32, initial=0.)        
    
    pset = ParticleSet.from_list(field_set, MyParticle, lon=location[2], lat=location[1], depth=location[0])
    pset.execute(pset.Kernel(sampleVel), runtime=0, dt=0)
    
#    print abs(pset[0].zonal - 0.015) 
#    print abs(pset[0].meridional - 0.01)
#    print abs(pset[0].vert)# + 0.01) 
#    print abs(pset[0].tracer -1) 
    
    assert abs(pset[0].zonal - 0.015) < 1e-6
    assert abs(pset[0].meridional - 0.01) < 1e-6
    assert abs(pset[0].vert + 0.01) < 1e-6
    assert abs(pset[0].tracer -1) < 1e-6