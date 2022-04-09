#%matplotlib inline
from parcels import AdvectionRK4
from parcels import FieldSet
from parcels import Field
from parcels import JITParticle
from parcels import ParticleSet
from parcels import plotTrajectoriesFile
from parcels import ErrorCode
from parcels import ParcelsRandom

import numpy as np
import math
from datetime import timedelta as delta


# Add diffusivity
def smagorinsky(particle, fieldset, time):
    dx = 0.01;
    dudx = (fieldset.U[time, particle.depth, particle.lat, particle.lon+dx]-fieldset.U[time, particle.depth, particle.lat, particle.lon-dx]) / (2*dx)
    dudy = (fieldset.U[time, particle.depth, particle.lat+dx, particle.lon]-fieldset.U[time, particle.depth, particle.lat-dx, particle.lon]) / (2*dx)
    dvdx = (fieldset.V[time, particle.depth, particle.lat, particle.lon+dx]-fieldset.V[time, particle.depth, particle.lat, particle.lon-dx]) / (2*dx)
    dvdy = (fieldset.V[time, particle.depth, particle.lat+dx, particle.lon]-fieldset.V[time, particle.depth, particle.lat-dx, particle.lon]) / (2*dx)

    A = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
    deg_to_m = (1852*60)**2*math.cos(particle.lat*math.pi/180)
    A = A / deg_to_m
    Vh = fieldset.Cs * A * math.sqrt(dudx**2 + 0.5*(dudy + dvdx)**2 + dvdy**2)

    xres = 1# [degrees] 
    yres = 1

    dlat = yres * ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Vh) 
    dlon = xres * ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Vh) 

    particle.lat += dlat
    particle.lon += dlon


filenames = {'U': "GlobCurrent_example_data/20*.nc", 'V': "GlobCurrent_example_data/20*.nc"}
variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)

x = fieldset.U.grid.lon
y = fieldset.U.grid.lat
    
cell_areas = Field(name='cell_areas', data=fieldset.U.cell_areas(), lon=x, lat=y)
fieldset.add_field(cell_areas)

fieldset.add_constant('Cs', 0.1)

lon = 29
lat = -33
repeatdt = delta(hours=12)
pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lon, lat=lat,  repeatdt=repeatdt)

def DeleteParticle(particle, fieldset, time):
    particle.delete()

kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(smagorinsky)
output_file = pset.ParticleFile(name="Global_smagdiff.nc", outputdt=delta(hours=6))

pset.execute(kernels, runtime=delta(days=5), dt=delta(minutes=5), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
pset.show(field=fieldset.U)

pset.repeatdt = None
pset.execute(kernels, runtime=delta(days=25), dt=delta(minutes=5), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
pset.show(field=fieldset.U)

output_file.export()
plotTrajectoriesFile('Global_smagdiff.nc', tracerfile='GlobCurrent_example_data/20020120000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc', tracerlon='lon', tracerlat='lat', tracerfield='eastward_eulerian_current_velocity')
anim = plotTrajectoriesFile('Global_smagdiff.nc', tracerfile='GlobCurrent_example_data/20020120000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc', tracerlon='lon', tracerlat='lat', tracerfield='eastward_eulerian_current_velocity', mode='movie2d_notebook')
anim.save('Global_smagdiff.gif')



