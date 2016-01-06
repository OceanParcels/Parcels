from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

def particleplotting(filename,tracerfile, mode):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile=Dataset(filename,'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    z = pfile.variables['z']

    fig = plt.figure()
    if tracerfile != 'none':
      tfile = Dataset(tracerfile,'r')
      X = tfile.variables['x']
      Y = tfile.variables['y']
      P = tfile.variables['P']
      plt.contourf(np.squeeze(X),np.squeeze(Y),np.squeeze(P))

    if mode == '3d':
      ax = fig.gca(projection='3d')
      for p in range(len(lon)):
        ax.plot(lon[p,:],lat[p,:],z[p,:],'.-')
      ax.set_xlabel('Longitude')
      ax.set_ylabel('Latitude')
      ax.set_zlabel('Depth')
    elif mode == '2d':
      plt.plot(np.transpose(lon),np.transpose(lat),'.-')
      plt.xlabel('Longitude')
      plt.ylabel('Latitude')      


    plt.show()

if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-f', '--tracerfile', type=str, default='none',
                   help='Name of tracer file to display underneath particle trajectories')
    args = p.parse_args()

    particleplotting(args.particlefile, args.tracerfile, mode=args.mode)
