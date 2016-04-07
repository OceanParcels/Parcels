#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation


def particleplotting(filename, recordedvar, backgroundfield, mode):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    time = pfile.variables['time']
    z = pfile.variables['z']

    if(recordedvar is not 'none'):
        rMax = np.max(pfile.variables[recordedvar])
        record = pfile.variables[recordedvar]

    if backgroundfield is not 'none':
        bfile = Dataset(backgroundfield, 'r')
        bX = bfile.variables['x']
        bY = bfile.variables['y']
        bT = bfile.variables['time_counter']
        # Find the variable that exists across at least two spatial and one time dimension

        def checkShape(var):
            if len(np.shape(var)) > 3:
                return True
        for v in bfile.variables:
            if checkShape(bfile.variables[v]):
                bVar = bfile.variables[v]
        bMax = np.max(bVar)

    if mode == '3d':
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        for p in range(len(lon)):
            ax.plot(lon[p, :], lat[p, :], z[p, :], '.-')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth')
    elif mode == '2d':
        if backgroundfield is not 'none':
            plt.contourf(np.squeeze(bX), np.squeeze(bY), np.squeeze(bVar), zorder=-1, vmin=0, vmax=bMax)
        plt.plot(np.transpose(lon), np.transpose(lat), '.-')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    elif mode == 'movie2d':

        fig = plt.figure(1)
        ax = plt.axes(xlim=(np.amin(lon), np.amax(lon)), ylim=(np.amin(lat), np.amax(lat)))
        ax.scatter(lon[:, 0], lat[:, 0], s=60, c='black')

        def animate(i):
            ax.cla()
            if recordedvar is not 'none':
                scat = ax.scatter(lon[:, i], lat[:, i], s=60, c=record[:, i]/rMax)
            else:
                scat = ax.scatter(lon[:, i], lat[:, i], s=60, c='white', edgecolors='black')
            if backgroundfield is not 'none':
                field_time = np.argmax(bT > time[0, i]) - 1
                plt.contourf(bX[:], bY[:], bVar[field_time, 0, :, :], zorder=-1, vmin=0, vmax=bMax,
                             levels=np.linspace(0, bMax, 10))
            return scat,

        anim = animation.FuncAnimation(fig, animate, frames=np.arange(1, lon.shape[1]),
                                       interval=100, blit=False)

    plt.show()

if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d', 'movie2d'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-r', '--recordedvar', type=str, default='none',
                   help='Name of a variable recorded along trajectory')
    p.add_argument('-b', '--background', type=str, default='none',
                   help='Name of background field to display')
    args = p.parse_args()

    particleplotting(args.particlefile, args.recordedvar, args.background, mode=args.mode)
