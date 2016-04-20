#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation


def particleplotting(filename, psize, recordedvar, backgroundfield, cmap, drawland, mode):
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
        bfile = Dataset(backgroundfield.values()[0], 'r')
        bX = bfile.variables['x']
        bY = bfile.variables['y']
        bT = bfile.variables['time_counter']
        # Find the variable that exists across at least two spatial and one time dimension
        if backgroundfield.keys()[0] is 'none':
            def checkShape(var):
                if len(np.shape(var)) > 3:
                    return True

            for v in bfile.variables:
                if checkShape(bfile.variables[v]):
                    bVar = bfile.variables[v]
        else:
            bVar = bfile.variables[backgroundfield.keys()[0]]
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
            plt.contourf(bX[:], bY[:], bVar[0, 0, :, :], zorder=-1, vmin=0, vmax=bMax,
                         levels=np.linspace(0, bMax, 10), xlim=[np.amin(lon), np.amax(lon)],
                         ylim=[np.amin(lat), np.amax(lat)], cmap=cmap)
            plt.plot(np.transpose(lon), np.transpose(lat), '.-', linewidth=psize,
                     markersize=psize, c='black')
        else:
            plt.plot(np.transpose(lon), np.transpose(lat), '.-', linewidth=psize,
                     markersize=psize)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        if drawland:
            m = Basemap(width=12000000, height=9000000, projection='cyl',
                        resolution='c', llcrnrlon=np.amin(lon), llcrnrlat=np.amin(lat),
                        urcrnrlon=np.amax(lon), urcrnrlat=np.amax(lat))
            m.drawcoastlines()
            m.fillcontinents(color='forestgreen', lake_color='aqua')

    elif mode == 'movie2d':

        fig = plt.figure(1)
        print((np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)))
        ax = plt.axes(xlim=[np.amin(lon), np.amax(lon)], ylim=[np.amin(lat), np.amax(lat)])
        ax.set_xlim([np.amin(lon), np.amax(lon)])
        ax.set_ylim([np.amin(lat), np.amax(lat)])
        scat = ax.scatter(lon[:, 0], lat[:, 0], s=psize, c='black')

        def animate(i):
            ax.cla()
            if drawland:
                m.drawcoastlines()
                m.fillcontinents(color='forestgreen', lake_color='aqua')
            if recordedvar is not 'none':
                scat = ax.scatter(lon[:, i], lat[:, i], s=psize, c=record[:, i]/rMax, cmap=cmap)
            else:
                scat = ax.scatter(lon[:, i], lat[:, i], s=psize, c='white', edgecolors='black')
            ax.set_xlim([np.amin(lon), np.amax(lon)])
            ax.set_ylim([np.amin(lat), np.amax(lat)])
            if backgroundfield is not 'none':
                field_time = np.argmax(bT > time[0, i]) - 1
                plt.contourf(bX[:], bY[:], bVar[field_time, 0, :, :], zorder=-1, vmin=0, vmax=bMax,
                             levels=np.linspace(0, bMax, 10), xlim=[np.amin(lon), np.amax(lon)],
                             ylim=[np.amin(lat), np.amax(lat)], cmap=cmap)
            return scat,

        if drawland:
            m = Basemap(width=12000000, height=9000000, projection='cyl',
                        resolution='c', llcrnrlon=np.round(np.amin(lon)), llcrnrlat=np.amin(lat),
                        urcrnrlon=np.amax(lon), urcrnrlat=np.amax(lat))

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
                   help='Name of file containing background field to display')
    p.add_argument('-v', '--variable', type=str, default='none',
                   help='Name of variable to display in background field')
    p.add_argument('-c', '--colourmap', type=str, default='jet',
                   help='Colourmap for field data')
    p.add_argument('-s', '--size', type=int, default='none',
                   help='Size of drawn particles and tracks')
    p.add_argument('-l', '--landdraw', type=bool, default=False,
                   help='Boolean for whether to draw land using mpl.basemap package')
    args = p.parse_args()

    if args.background is not 'none':
        args.background = {args.variable: args.background}

    if args.size is 'none':
        if args.mode is 'movie2d':
            args.size = 60
        else:
            args.size = 1

    particleplotting(args.particlefile, args.size, args.recordedvar, args.background, args.colourmap, args.landdraw,
                     mode=args.mode)
