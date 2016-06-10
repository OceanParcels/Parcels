#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation


def particleplotting(filename, psize, recordedvar, rcmap, backgroundfield, dimensions, cmap, limits, mode):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    time = pfile.variables['time']
    z = pfile.variables['z']

    if limits is -1:
        limits = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)]

    if recordedvar is not 'none':
        print('Particles coloured by: %s' % recordedvar)
        rMin = np.min(pfile.variables[recordedvar])
        rMax = np.max(pfile.variables[recordedvar])
        print('Min = %f, Max = %f' % (rMin, rMax))
        record = (pfile.variables[recordedvar]-rMin)/(rMax-rMin)

    if backgroundfield is not 'none':
        bfile = Dataset(backgroundfield.values()[0], 'r')
        bX = bfile.variables[dimensions[0]]
        bY = bfile.variables[dimensions[1]]
        bT = bfile.variables[dimensions[2]]
        # Find the variable that exists across at least two spatial and one time dimension
        if backgroundfield.keys()[0] is 'none':
            def checkShape(var):
                if len(np.shape(var)) > 3:
                    return True

            for v in bfile.variables:
                if checkShape(bfile.variables[v]):
                    bVar = bfile.variables[v]
                    print('Background variable is %s' % v)
                    break
        else:
            bVar = bfile.variables[backgroundfield.keys()[0]]

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
            plt.contourf(bX[:], bY[:], bVar[0, 0, :, :], zorder=-1, vmin=0, vmax=np.max(bVar[0, 0, :, :]),
                         levels=np.linspace(0, np.max(bVar[0, 0, :, :]), 100), xlim=[limits[0], limits[1]],
                         ylim=[limits[2], limits[3]], cmap=cmap)
            plt.plot(np.transpose(lon), np.transpose(lat), '.-', linewidth=psize,
                     markersize=psize, c='black')
            plt.xlim([limits[0], limits[1]])
            plt.ylim([limits[2], limits[3]])
        else:
            plt.plot(np.transpose(lon), np.transpose(lat), '.-', linewidth=psize,
                     markersize=psize)
            plt.xlim([limits[0], limits[1]])
            plt.ylim([limits[2], limits[3]])

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    elif mode == 'movie2d':

        fig = plt.figure(1)
        ax = plt.axes(xlim=[limits[0], limits[1]], ylim=[limits[2], limits[3]])
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        scat = ax.scatter(lon[:, 0], lat[:, 0], s=psize, c='black')

        def animate(i):
            ax.cla()
            if recordedvar is not 'none':
                scat = ax.scatter(lon[:, i], lat[:, i], s=psize, c=record[:, i], cmap=rcmap, vmin=0, vmax=1)
            else:
                scat = ax.scatter(lon[:, i], lat[:, i], s=psize, c='white', edgecolors='black')
            ax.set_xlim([limits[0], limits[1]])
            ax.set_ylim([limits[2], limits[3]])
            if backgroundfield is not 'none':
                field_time = np.argmax(bT > time[0, i]) - 1
                plt.contourf(bX[:], bY[:], bVar[field_time, 0, :, :], zorder=-1, vmin=0, vmax=np.max(bVar[field_time, 0, :, :]),
                             levels=np.linspace(0, np.max(bVar[field_time, 0, :, :]), 100), xlim=[limits[0], limits[1]],
                             ylim=[limits[2], limits[3]], cmap=cmap)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
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
                   help='Name of file containing background field to display')
    p.add_argument('-v', '--variable', type=str, default='none',
                   help='Name of variable to display in background field')
    p.add_argument('-c', '--colourmap', type=str, default='jet',
                   help='Colourmap for field data')
    p.add_argument('-cr', '--colourmap_recorded', type=str, default='autumn',
                   help='Colourmap for particle recorded data')
    p.add_argument('-s', '--size', type=str, default='none',
                   help='Size of drawn particles and tracks')
    p.add_argument('-l', '--limits', type=float, nargs=4, default=-1,
                   help='Limits for plotting, given min_lon, max_lon, min_lat, max_lat')
    p.add_argument('-d', '--dimensions', type=str, nargs=3, default=['nav_lon', 'nav_lat', 'time_counter'],
                   help='Name of background field dimensions in order of lon, lat, and time')
    args = p.parse_args()

    if args.background is not 'none':
        args.background = {args.variable: args.background}

    if args.size is 'none':
        if args.mode is 'movie2d':
            psize = 60
        else:
            psize = 1
    else:
        psize = int(args.size)

    particleplotting(args.particlefile, psize, args.recordedvar, args.colourmap_recorded, args.background, args.dimensions, args.colourmap,
                     args.limits, mode=args.mode)
