#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation


def particleplotting(filename, tracerfile='none', tracerlon='x', tracerlat='y',
                     tracerfield='P', recordedvar='none', mode='2d'):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    z = pfile.variables['z']

    if(recordedvar is not 'none'):
        record = pfile.variables[recordedvar]

    if tracerfile != 'none':
        tfile = Dataset(tracerfile, 'r')
        X = tfile.variables[tracerlon]
        Y = tfile.variables[tracerlat]
        P = tfile.variables[tracerfield]
        plt.contourf(np.squeeze(X), np.squeeze(Y), np.squeeze(P))

    if mode == '3d':
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        for p in range(len(lon)):
            ax.plot(lon[p, :], lat[p, :], z[p, :], '.-')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth')
    elif mode == '2d':
        plt.plot(np.transpose(lon), np.transpose(lat), '.-')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    elif mode == 'movie2d':

        fig = plt.figure(1)
        ax = plt.axes(xlim=(np.amin(lon), np.amax(lon)), ylim=(np.amin(lat), np.amax(lat)))
        scat = ax.scatter(lon[:, 0], lat[:, 0], s=60, cmap=plt.get_cmap('autumn'))  # cmaps not working?

        def animate(i):
            scat.set_offsets(np.matrix((lon[:, i], lat[:, i])).transpose())
            if recordedvar is not 'none':
                scat.set_array(record[:, i])
            return scat,

        animation.FuncAnimation(fig, animate, frames=np.arange(1, lon.shape[1]),
                                interval=100, blit=False)

    plt.show()

if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d', 'movie2d'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-f', '--tracerfile', type=str, default='none',
                   help='Name of tracer file to display underneath particle trajectories')
    p.add_argument('-flon', '--tracerfilelon', type=str, default='x',
                   help='Name of longitude dimension in tracer file')
    p.add_argument('-flat', '--tracerfilelat', type=str, default='y',
                   help='Name of latitude dimension in tracer file')
    p.add_argument('-ffld', '--tracerfilefield', type=str, default='P',
                   help='Name of field in tracer file')
    p.add_argument('-r', '--recordedvar', type=str, default='none',
                   help='Name of a variable recorded along trajectory')
    args = p.parse_args()

    particleplotting(args.particlefile, args.tracerfile, args.tracerfilelon,
                     args.tracerfilelat, args.tracerfilefield, args.recordedvar, mode=args.mode)
