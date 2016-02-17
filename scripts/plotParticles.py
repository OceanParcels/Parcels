#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation


def particleplotting(filename, tracerfile, mode, tracer_vars={}):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    z = pfile.variables['z']

    fig, ax = plt.subplots()
    if tracerfile != 'none':
        tfile = Dataset(tracerfile, 'r')
        X = tfile.variables[tracer_vars['X']]
        Y = tfile.variables[tracer_vars['Y']]
        V = tfile.variables[tracer_vars['V']]
        plt.contourf(np.squeeze(X[:]), np.squeeze(Y[:]), np.squeeze(V[0, 0, :, :]))

    if mode == '3d':
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

        line, = ax.plot(lon[:, 0], lat[:, 0], 'ow')
        if tracerfile == 'none':  # need to set ax limits
            plt.axis((np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)))

        def animate(i):
            line.set_xdata(lon[:, i])
            line.set_ydata(lat[:, i])
            return line,

        animation.FuncAnimation(fig, animate, np.arange(1, lon.shape[1]),
                                interval=100, blit=False)
        plt.show()

    plt.show()


if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d', 'movie2d'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-f', '--tracerfile', type=str, default='none',
                   help='Name of tracer file to display underneath particle trajectories')
    p.add_argument('-v', '--variables', type=str, nargs=3, default=['x', 'y', 'vozocrtx'],
                   help='Name of tracer file to display underneath particle trajectories')
    args = p.parse_args()

    variables = {'X': args.variables[0],
                 'Y': args.variables[1],
                 'V': args.variables[2]}

    particleplotting(args.particlefile, args.tracerfile, mode=args.mode, tracer_vars=variables)
