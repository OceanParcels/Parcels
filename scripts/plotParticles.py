#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
from argparse import ArgumentParser
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import rc
except:
    plt = None


def plotTrajectoriesFile(filename, mode='2d', tracerfile=None, tracerfield='P',
                         tracerlon='x', tracerlat='y', recordedvar=None):
    """Quick and simple plotting of Parcels trajectories

    :param filename: Name of Parcels-generated NetCDF file with particle positions
    :param mode: Type of plot to show. Supported are '2d', '3d'
                'movie2d' and 'movie2d_notebook'. The latter two give animations,
                with 'movie2d_notebook' specifically designed for jupyter notebooks
    :param tracerfile: Name of NetCDF file to show as background
    :param tracerfield: Name of variable to show as background
    :param tracerlon: Name of longitude dimension of variable to show as background
    :param tracerlat: Name of latitude dimension of variable to show as background
    :param recordedvar: Name of variable used to color particles in scatter-plot.
                Only works in 'movie2d' or 'movie2d_notebook' mode.
    """

    if plt is None:
        print("Visualisation is not possible. Matplotlib not found.")
        return

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    z = pfile.variables['z']
    if len(lon.shape) == 1:
        type = 'indexed'
        id = pfile.variables['trajectory'][:]
        time = pfile.variables['time'][:]
    else:
        type = 'array'

    if(recordedvar is not None):
        record = pfile.variables[recordedvar]

    if tracerfile is not None:
        tfile = Dataset(tracerfile, 'r')
        X = tfile.variables[tracerlon]
        Y = tfile.variables[tracerlat]
        P = tfile.variables[tracerfield]
        plt.contourf(np.squeeze(X), np.squeeze(Y), np.squeeze(P))

    if mode == '3d':
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        if type == 'array':
            for p in range(len(lon)):
                ax.plot(lon[p, :], lat[p, :], z[p, :], '.-')
        elif type == 'indexed':
            for t in range(max(id)+1):
                ax.plot(lon[id == t], lat[id == t],
                        z[id == t], '.-')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth')
    elif mode == '2d':
        if type == 'array':
            plt.plot(np.transpose(lon), np.transpose(lat), '.-')
        elif type == 'indexed':
            for t in range(max(id)+1):
                plt.plot(lon[id == t], lat[id == t], '.-')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    elif mode == 'movie2d' or 'movie2d_notebook':

        fig = plt.figure()
        ax = plt.axes(xlim=(np.amin(lon), np.amax(lon)), ylim=(np.amin(lat), np.amax(lat)))
        if type == 'array':
            scat = ax.scatter(lon[:, 0], lat[:, 0], s=60, cmap=plt.get_cmap('autumn'))  # cmaps not working?
            frames = np.arange(1, lon.shape[1])
        elif type == 'indexed':
            mintime = min(time)
            scat = ax.scatter(lon[time == mintime], lat[time == mintime],
                              s=60, cmap=plt.get_cmap('autumn'))
            frames = np.unique(time)

        def animate(t):
            if type == 'array':
                scat.set_offsets(np.matrix((lon[:, t], lat[:, t])).transpose())
            elif type == 'indexed':
                scat.set_offsets(np.matrix((lon[time == t], lat[time == t])).transpose())
            if recordedvar is not None:
                scat.set_array(record[:, t])
            return scat,

        rc('animation', html='html5')
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                       interval=100, blit=False)
    if mode == 'movie2d_notebook':
        plt.close()
        return anim
    else:
        plt.show()


if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d', 'movie2d', 'movie2d_notebook'), nargs='?',
                   default='movie2d', help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-f', '--tracerfile', type=str, default=None,
                   help='Name of tracer file to display underneath particle trajectories')
    p.add_argument('-flon', '--tracerfilelon', type=str, default='x',
                   help='Name of longitude dimension in tracer file')
    p.add_argument('-flat', '--tracerfilelat', type=str, default='y',
                   help='Name of latitude dimension in tracer file')
    p.add_argument('-ffld', '--tracerfilefield', type=str, default='P',
                   help='Name of field in tracer file')
    p.add_argument('-r', '--recordedvar', type=str, default=None,
                   help='Name of a variable recorded along trajectory')
    args = p.parse_args()

    plotTrajectoriesFile(args.particlefile, mode=args.mode, tracerfile=args.tracerfile,
                         tracerfield=args.tracerfilefield, tracerlon=args.tracerfilelon,
                         tracerlat=args.tracerfilelat, recordedvar=args.recordedvar)
