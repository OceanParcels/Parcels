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
                         tracerlon='x', tracerlat='y', recordedvar=None, show_plt=True):
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
    :param show_plt: Boolean whether plot should directly be show (for py.test)
    """

    if plt is None:
        print("Visualisation is not possible. Matplotlib not found.")
        return

    pfile = Dataset(filename, 'r')
    lon = np.ma.filled(pfile.variables['lon'], np.nan)
    lat = np.ma.filled(pfile.variables['lat'], np.nan)
    time = np.ma.filled(pfile.variables['time'], np.nan)
    z = np.ma.filled(pfile.variables['z'], np.nan)

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
        for p in range(len(lon)):
            ax.plot(lon[p, :], lat[p, :], z[p, :], '.-')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth')
    elif mode == '2d':
        plt.plot(np.transpose(lon), np.transpose(lat), '.-')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    elif mode == 'movie2d' or 'movie2d_notebook':
        fig = plt.figure()
        ax = plt.axes(xlim=(np.nanmin(lon), np.nanmax(lon)), ylim=(np.nanmin(lat), np.nanmax(lat)))
        plottimes = np.unique(time)
        plottimes = plottimes[~np.isnan(plottimes)]
        b = time == plottimes[0]
        scat = ax.scatter(lon[b], lat[b], s=60, cmap=plt.get_cmap('autumn'))  # cmaps not working?
        ttl = ax.set_title('Particle at time ' + str(plottimes[0]))
        frames = np.arange(1, len(plottimes))

        def animate(t):
            b = time == plottimes[t]
            scat.set_offsets(np.matrix((lon[b], lat[b])).transpose())
            ttl.set_text('Particle at time ' + str(plottimes[t]))
            if recordedvar is not None:
                scat.set_array(record[b])
            return scat,

        rc('animation', html='html5')
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                       interval=100, blit=False)
    if mode == 'movie2d_notebook':
        plt.close()
        return anim
    else:
        if show_plt:
            plt.show()
        return plt


if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of Parcels trajectories""")
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
                         tracerlat=args.tracerfilelat, recordedvar=args.recordedvar,
                         show_plt=True)
