#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
from argparse import ArgumentParser
from parcels import Field
from parcels.plotting import create_parcelsfig_axis, plotfield
try:
    import matplotlib.animation as animation
    from matplotlib import rc
except:
    anim = None


def plotTrajectoriesFile(filename, mode='2d', tracerfile=None, tracerfield='P',
                         tracerlon='x', tracerlat='y', recordedvar=None, bins=20, show_plt=True):
    """Quick and simple plotting of Parcels trajectories

    :param filename: Name of Parcels-generated NetCDF file with particle positions
    :param mode: Type of plot to show. Supported are '2d', '3d', 'hist2d',
                'movie2d' and 'movie2d_notebook'. The latter two give animations,
                with 'movie2d_notebook' specifically designed for jupyter notebooks
    :param tracerfile: Name of NetCDF file to show as background
    :param tracerfield: Name of variable to show as background
    :param tracerlon: Name of longitude dimension of variable to show as background
    :param tracerlat: Name of latitude dimension of variable to show as background
    :param recordedvar: Name of variable used to color particles in scatter-plot.
                Only works in 'movie2d' or 'movie2d_notebook' mode.
    :param bins: Number of bins to use in `hist2d` mode. See also https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist2d.html
    :param show_plt: Boolean whether plot should directly be show (for py.test)
    """

    pfile = Dataset(filename, 'r')
    lon = np.ma.filled(pfile.variables['lon'], np.nan)
    lat = np.ma.filled(pfile.variables['lat'], np.nan)
    time = np.ma.filled(pfile.variables['time'], np.nan)
    z = np.ma.filled(pfile.variables['z'], np.nan)

    if(recordedvar is not None):
        record = pfile.variables[recordedvar]

    if tracerfile is not None and mode is not 'hist2d':
        tracerfld = Field.from_netcdf(tracerfile, tracerfield, {'lon': tracerlon, 'lat': tracerlat})
        plt, fig, ax = plotfield(tracerfld)
        if plt is None:
            return  # creating axes was not possible
        titlestr = ' and ' + tracerfield
    else:
        geomap = False if mode is '3d' else True
        plt, fig, ax = create_parcelsfig_axis(geomap=geomap, land=geomap)
        if plt is None:
            return  # creating axes was not possible
        ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
        ax.set_ylim(np.nanmin(lat), np.nanmax(lat))
        titlestr = ''

    if mode == '3d':
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        plt.clf()  # clear the figure
        ax = fig.gca(projection='3d')
        for p in range(len(lon)):
            ax.plot(lon[p, :], lat[p, :], z[p, :], '.-')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth')
        ax.set_title('Particle trajectories')
    elif mode == '2d':
        ax.plot(np.transpose(lon), np.transpose(lat), '.-')
        ax.set_title('Particle trajectories' + titlestr)
    elif mode == 'hist2d':
        plt.hist2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)], bins=bins)
        plt.colorbar()
        ax.set_title('Particle histogram')
    elif mode in ('movie2d', 'movie2d_notebook'):
        # ax = plt.axes(xlim=(np.nanmin(lon), np.nanmax(lon)), ylim=(np.nanmin(lat), np.nanmax(lat)))
        plottimes = np.unique(time)
        plottimes = plottimes[~np.isnan(plottimes)]
        b = time == plottimes[0]
        scat = ax.scatter(lon[b], lat[b], s=60, color='k')
        ttl = ax.set_title('Particle' + titlestr + ' at time ' + str(plottimes[0]))
        frames = np.arange(1, len(plottimes))

        def animate(t):
            b = time == plottimes[t]
            scat.set_offsets(np.vstack((lon[b], lat[b])).transpose())
            ttl.set_text('Particle' + titlestr + ' at time ' + str(plottimes[t]))
            if recordedvar is not None:
                scat.set_array(record[b])
            return scat,

        rc('animation', html='html5')
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
    else:
        raise RuntimeError('mode %s not known' % mode)

    if mode == 'movie2d_notebook':
        plt.close()
        return anim
    else:
        if show_plt:
            plt.show()
        return plt


if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of Parcels trajectories""")
    p.add_argument('mode', choices=('2d', '3d', 'hist2d', 'movie2d', 'movie2d_notebook'), nargs='?',
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
    p.add_argument('-bins', type=int, default=20,
                   help='Number of bins for mode=hist2d')
    args = p.parse_args()

    plotTrajectoriesFile(args.particlefile, mode=args.mode, tracerfile=args.tracerfile,
                         tracerfield=args.tracerfilefield, tracerlon=args.tracerfilelon,
                         tracerlat=args.tracerfilelat, recordedvar=args.recordedvar,
                         bins=args.bins, show_plt=True)
