from argparse import ArgumentParser
from os import environ

import numpy as np
import xarray as xr

from parcels import Field
from parcels.plotting import cartopy_colorbar
from parcels.plotting import create_parcelsfig_axis
from parcels.plotting import plotfield
try:
    import matplotlib.animation as animation
    from matplotlib import rc
except:
    anim = None


def plotTrajectoriesFile(filename, mode='2d', tracerfile=None, tracerfield='P',
                         tracerlon='x', tracerlat='y', recordedvar=None, movie_forward=True,
                         bins=20, show_plt=True, central_longitude=0):
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
    :param movie_forward: Boolean whether to show movie in forward or backward mode (default True)
    :param bins: Number of bins to use in `hist2d` mode. See also https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist2d.html
    :param show_plt: Boolean whether plot should directly be show (for py.test)
    :param central_longitude: Degrees East at which to center the plot
    """

    environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    try:
        pfile = xr.open_dataset(str(filename), decode_cf=True)
    except:
        pfile = xr.open_dataset(str(filename), decode_cf=False)
    lon = np.ma.filled(pfile.variables['lon'], np.nan)
    lat = np.ma.filled(pfile.variables['lat'], np.nan)
    time = np.ma.filled(pfile.variables['time'], np.nan)
    z = np.ma.filled(pfile.variables['z'], np.nan)
    mesh = pfile.attrs['parcels_mesh'] if 'parcels_mesh' in pfile.attrs else 'spherical'

    if(recordedvar is not None):
        record = np.ma.filled(pfile.variables[recordedvar], np.nan)
    pfile.close()

    if tracerfile is not None and mode != 'hist2d':
        tracerfld = Field.from_netcdf(tracerfile, tracerfield, {'lon': tracerlon, 'lat': tracerlat})
        plt, fig, ax, cartopy = plotfield(tracerfld)
        if plt is None:
            return  # creating axes was not possible
        titlestr = ' and ' + tracerfield
    else:
        spherical = False if mode == '3d' or mesh == 'flat' else True
        plt, fig, ax, cartopy = create_parcelsfig_axis(spherical=spherical, central_longitude=central_longitude)
        if plt is None:
            return  # creating axes was not possible
        titlestr = ''

    if cartopy:
        for p in range(lon.shape[1]):
            lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

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
        if cartopy:
            ax.plot(np.transpose(lon), np.transpose(lat), '.-', transform=cartopy.crs.Geodetic())
        else:
            ax.plot(np.transpose(lon), np.transpose(lat), '.-')
        ax.set_title('Particle trajectories' + titlestr)
    elif mode == 'hist2d':
        _, _, _, cs = plt.hist2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)], bins=bins)
        cartopy_colorbar(cs, plt, fig, ax)
        ax.set_title('Particle histogram')
    elif mode in ('movie2d', 'movie2d_notebook'):
        if mesh == 'flat':
            ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
        else:
            ax.set_xlim(np.nanmin((lon+central_longitude+180) % 360 - 180), np.nanmax((lon+central_longitude+180) % 360 - 180))
        ax.set_ylim(np.nanmin(lat), np.nanmax(lat))
        plottimes = np.unique(time)
        if not movie_forward:
            plottimes = np.flip(plottimes, 0)
        if isinstance(plottimes[0], (np.datetime64, np.timedelta64)):
            plottimes = plottimes[~np.isnat(plottimes)]
        else:
            try:
                plottimes = plottimes[~np.isnan(plottimes)]
            except:
                pass
        b = time == plottimes[0]

        def timestr(plottimes, index):
            if isinstance(plottimes[index], np.timedelta64):
                if plottimes[-1] > np.timedelta64(1, 'h'):
                    return str(plottimes[index].astype('timedelta64[h]'))
                elif plottimes[-1] > np.timedelta64(1, 's'):
                    return str(plottimes[index].astype('timedelta64[s]'))
            else:
                return str(plottimes[index])

        if cartopy:
            scat = ax.scatter(lon[b], lat[b], s=20, color='k', transform=cartopy.crs.Geodetic())
        else:
            scat = ax.scatter(lon[b], lat[b], s=20, color='k')
        ttl = ax.set_title('Particles' + titlestr + ' at time ' + timestr(plottimes, 0))
        frames = np.arange(0, len(plottimes))

        def animate(t):
            b = time == plottimes[t]
            scat.set_offsets(np.vstack((lon[b], lat[b])).transpose())
            ttl.set_text('Particle' + titlestr + ' at time ' + timestr(plottimes, t))
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
