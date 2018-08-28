from parcels.field import Field, VectorField
from parcels.grid import CurvilinearGrid
from parcels.loggers import logger
import numpy as np
from datetime import timedelta as delta
from datetime import datetime
import cftime


def plotparticles(particles, with_particles=True, show_time=None, field=None, domain=None, projection=None,
                  land=True, vmin=None, vmax=None, savefile=None, animation=False):
    """Function to plot a Parcels ParticleSet

    :param show_time: Time at which to show the ParticleSet
    :param with_particles: Boolean whether particles are also plotted on Field
    :param field: Field to plot under particles (either None, a Field object, or 'vector')
    :param domain: Four-vector (latN, latS, lonE, lonW) defining domain to show
    :param projection: type of cartopy projection to use (default PlateCarree)
    :param land: Boolean whether to show land. This is ignored for flat meshes
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    :param animation: Boolean whether result is a single plot, or an animation
    """

    show_time = particles[0].time if show_time is None else show_time
    if isinstance(show_time, datetime):
        show_time = np.datetime64(show_time)
    if isinstance(show_time, np.datetime64):
        if not particles.time_origin:
            raise NotImplementedError(
                'If fieldset.time_origin is not a date, showtime cannot be a date in particleset.show()')
        show_time = (show_time - particles.time_origin) / np.timedelta64(1, 's')
    if isinstance(show_time, delta):
        show_time = show_time.total_seconds()
    if np.isnan(show_time):
        show_time, _ = particles.fieldset.gridset.dimrange('time_full')

    if field is None:
        spherical = True if particles.fieldset.U.grid.mesh == 'spherical' else False
        plt, fig, ax, cartopy = create_parcelsfig_axis(spherical, land, projection)
        if plt is None:
            return  # creating axes was not possible
        ax.set_title('Particles' + parsetimestr(particles.fieldset.U.grid.time_origin, show_time))
        latN, latS, lonE, lonW = parsedomain(domain, particles.fieldset.U)
        if cartopy is None or projection is None:
            if domain is not None:
                if isinstance(particles.fieldset.U.grid, CurvilinearGrid):
                    ax.set_xlim(particles.fieldset.U.grid.lon[latS, lonW], particles.fieldset.U.grid.lon[latN, lonE])
                    ax.set_ylim(particles.fieldset.U.grid.lat[latS, lonW], particles.fieldset.U.grid.lat[latN, lonE])
                else:
                    ax.set_xlim(particles.fieldset.U.grid.lon[lonW], particles.fieldset.U.grid.lon[lonE])
                    ax.set_ylim(particles.fieldset.U.grid.lat[latS], particles.fieldset.U.grid.lat[latN])
            else:
                ax.set_xlim(np.nanmin(particles.fieldset.U.grid.lon), np.nanmax(particles.fieldset.U.grid.lon))
                ax.set_ylim(np.nanmin(particles.fieldset.U.grid.lat), np.nanmax(particles.fieldset.U.grid.lat))
        elif domain is not None:
            if isinstance(particles.fieldset.U.grid, CurvilinearGrid):
                ax.set_extent([particles.fieldset.U.grid.lon[latS, lonW], particles.fieldset.U.grid.lon[latN, lonE],
                               particles.fieldset.U.grid.lat[latS, lonW], particles.fieldset.U.grid.lat[latN, lonE]])
            else:
                ax.set_extent([particles.fieldset.U.grid.lon[lonW], particles.fieldset.U.grid.lon[lonE],
                               particles.fieldset.U.grid.lat[latS], particles.fieldset.U.grid.lat[latN]])

    else:
        if field is 'vector':
            field = particles.fieldset.UV
        elif not isinstance(field, Field):
            field = getattr(particles.fieldset, field)

        plt, fig, ax, cartopy = plotfield(field=field, animation=animation, show_time=show_time, domain=domain,
                                          projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=None,
                                          titlestr='Particles and ')
        if plt is None:
            return  # creating axes was not possible

    if with_particles:
        plon = np.array([p.lon for p in particles])
        plat = np.array([p.lat for p in particles])
        if cartopy:
            ax.scatter(plon, plat, s=20, color='black', zorder=20, transform=cartopy.crs.PlateCarree())
        else:
            ax.scatter(plon, plat, s=20, color='black', zorder=20)

    if animation:
        plt.draw()
        plt.pause(0.0001)
    elif savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()


def plotfield(field, show_time=None, domain=None, projection=None, land=True,
              vmin=None, vmax=None, savefile=None, **kwargs):
    """Function to plot a Parcels Field

    :param show_time: Time at which to show the Field
    :param domain: Four-vector (latN, latS, lonE, lonW) defining domain to show
    :param projection: type of cartopy projection to use (default PlateCarree)
    :param land: Boolean whether to show land. This is ignored for flat meshes
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    :param animation: Boolean whether result is a single plot, or an animation
    """

    if type(field) is VectorField:
        spherical = True if field.U.grid.mesh == 'spherical' else False
        field = [field.U, field.V]
        plottype = 'vector'
    elif type(field) is Field:
        spherical = True if field.grid.mesh == 'spherical' else False
        field = [field]
        plottype = 'scalar'
    else:
        raise RuntimeError('field needs to be a Field or VectorField object')

    plt, fig, ax, cartopy = create_parcelsfig_axis(spherical, land, projection=projection)
    if plt is None:
        return None, None, None, None  # creating axes was not possible

    data = {}
    plotlon = {}
    plotlat = {}
    for i, fld in enumerate(field):
        show_time = fld.grid.time[0] if show_time is None else show_time
        if fld.grid.defer_load:
            fld.fieldset.computeTimeChunk(show_time, 1)
        (idx, periods) = fld.time_index(show_time)
        show_time -= periods * (fld.grid.time[-1] - fld.grid.time[0])

        latN, latS, lonE, lonW = parsedomain(domain, fld)
        if isinstance(fld.grid, CurvilinearGrid):
            plotlon[i] = fld.grid.lon[latS:latN, lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN, lonW:lonE]
        else:
            plotlon[i] = fld.grid.lon[lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN]
        if i > 0 and not np.allclose(plotlon[i], plotlon[0]):
            raise RuntimeError('VectorField needs to be on an A-grid for plotting')
        if fld.grid.time.size > 1:
            data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[latS:latN, lonW:lonE]
        else:
            data[i] = np.squeeze(fld.data)[latS:latN, lonW:lonE]

    if plottype is 'vector':
        spd = data[0] ** 2 + data[1] ** 2
        speed = np.sqrt(spd, where=spd > 0)
        vmin = speed.min() if vmin is None else vmin
        vmax = speed.max() if vmax is None else vmax
        if isinstance(field[0].grid, CurvilinearGrid):
            x, y = plotlon[0], plotlat[0]
        else:
            x, y = np.meshgrid(plotlon[0], plotlat[0])
        nonzerospd = speed != 0
        u, v = (np.zeros_like(data[0]) * np.nan, np.zeros_like(data[1]) * np.nan)
        np.place(u, nonzerospd, data[0][nonzerospd] / speed[nonzerospd])
        np.place(v, nonzerospd, data[1][nonzerospd] / speed[nonzerospd])
        if cartopy:
            cs = ax.quiver(x, y, u, v, speed, cmap=plt.cm.gist_ncar, clim=[vmin, vmax], scale=50, transform=cartopy.crs.PlateCarree())
        else:
            cs = ax.quiver(x, y, u, v, speed, cmap=plt.cm.gist_ncar, clim=[vmin, vmax], scale=50)
    else:
        vmin = data[0].min() if vmin is None else vmin
        vmax = data[0].max() if vmax is None else vmax
        if cartopy:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], data[0], transform=cartopy.crs.PlateCarree())
        else:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], data[0])

    if cartopy is None or projection is None:
        ax.set_xlim(np.nanmin(plotlon[0]), np.nanmax(plotlon[0]))
        ax.set_ylim(np.nanmin(plotlat[0]), np.nanmax(plotlat[0]))
    elif domain is not None:
        ax.set_extent([np.nanmin(plotlon[0]), np.nanmax(plotlon[0]), np.nanmin(plotlat[0]), np.nanmax(plotlat[0])])
    cs.cmap.set_over('k')
    cs.cmap.set_under('w')
    cs.set_clim(vmin, vmax)

    cartopy_colorbar(cs, plt, fig, ax)

    timestr = parsetimestr(field[0].grid.time_origin, show_time)
    titlestr = kwargs.pop('titlestr', '')
    if plottype is 'vector':
        ax.set_title(titlestr + 'Velocity field' + timestr)
    else:
        ax.set_title(titlestr + field[0].name + timestr)

    if not spherical:
        ax.set_xlabel('Zonal distance [m]')
        ax.set_ylabel('Meridional distance [m]')

    plt.draw()

    if savefile:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()

    return plt, fig, ax, cartopy


def create_parcelsfig_axis(spherical, land=True, projection=None, central_longitude=0):
    try:
        import matplotlib.pyplot as plt
    except:
        logger.info("Visualisation is not possible. Matplotlib not found.")
        return None, None, None, None  # creating axes was not possible

    if projection is not None and not spherical:
        raise RuntimeError('projection not accepted when Field doesn''t have geographic coordinates')

    if spherical:
        try:
            import cartopy
        except:
            logger.info("Visualisation of field with geographic coordinates is not possible. Cartopy not found.")
            return None, None, None, None  # creating axes was not possible

        projection = cartopy.crs.PlateCarree(central_longitude) if projection is None else projection
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': projection})
        try:  # gridlines not supported for all projections
            gl = ax.gridlines(crs=projection, draw_labels=True)
            gl.xlabels_top, gl.ylabels_right = (False, False)
            gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
            gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        except:
            pass

        if land:
            ax.coastlines()
    else:
        cartopy = None
        fig, ax = plt.subplots(1, 1)
        ax.grid()
    return plt, fig, ax, cartopy


def parsedomain(domain, field):
    field.grid.check_zonal_periodic()
    if domain is not None:
        _, _, _, lonW, latS, _ = field.search_indices(domain[3], domain[1], 0, 0, 0, search2D=True)
        _, _, _, lonE, latN, _ = field.search_indices(domain[2], domain[0], 0, 0, 0, search2D=True)
        return latN, latS, lonE, lonW
    else:
        return -1, 0, -1, 0


def parsetimestr(time_origin, show_time):
    if not time_origin:
        return ' after ' + str(delta(seconds=show_time)) + ' hours'
    else:
        if isinstance(time_origin, np.datetime64):
            date_str = str(time_origin + np.timedelta64(int(show_time), 's'))
        elif isinstance(time_origin, cftime._cftime.DatetimeNoLeap):
            date_str = str(time_origin + delta(seconds=show_time))
        return ' on ' + date_str[:10] + ' ' + date_str[11:19]


def cartopy_colorbar(cs, plt, fig, ax):
    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    plt.colorbar(cs, cax=cbar_ax)

    def resize_colorbar(event):
        plt.draw()
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.04, posn.height])

    fig.canvas.mpl_connect('resize_event', resize_colorbar)
    resize_colorbar(None)
