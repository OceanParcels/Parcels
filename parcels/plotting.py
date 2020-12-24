from datetime import datetime
from datetime import timedelta as delta

import numpy as np
import copy

from parcels.field import Field
from parcels.field import VectorField
from parcels.grid import CurvilinearGrid
from parcels.grid import GridCode
from parcels.tools.statuscodes import TimeExtrapolationError
from parcels.tools.loggers import logger


def plotparticles(particles, with_particles=True, show_time=None, field=None, domain=None, projection=None,
                  land=True, vmin=None, vmax=None, savefile=None, animation=False, **kwargs):
    """Function to plot a Parcels ParticleSet

    :param show_time: Time at which to show the ParticleSet
    :param with_particles: Boolean whether particles are also plotted on Field
    :param field: Field to plot under particles (either None, a Field object, or 'vector')
    :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
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
        show_time = particles.time_origin.reltime(show_time)
    if isinstance(show_time, delta):
        show_time = show_time.total_seconds()
    if np.isnan(show_time):
        show_time, _ = particles.fieldset.gridset.dimrange('time_full')

    if field is None:
        spherical = True if particles.fieldset.U.grid.mesh == 'spherical' else False
        plt, fig, ax, cartopy = create_parcelsfig_axis(spherical, land, projection, cartopy_features=kwargs.pop('cartopy_features', []))
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
        if field == 'vector':
            field = particles.fieldset.UV
        elif not isinstance(field, Field):
            field = getattr(particles.fieldset, field)

        depth_level = kwargs.pop('depth_level', 0)
        plt, fig, ax, cartopy = plotfield(field=field, animation=animation, show_time=show_time, domain=domain,
                                          projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=None,
                                          titlestr='Particles and ', depth_level=depth_level, **kwargs)
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


def plotfield(field, show_time=None, domain=None, depth_level=0, projection=None, land=True,
              vmin=None, vmax=None, savefile=None, **kwargs):
    """Function to plot a Parcels Field

    :param show_time: Time at which to show the Field
    :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
    :param depth_level: depth level to be plotted (default 0)
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

    if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.CurvilinearSGrid]:
        logger.warning('Field.show() does not always correctly determine the domain for curvilinear grids. '
                       'Use plotting with caution and perhaps use domain argument as in the NEMO 3D tutorial')

    plt, fig, ax, cartopy = create_parcelsfig_axis(spherical, land, projection=projection, cartopy_features=kwargs.pop('cartopy_features', []))
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
        show_time -= periods * (fld.grid.time_full[-1] - fld.grid.time_full[0])
        if show_time > fld.grid.time[-1] or show_time < fld.grid.time[0]:
            raise TimeExtrapolationError(show_time, field=fld, msg='show_time')

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
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[depth_level, latS:latN, lonW:lonE]
            else:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[latS:latN, lonW:lonE]
        else:
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.data)[depth_level, latS:latN, lonW:lonE]
            else:
                data[i] = np.squeeze(fld.data)[latS:latN, lonW:lonE]

    if plottype == 'vector':
        if field[0].interp_method == 'cgrid_velocity':
            logger.warning_once('Plotting a C-grid velocity field is achieved via an A-grid projection, reducing the plot accuracy')
            d = np.empty_like(data[0])
            d[:-1, :] = (data[0][:-1, :] + data[0][1:, :]) / 2.
            d[-1, :] = data[0][-1, :]
            data[0] = d
            d = np.empty_like(data[0])
            d[:, :-1] = (data[0][:, :-1] + data[0][:, 1:]) / 2.
            d[:, -1] = data[0][:, -1]
            data[1] = d

        spd = data[0] ** 2 + data[1] ** 2
        speed = np.where(spd > 0, np.sqrt(spd), 0)
        vmin = speed.min() if vmin is None else vmin
        vmax = speed.max() if vmax is None else vmax
        ncar_cmap = copy.copy(plt.cm.gist_ncar)
        ncar_cmap.set_over('k')
        ncar_cmap.set_under('w')
        if isinstance(field[0].grid, CurvilinearGrid):
            x, y = plotlon[0], plotlat[0]
        else:
            x, y = np.meshgrid(plotlon[0], plotlat[0])
        u = np.where(speed > 0., data[0]/speed, 0)
        v = np.where(speed > 0., data[1]/speed, 0)
        if cartopy:
            cs = ax.quiver(np.asarray(x), np.asarray(y), np.asarray(u), np.asarray(v), speed, cmap=ncar_cmap, clim=[vmin, vmax], scale=50, transform=cartopy.crs.PlateCarree())
        else:
            cs = ax.quiver(x, y, u, v, speed, cmap=ncar_cmap, clim=[vmin, vmax], scale=50)
    else:
        vmin = data[0].min() if vmin is None else vmin
        vmax = data[0].max() if vmax is None else vmax
        pc_cmap = copy.copy(plt.cm.get_cmap('viridis'))
        pc_cmap.set_over('k')
        pc_cmap.set_under('w')
        assert len(data[0].shape) == 2
        if field[0].interp_method == 'cgrid_tracer':
            d = data[0][1:, 1:]
        elif field[0].interp_method == 'cgrid_velocity':
            if field[0].fieldtype == 'U':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][1:, :-1] + data[0][1:, 1:]) / 2.
            elif field[0].fieldtype == 'V':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][:-1, 1:] + data[0][1:, 1:]) / 2.
            else:  # W
                d = data[0][1:, 1:]
        else:  # if A-grid
            d = (data[0][:-1, :-1] + data[0][1:, :-1] + data[0][:-1, 1:] + data[0][1:, 1:])/4.
            d = np.where(data[0][:-1, :-1] == 0, 0, d)
            d = np.where(data[0][1:, :-1] == 0, 0, d)
            d = np.where(data[0][1:, 1:] == 0, 0, d)
            d = np.where(data[0][:-1, 1:] == 0, 0, d)
        if cartopy:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], d, cmap=pc_cmap, transform=cartopy.crs.PlateCarree())
        else:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], d, cmap=pc_cmap)

    if cartopy is None:
        ax.set_xlim(np.nanmin(plotlon[0]), np.nanmax(plotlon[0]))
        ax.set_ylim(np.nanmin(plotlat[0]), np.nanmax(plotlat[0]))
    elif domain is not None:
        ax.set_extent([np.nanmin(plotlon[0]), np.nanmax(plotlon[0]), np.nanmin(plotlat[0]), np.nanmax(plotlat[0])], crs=cartopy.crs.PlateCarree())
    cs.set_clim(vmin, vmax)

    cartopy_colorbar(cs, plt, fig, ax)

    timestr = parsetimestr(field[0].grid.time_origin, show_time)
    titlestr = kwargs.pop('titlestr', '')
    if field[0].grid.zdim > 1:
        if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.RectilinearZGrid]:
            gphrase = 'depth'
            depth_or_level = field[0].grid.depth[depth_level]
        else:
            gphrase = 'level'
            depth_or_level = depth_level
        depthstr = ' at %s %g ' % (gphrase, depth_or_level)
    else:
        depthstr = ''
    if plottype == 'vector':
        ax.set_title(titlestr + 'Velocity field' + depthstr + timestr)
    else:
        ax.set_title(titlestr + field[0].name + depthstr + timestr)

    if not spherical:
        ax.set_xlabel('Zonal distance [m]')
        ax.set_ylabel('Meridional distance [m]')

    plt.draw()

    if savefile:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()

    return plt, fig, ax, cartopy


def create_parcelsfig_axis(spherical, land=True, projection=None, central_longitude=0, cartopy_features=[]):
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
            if isinstance(projection, cartopy.crs.PlateCarree) and central_longitude != 0:
                gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True)  # central_lon=0 necessary for correct xlabels
            else:
                gl = ax.gridlines(crs=projection, draw_labels=True)
            gl.top_labels, gl.right_labels = (False, False)
            gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
            gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        except:
            pass

        for feature in cartopy_features:
            ax.add_feature(feature)

        if isinstance(land, str):
            ax.coastlines(land)
        elif land:
            ax.coastlines()
    else:
        cartopy = None
        fig, ax = plt.subplots(1, 1)
        ax.grid()
    return plt, fig, ax, cartopy


def parsedomain(domain, field):
    field.grid.check_zonal_periodic()
    if domain is not None:
        if not isinstance(domain, dict) and len(domain) == 4:  # for backward compatibility with <v2.0.0
            domain = {'N': domain[0], 'S': domain[1], 'E': domain[2], 'W': domain[3]}
        _, _, _, lonW, latS, _ = field.search_indices(domain['W'], domain['S'], 0, 0, 0, search2D=True)
        _, _, _, lonE, latN, _ = field.search_indices(domain['E'], domain['N'], 0, 0, 0, search2D=True)
        return latN+1, latS, lonE+1, lonW
    else:
        if field.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            return field.grid.lon.shape[0], 0, field.grid.lon.shape[1], 0
        else:
            return len(field.grid.lat), 0, len(field.grid.lon), 0


def parsetimestr(time_origin, show_time):
    if time_origin.calendar is None:
        return ' after ' + str(delta(seconds=show_time)) + ' hours'
    else:
        date_str = str(time_origin.fulltime(show_time))
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
