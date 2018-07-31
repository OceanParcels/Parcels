from parcels.field import Field, UnitConverter
from parcels.loggers import logger
import numpy as np
import bisect
from datetime import timedelta as delta
from datetime import datetime


def plotparticles(particles, with_particles=True, show_time=None, field=None, domain=None,
                  land=False, vmin=None, vmax=None, savefile=None):
    """Function to plot a Parcels ParticleSet

    :param with_particles: Boolean whether to show particles
    :param show_time: Time at which to show the ParticleSet
    :param field: Field to plot under particles (either None, a Field object, or 'vector')
    :param domain: Four-vector (latN, latS, lonE, lonW) defining domain to show
    :param land: Boolean whether to show land (in field='vector' mode only)
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    """

    try:
        import matplotlib.pyplot as plt
    except:
        logger.info("Visualisation is not possible. Matplotlib not found.")
        return
    try:
        import cartopy
    except:
        cartopy = None

    plon = np.array([p.lon for p in particles])
    plat = np.array([p.lat for p in particles])
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
        show_time, _ = particles.fieldset.gridset.dimrange('time')
    if domain is not None:
        def nearest_index(array, value):
            """returns index of the nearest value in array using O(log n) bisection method"""
            y = bisect.bisect(array, value)
            if y == len(array):
                return y - 1
            elif (abs(array[y - 1] - value) < abs(array[y] - value)):
                return y - 1
            else:
                return y

        latN = nearest_index(particles.fieldset.U.lat, domain[0])
        latS = nearest_index(particles.fieldset.U.lat, domain[1])
        lonE = nearest_index(particles.fieldset.U.lon, domain[2])
        lonW = nearest_index(particles.fieldset.U.lon, domain[3])
    else:
        latN, latS, lonE, lonW = (-1, 0, -1, 0)

    if field is not 'vector' and not land:
        plt.ion()
        plt.clf()
        if with_particles:
            plt.plot(np.transpose(plon), np.transpose(plat), 'ko')
        if field is None:
            axes = plt.gca()
            axes.set_xlim([particles.fieldset.U.lon[lonW], particles.fieldset.U.lon[lonE]])
            axes.set_ylim([particles.fieldset.U.lat[latS], particles.fieldset.U.lat[latN]])
        else:
            if not isinstance(field, Field):
                field = getattr(particles.fieldset, field)
            field.show(with_particles=True, show_time=show_time, vmin=vmin, vmax=vmax)
        xlbl = 'Zonal distance [m]' if type(particles.fieldset.U.units) is UnitConverter else 'Longitude [degrees]'
        ylbl = 'Meridional distance [m]' if type(particles.fieldset.U.units) is UnitConverter else 'Latitude [degrees]'
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
    elif cartopy is None:
        logger.info("Visualisation is not possible. Cartopy not found.")
    else:
        particles.fieldset.computeTimeChunk(show_time, 1)
        (idx, periods) = particles.fieldset.U.time_index(show_time)
        show_time -= periods * (particles.fieldset.U.time[-1] - particles.fieldset.U.time[0])
        lon = particles.fieldset.U.lon
        lat = particles.fieldset.U.lat
        lon = lon[lonW:lonE]
        lat = lat[latS:latN]

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': cartopy.crs.PlateCarree()})
        ax.gridlines()

        if land:
            ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
            ax.set_xlim([particles.fieldset.U.lon[lonW], particles.fieldset.U.lon[lonE]])
            ax.set_ylim([particles.fieldset.U.lat[latS], particles.fieldset.U.lat[latN]])
        if field is 'vector':
            # formatting velocity data for quiver plotting
            U = particles.fieldset.U.temporal_interpolate_fullfield(idx, show_time)[latS:latN, lonW:lonE]
            V = particles.fieldset.V.temporal_interpolate_fullfield(idx, show_time)[latS:latN, lonW:lonE]
            speed = np.sqrt(U ** 2 + V ** 2)
            x, y = np.meshgrid(lon, lat)

            nonzerospd = speed != 0
            u, v = (np.zeros_like(U) * np.nan, np.zeros_like(U) * np.nan)
            np.place(u, nonzerospd, U[nonzerospd] / speed[nonzerospd])
            np.place(v, nonzerospd, V[nonzerospd] / speed[nonzerospd])
            fld = ax.quiver(x, y, u, v, speed, cmap=plt.cm.gist_ncar, clim=[vmin, vmax], scale=50)

            cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
            fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
            plt.colorbar(fld, cax=cbar_ax)

            def resize_colorbar(event):
                plt.draw()
                posn = ax.get_position()
                cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.04, posn.height])

            fig.canvas.mpl_connect('resize_event', resize_colorbar)
            resize_colorbar(None)

        elif field is not None:
            data = field.temporal_interpolate_fullfield(idx, show_time)[latS:latN, lonW:lonE]

            vmin = data.min() if vmin is None else vmin
            vmax = data.max() if vmax is None else vmax
            fld = ax.contourf(lon, lat, data, levels=np.linspace(vmin, vmax, 256))

        # plotting particle data
        if with_particles:
            ax.scatter(plon, plat, s=20, color='black')

    if not particles.time_origin:
        timestr = ' after ' + str(delta(seconds=show_time)) + ' hours'
    else:
        date_str = str(particles.time_origin + np.timedelta64(int(show_time), 's'))
        timestr = ' on ' + date_str[:10] + ' ' + date_str[11:19]

    if with_particles:
        if field is None:
            plt.title('Particles' + timestr)
        elif field is 'vector':
            ax.set_title('Particles and velocity field' + timestr)
        else:
            plt.title('Particles and ' + field.name + timestr)
    else:
        if field is 'vector':
            ax.title('Velocity field' + timestr)
        else:
            plt.title(field.name + timestr)

    if savefile is None:
        plt.show()
        plt.pause(0.0001)
    else:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()


def plotfield(field, with_particles=False, animation=False, show_time=None, vmin=None, vmax=None):
    """Function to plot a Parcels Field

    :param with_particles: Boolean whether particles are also plotted on Field
    :param animation: Boolean whether result is a single plot, or an animation
    :param show_time: Time at which to show the Field (only in single-plot mode)
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    """

    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation_plt
        from matplotlib import rc
    except:
        logger.info("Visualisation is not possible. Matplotlib not found.")
        return

    if with_particles or (not animation):
        show_time = field.grid.time[0] if show_time is None else show_time
        if field.grid.defer_load:
            field.fieldset.computeTimeChunk(show_time, 1)
        (idx, periods) = field.time_index(show_time)
        show_time -= periods * (field.grid.time[-1] - field.grid.time[0])
        if field.grid.time.size > 1:
            data = np.squeeze(field.temporal_interpolate_fullfield(idx, show_time))
        else:
            data = np.squeeze(field.data)

        vmin = data.min() if vmin is None else vmin
        vmax = data.max() if vmax is None else vmax
        cs = plt.contourf(field.grid.lon, field.grid.lat, data,
                          levels=np.linspace(vmin, vmax, 256))
        cs.cmap.set_over('k')
        cs.cmap.set_under('w')
        cs.set_clim(vmin, vmax)
        plt.colorbar(cs)
        if not with_particles:
            plt.show()
    else:
        fig = plt.figure()
        ax = plt.axes(xlim=(field.grid.lon[0], field.grid.lon[-1]), ylim=(field.grid.lat[0], field.grid.lat[-1]))

        def animate(i):
            data = np.squeeze(field.data[i, :, :])
            cont = ax.contourf(field.grid.lon, field.grid.lat, data,
                               levels=np.linspace(data.min(), data.max(), 256))
            return cont

        rc('animation', html='html5')
        anim = animation_plt.FuncAnimation(fig, animate, frames=np.arange(1, field.data.shape[0]),
                                           interval=100, blit=False)
        plt.close()
        return anim
