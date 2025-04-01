from parcels.v4.grid import Grid


class GridAdapter(Grid):
    def __init__(self, ds, *args, **kwargs):
        super().__init__(ds, *args, **kwargs)

    @property
    def lon(self): ...

    @property
    def lat(self): ...

    @property
    def depth(self): ...

    @property
    def time(self): ...

    @property
    def xdim(self): ...

    @property
    def ydim(self): ...

    @property
    def zdim(self): ...
    @property
    def tdim(self): ...

    @property
    def time_origin(self): ...

    @property
    def mesh(self): ...  # ? hmmm

    @property
    def zonal_periodic(self): ...  # ? hmmm

    @property
    def lonlat_minmax(self): ...  # ? hmmm

    @staticmethod
    def create_grid(lon, lat, depth, time, time_origin, mesh, **kwargs): ...  # ? hmmm

    def _check_zonal_periodic(self): ...  # ? hmmm

    def _add_Sdepth_periodic_halo(self, zonal, meridional, halosize): ...  # ? hmmm
