import numpy as np

__all__ = ['GridSet']


class GridSet(object):
    """GridSet class that holds the Grids on which the Fields are defined

    """

    def __init__(self):
        self.grids = []
        self.size = 0

    def add_grid(self, field):
        grid = field.grid
        existing_grid = False
        for g in self.grids:
            sameGrid = True
            if grid.time_origin != g.time_origin:
                continue
            for attr in ['lon', 'lat', 'depth', 'time']:
                gattr = getattr(g, attr)
                gridattr = getattr(grid, attr)
                if gattr.shape != gridattr.shape or not np.allclose(gattr, gridattr):
                    sameGrid = False
                    break
            if not sameGrid:
                continue
            existing_grid = True
            field.grid = g
            break

        if not existing_grid:
            self.grids.append(grid)
            self.size += 1
        field.igrid = self.grids.index(field.grid)

    def dimrange(self, dim):
        """Returns maximum value of a dimension (lon, lat, depth or time)
           on 'left' side and minimum value on 'right' side for all grids
           in a gridset. Useful for finding e.g. longitude range that
           overlaps on all grids in a gridset"""

        maxleft, minright = (0, np.infty)
        for g in self.grids:
            maxleft = max(maxleft, getattr(g, dim)[0])
            minright = min(minright, getattr(g, dim)[-1])
        return maxleft, minright
