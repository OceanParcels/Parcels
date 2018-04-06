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
