from parcels.grid import Grid

__all__ = ['GridSet', 'GridIndexSet']

class GridSet(object):
    """GridSet class that holds the Grids on which the Fields are defined


    :param grids: Dictionary of :class:`parcels.grid.Grid` objects
    """

    def __init__(self, grids=[]):
        self.grids = {}
        for grid in grids:
            setattr(self, grid.name, grid)
            self.grids[grid.name] = grid
    
    def add_grid(self, grid):
        setattr(self, grid.name, grid)

class GridIndexSet(object):
    """GridIndexSet class that holds the GridIndices which store the particle position indices for the different grids


    :param gridset: GridSet object 
    """

    def __init__(self, gridSet):
        for grid in gridSet:
            self.grid = GridIndex(grid)
    
