from parcels.interaction.base_neighbor import BaseFlatNeighborSearch
from parcels.interaction.base_neighbor import BaseSphericalNeighborSearch


class BruteFlatNeighborSearch(BaseFlatNeighborSearch):
    '''Brute force implementation to find the neighbors.'''
    name = "brute force"

    def find_neighbors_by_coor(self, coor):
        return self._get_close_neighbor_dist(coor, self.active_idx)


class BruteSphericalNeighborSearch(BaseSphericalNeighborSearch):
    '''Brute force implementation to find the neighbors.'''
    name = "brute force"

    def find_neighbors_by_coor(self, coor):
        return self._get_close_neighbor_dist(coor, self.active_idx)
