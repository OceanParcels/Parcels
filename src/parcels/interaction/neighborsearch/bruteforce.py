from parcels.interaction.neighborsearch.base import (
    BaseFlatNeighborSearch,
    BaseSphericalNeighborSearch,
)


class BruteFlatNeighborSearch(BaseFlatNeighborSearch):
    """Brute force implementation to find the neighbors."""

    def find_neighbors_by_coor(self, coor):
        return self._get_close_neighbor_dist(coor, self.active_idx)


class BruteSphericalNeighborSearch(BaseSphericalNeighborSearch):
    """Brute force implementation to find the neighbors."""

    def find_neighbors_by_coor(self, coor):
        return self._get_close_neighbor_dist(coor, self.active_idx)
