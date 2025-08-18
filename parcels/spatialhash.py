import numpy as np


class SpatialHash:
    """Custom data structure that is used for performing grid searches using Spatial Hashing. This class constructs an overlying
    uniformly spaced rectilinear grid, called the "hash grid" on top parcels.xgrid.XGrid. It is particularly useful for grid searching
    on curvilinear grids. Faces in the Xgrid are related to the cells in the hash grid by determining the hash cells the bounding box
    of the unstructured face cells overlap with.

    Parameters
    ----------
    grid : parcels.xgrid.XGrid
        Source grid used to construct the hash grid and hash table
    dh_factor : float, default=2.0
        Factor by which the hash cell size is multiplied. The hash cell size is set to the square root of the median cell area
        multiplied by this factor.
    reconstruct : bool, default=False
        If true, reconstructs the spatial hash

    Note
    ----
    Does not currently support queries on periodic elements.
    """

    def __init__(
        self,
        grid,
        dh_factor=2.0,
        reconstruct=False,
    ):
        # TODO : Enforce grid to be an instance of parcels.xgrid.XGrid
        # Currently, this is not done due to circular import with parcels.xgrid

        self._source_grid = grid
        self.reconstruct = reconstruct

        # Hash grid size
        self._dh_factor = dh_factor
        self._dh = self._hash_cell_size()

        # Lower left corner of the hash grid
        lon_min = self._source_grid.lon.min()
        lat_min = self._source_grid.lat.min()
        lon_max = self._source_grid.lon.max()
        lat_max = self._source_grid.lat.max()

        # Get corner vertices of each face
        self._xbound = np.stack(
            (
                self._source_grid.lon[:-1, :-1],
                self._source_grid.lon[:-1, 1:],
                self._source_grid.lon[1:, 1:],
                self._source_grid.lon[1:, :-1],
            ),
            axis=-1,
        )
        self._ybound = np.stack(
            (
                self._source_grid.lat[:-1, :-1],
                self._source_grid.lat[:-1, 1:],
                self._source_grid.lat[1:, 1:],
                self._source_grid.lat[1:, :-1],
            ),
            axis=-1,
        )

        self._xmin = lon_min - self._dh
        self._ymin = lat_min - self._dh
        self._xmax = lon_max + self._dh
        self._ymax = lat_max + self._dh

        # Number of x points in the hash grid; used for
        # array flattening
        Lx = self._xmax - self._xmin
        Ly = self._ymax - self._ymin
        self._nx = int(np.ceil(Lx / self._dh))
        self._ny = int(np.ceil(Ly / self._dh))

        # Generate the mapping from the hash indices to unstructured grid elements
        self._face_hash_table = None
        self._face_hash_table = self._initialize_face_hash_table()

    def _hash_cell_size(self):
        """Computes the size of the hash cells from the source grid.
        The hash cell size is set to of the square root of the median cell area multiplied by the dh_factor.
        """
        return np.sqrt(np.median(_planar_quad_area(self._source_grid.lat, self._source_grid.lon))) * self._dh_factor

    def _hash_index2d(self, coords):
        """Computes the 2-d hash index (i,j) for the location (x,y), where x and y is the same units
        as the source grid coordinates
        """
        # Wrap longitude to [-180, 180]
        if self._source_grid.mesh == "spherical":
            lon = (coords[:, 1] + 180.0) % (360.0) - 180.0
        else:
            lon = coords[:, 1]
        i = ((lon - self._xmin) / self._dh).astype(np.int32)
        j = ((coords[:, 0] - self._ymin) / self._dh).astype(np.int32)
        return j, i

    def _hash_index(self, coords):
        """Computes the flattened hash index for the location (x,y), where x and y are given in spherical
        coordinates (in degrees). The single dimensioned hash index orders the flat index with all of the
        i-points first and then all the j-points.
        """
        j, i = self._hash_index2d(coords)
        return i + self._nx * j

    def _grid_ji_for_eid(self, eid):
        """Returns the (i,j) grid coordinates for the given element id (eid)"""
        j = eid // (self._source_grid.xdim)
        i = eid - j * (self._source_grid.xdim)
        return j, i

    def _get_hash(self, index):
        """Returns the list of face ids for the given hash index"""
        start = self._face_hash_table["row_ptr"][index]
        end = self._face_hash_table["row_ptr"][index + 1]
        return self._face_hash_table["column_index"][start:end]

    def _initialize_face_hash_table(self):
        """Create a mapping that relates unstructured grid faces to hash indices by determining
        which faces overlap with which hash cells
        """
        # Cache attributes to local variables for speed
        nx, ny = self._nx, self._ny
        n_cells = nx * ny
        if self._face_hash_table is None or self.reconstruct:
            lat_bounds, lon_bounds = _curvilinear_grid_facebounds(
                self._source_grid.lat,
                self._source_grid.lon,
            )
            # Use ravel instead of flatten to return a view when possible instead of a copy
            coords0 = np.stack(
                (
                    lat_bounds[:, :, 0].ravel(),
                    lon_bounds[:, :, 0].ravel(),
                ),
                axis=-1,
            )
            yi0, xi0 = self._hash_index2d(coords0)
            coords1 = np.stack(
                (
                    lat_bounds[:, :, 1].ravel(),
                    lon_bounds[:, :, 1].ravel(),
                ),
                axis=-1,
            )
            yi1, xi1 = self._hash_index2d(coords1)

            nface = (self._source_grid.xdim) * (self._source_grid.ydim)

            # Compute indices for the bounding boxes
            wrap = xi0 > xi1  # Find all hash cell index maps that wrap cross over the antimeridian

            # Count up the total number of entries in the hash table
            len_j = yi1 - yi0 + 1
            len_i_nowrap = xi1 - xi0 + 1
            len_i = np.where(wrap, (nx - xi0) + (xi1 + 1), len_i_nowrap)
            counts = len_j * len_i  # Count the number of hash cells that each face overlaps with
            total = int(counts.sum())

            # Here we first store the raveled 2d index of the parent grid and the hash grid cell ids
            idx_cells = np.empty(total, dtype=np.int32)
            face_ids = np.empty(total, dtype=np.int32)

            offset = 0
            for eid in range(nface):
                ylo = yi0[eid]
                yhi = yi1[eid]
                if wrap[eid]:
                    i_range = np.concatenate(
                        (
                            np.arange(xi0[eid], nx, dtype=np.int32),
                            np.arange(0, xi1[eid] + 1, dtype=np.int32),
                        )
                    )
                else:
                    i_range = np.arange(xi0[eid], xi1[eid] + 1, dtype=np.int32)

                width = i_range.size
                for j in range(ylo, yhi + 1):
                    idx_cells[offset : offset + width] = i_range + nx * j  # get the hash cell index
                    face_ids[offset : offset + width] = eid  # get the flattened parent grid 2d index (face id)
                    offset += width  # bump the offset by the width of this strip of indices

        # Sort by the hash cell index
        order = np.argsort(idx_cells, kind="mergesort")
        idx_sorted = idx_cells[order]
        faces_sorted = face_ids[order]

        # Create the CSR representation of the hash table
        # Recall the hash table is queried by the hash index (idx_sorted) and returns a list of face ids on the parent grid
        # In a CSR represntation, we have a row pointer that indicates the start of each row in the data array
        # and a data array that contains the face ids for each hash cell
        self._face_hash_table = np.empty(n_cells, dtype=object)
        row_ptr = np.zeros(n_cells + 1, dtype=np.int32)  # Initialize the row pointer for the CSR representation
        unique_idx, counts_per_cell = np.unique(
            idx_sorted, return_counts=True
        )  # Get a unique list of hash indices and how many times each occurs
        row_ptr[unique_idx + 1] = counts_per_cell  # Set the row pointer for each unique index
        row_ptr = np.cumsum(row_ptr)  # Cumulatively sum the row pointer to get the start of each row in the data array

        return {"row_ptr": row_ptr, "column_index": faces_sorted}

    def query(
        self,
        coords,
        tol=1e-6,
    ):
        """Queries the hash table.

        Parameters
        ----------
        coords : array_like
            coordinate pairs in degrees (lat, lon) to query.


        Returns
        -------
        faces : ndarray of shape (coords.shape[0]), dtype=np.int32
            Face id's in the self._source_grid where each coords element is found. When a coords element is not found, the
            corresponding array entry in faces is set to -1.
        """
        num_coords = coords.shape[0]

        # Preallocate results
        faces = np.full((num_coords, 2), -1, dtype=np.int32)

        # Get the list of candidate faces for each coordinate
        candidate_faces = [self._get_hash(pid) for pid in self._hash_index(coords)]

        for i, (coord, candidates) in enumerate(zip(coords, candidate_faces, strict=False)):
            for face_id in candidates:
                yi, xi = self._grid_ji_for_eid(face_id)
                nodes = np.stack(
                    (
                        self._ybound[yi, xi, :],
                        self._xbound[yi, xi, :],
                    ),
                    axis=-1,
                )

                bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
                err = abs(np.dot(bcoord, nodes[:, 0]) - coord[0]) + abs(np.dot(bcoord, nodes[:, 1]) - coord[1])
                if (bcoord >= 0).all() and err < tol:
                    faces[i, :] = [yi, xi]
                    break

        return faces


def _triangle_area(A, B, C):
    """Compute the area of a triangle given by three points."""
    d1 = B - A
    d2 = C - A
    d3 = np.cross(d1, d2)
    return 0.5 * np.linalg.norm(d3)


def _barycentric_coordinates(nodes, point, min_area=1e-8):
    """
    Compute the barycentric coordinates of a point P inside a convex polygon using area-based weights.
    So that this method generalizes to n-sided polygons, we use the Waschpress points as the generalized
    barycentric coordinates, which is only valid for convex polygons.

    Parameters
    ----------
        nodes : numpy.ndarray
            Spherical coordinates (lat,lon) of each corner node of a face
        point : numpy.ndarray
            Spherical coordinates (lat,lon) of the point

    Returns
    -------
    numpy.ndarray
        Barycentric coordinates corresponding to each vertex.

    """
    n = len(nodes)
    sum_wi = 0
    w = []

    for i in range(0, n):
        vim1 = nodes[i - 1]
        vi = nodes[i]
        vi1 = nodes[(i + 1) % n]
        a0 = _triangle_area(vim1, vi, vi1)
        a1 = max(_triangle_area(point, vim1, vi), min_area)
        a2 = max(_triangle_area(point, vi, vi1), min_area)
        sum_wi += a0 / (a1 * a2)
        w.append(a0 / (a1 * a2))
    barycentric_coords = [w_i / sum_wi for w_i in w]

    return barycentric_coords


def _planar_quad_area(lat, lon):
    """Computes the area of each quadrilateral face in a curvilinear grid.
    The lon and lat arrays are assumed to be 2D arrays of points with dimensions (n_y, n_x).
    The area is computed using the Shoelace formula.
    This method is only used during hashgrid construction to determine the size of the hash cells.

    Parameters
    ----------
    lon : np.ndarray
        2D array of shape (n_y, n_x) containing the longitude of each corner node of the curvilinear grid.
    lat : np.ndarray
        2D array of shape (n_y, n_x) containing the latitude of each corner node of the curvilinear grid.

    Returns
    -------
    area : np.ndarray
        2D array of shape (n_y-1, n_x-1) containing the area of each quadrilateral face in the curvilinear grid.
    """
    x0 = lon[:-1, :-1]
    x1 = lon[:-1, 1:]
    x2 = lon[1:, 1:]
    x3 = lon[1:, :-1]

    y0 = lat[:-1, :-1]
    y1 = lat[:-1, 1:]
    y2 = lat[1:, 1:]
    y3 = lat[1:, :-1]

    # Shoelace formula: 0.5 * |sum(x_i*y_{i+1} - x_{i+1}*y_i)|
    area = 0.5 * np.abs(x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0 - y0 * x1 - y1 * x2 - y2 * x3 - y3 * x0)
    return area


def _curvilinear_grid_facebounds(lat, lon):
    """Computes the bounds of each curvilinear face in the grid.
    The lon and lat arrays are assumed to be 2D arrays of points with dimensions (n_y, n_x).
    The bounds are for faces whose corner node vertices are defined by lat,lon.
    Face(yi,xi) is surrounding by points (yi,xi), (yi,xi+1), (yi+1,xi+1), (yi+1,xi).
    This method is only used during hashgrid construction to determine which curvilinear
    faces overlap with which hash cells.

    Parameters
    ----------
    lon : np.ndarray
        2D array of shape (n_y, n_x) containing the longitude of each corner node of the curvilinear grid.
    lat : np.ndarray
        2D array of shape (n_y, n_x) containing the latitude of each corner node of the curvilinear grid.

    Returns
    -------
    xbounds : np.ndarray
        Array of shape (n_y-1, n_x-1, 2) containing the bounds of each face in the x-direction.
    ybounds : np.ndarray
        Array of shape (n_y-1, n_x-1, 2) containing the bounds of each face in the y-direction.
    """
    xf = np.stack((lon[:-1, :-1], lon[:-1, 1:], lon[1:, 1:], lon[1:, :-1]), axis=-1)
    xf_low = xf.min(axis=-1)
    xf_high = xf.max(axis=-1)
    xbounds = np.stack([xf_low, xf_high], axis=-1)

    yf = np.stack((lat[:-1, :-1], lat[:-1, 1:], lat[1:, 1:], lat[1:, :-1]), axis=-1)
    yf_low = yf.min(axis=-1)
    yf_high = yf.max(axis=-1)
    ybounds = np.stack([yf_low, yf_high], axis=-1)

    return ybounds, xbounds
