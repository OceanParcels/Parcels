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
    reconstruct : bool, default=False
        If true, reconstructs the spatial hash

    Note
    ----
    Does not currently support queries on periodic elements.
    """

    def __init__(
        self,
        grid,
        reconstruct=False,
    ):
        # TODO : Enforce grid to be an instance of parcels.xgrid.XGrid
        # Currently, this is not done due to circular import with parcels.xgrid

        self._source_grid = grid
        self.reconstruct = reconstruct

        # Hash grid size
        self._dh = self._hash_cell_size()

        # # Lower left corner of the hash grid
        # lon_min = self._source_grid.lon.min()
        # lat_min = self._source_grid.lat.min()
        # lon_max = self._source_grid.lon.max()
        # lat_max = self._source_grid.lat.max()

        if self._source_grid.mesh == "spherical":
            # Boundaries of the hash grid are the unit cube
            self._xmin = -1.0
            self._ymin = -1.0
            self._zmin = -1.0
            self._xmax = 1.0
            self._ymax = 1.0
            self._zmax = 1.0  # Compute the cell centers of the source grid (for now, assuming Xgrid)
            lon = np.deg2rad(self._source_grid.lon)
            lat = np.deg2rad(self._source_grid.lat)
            x, y, z = _latlon_rad_to_xyz(lat, lon)
            _xbound = np.stack(
                (
                    x[:-1, :-1],
                    x[:-1, 1:],
                    x[1:, 1:],
                    x[1:, :-1],
                ),
                axis=-1,
            )
            _ybound = np.stack(
                (
                    y[:-1, :-1],
                    y[:-1, 1:],
                    y[1:, 1:],
                    y[1:, :-1],
                ),
                axis=-1,
            )
            _zbound = np.stack(
                (
                    z[:-1, :-1],
                    z[:-1, 1:],
                    z[1:, 1:],
                    z[1:, :-1],
                ),
                axis=-1,
            )
            # Compute centroid locations of each cells
            self._xc = np.mean(_xbound, axis=-1)
            self._yc = np.mean(_ybound, axis=-1)
            self._zc = np.mean(_zbound, axis=-1)

        else:
            # Boundaries of the hash grid are the bounding box of the source grid
            self._xmin = self._source_grid.lon.min()
            self._ymin = self._source_grid.lat.min()
            self._zmin = 0.0
            x = self._source_grid.lon
            y = self._source_grid.lat

            _xbound = np.stack(
                (
                    x[:-1, :-1],
                    x[:-1, 1:],
                    x[1:, 1:],
                    x[1:, :-1],
                ),
                axis=-1,
            )
            _ybound = np.stack(
                (
                    y[:-1, :-1],
                    y[:-1, 1:],
                    y[1:, 1:],
                    y[1:, :-1],
                ),
                axis=-1,
            )
            # Compute centroid locations of each cells
            self._xc = np.mean(_xbound, axis=-1)
            self._yc = np.mean(_ybound, axis=-1)
            self._zc = np.zeros_like(self._xc)

        # Generate the mapping from the hash indices to unstructured grid elements
        self._hash_table = None
        self._hash_table = self._initialize_hash_table()

    def _initialize_hash_table(self):
        """Create a mapping that relates unstructured grid faces to hash indices by determining
        which faces overlap with which hash cells
        """
        if self._hash_table is None or self.reconstruct:
            j, i = np.indices(self._xc.shape)  # Get the indices of the curvilinear grid

            morton_codes = _encode_morton3d(self._xc, self._yc, self._zc)
            ## Prepare quick lookup (hash) table for relating i,j indices to morton codes
            # Sort i,j indices by morton code
            order = np.argsort(morton_codes.ravel())
            morton_codes_sorted = morton_codes.ravel()[order]
            i_sorted = i.ravel()[order]
            j_sorted = j.ravel()[order]

            # Get a list of unique morton codes and their corresponding starts and counts (CSR format)
            keys, starts, counts = np.unique(morton_codes_sorted, return_index=True, return_counts=True)
            hash_table = {
                "keys": keys,
                "starts": starts,
                "counts": counts,
                "i": i_sorted,
                "j": j_sorted,
            }
            return hash_table

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
        keys = self._hash_table["keys"], starts = self._hash_table["starts"], counts = self._hash_table["counts"]
        i = self._hash_table["i"], j = self._hash_table["j"]

        if self._source_grid.mesh == "spherical":
            # Convert coords to Cartesian coordinates (x, y, z)
            lat = np.deg2rad(coords[:, 0])
            lon = np.deg2rad(coords[:, 1])
            x, y, z = _latlon_rad_to_xyz(lat, lon)
        else:
            # For Cartesian grids, use the coordinates directly
            x = coords[:, 0]
            y = coords[:, 1]
            z = np.zeros_like(x)

        query_codes = _encode_morton3d(x, y, z)

        # Locate each query in the unique key array
        pos = np.searchsorted(keys, query_codes)

        # Valid hits: inside range and exact match
        valid = (pos < len(keys)) & (keys[pos] == query_codes)

        # How many matches each query has
        hit_counts = np.where(valid, counts[pos], 0).astype(np.int64)

        # CSR-style offsets (prefix sum), total number of hits
        offsets = np.empty(hit_counts.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(hit_counts, out=offsets[1:])
        total = int(offsets[-1])

        # Preallocate output (total,2); early exit if no hits at all
        matches = np.empty((total, 2), dtype=i.dtype)
        # if total == 0: # TODO : Do we want to error out here ?
        #     return matches, offsets

        # For each *element* in the flattened output, figure out which query it belongs to.
        # Repeat each query index i by its hit_count to build a per-element mapping.
        q_index_for_elem = np.repeat(np.arange(query_codes.size, dtype=np.int64), hit_counts)

        # For each element, compute its "intra-group" offset (0..hits_i-1).
        # This is a vectorized trick: a global ramp minus the repeated group starts.
        intra = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], hit_counts)

        # Map each element to a source index in I/J:
        #   source_start = starts[pos[i]] for the query i
        #   source_idx   = source_start + intra
        source_idx = starts[pos[q_index_for_elem]].astype(np.int64) + intra

        # Gather all (I,J) pairs in one shot
        matches[:, 0] = j[source_idx]
        matches[:, 1] = i[source_idx]

        # Split into per-query blocks (views, no copies)
        blocks = np.split(matches, offsets[1:-1])
        # Wrap into an object array with the same leading shape as query_codes
        faces = np.empty(np.prod(query_codes.shape, dtype=int), dtype=object)
        faces[:] = blocks  # no append, just vectorized assignment

        return faces.reshape(query_codes.shape)


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


def _latlon_rad_to_xyz(lat, lon):
    """Converts Spherical latitude and longitude coordinates into Cartesian x,
    y, z coordinates.
    """
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z


def _dilate_bits(n):
    """
    Takes a 10-bit integer n, in range [0,1023], and "dilates" its bits so that
    there are two zeros between each bit of n in the result.

    This is a preparation step for building a 3D Morton code:
    - One axis (x, y, or z) is dilated like this.
    - Then the three dilated coordinates are bitwise interleaved
      to produce the full 30-bit Morton code.

    Example:
        Input n:  b9 b8 b7 b6 b5 b4 b3 b2 b1 b0
        Output:   b9 0 0 b8 0 0 b7 0 0 ... b0 0 0
    """
    n = np.asarray(n, dtype=np.uint32)

    # Step 1: Keep only the lowest 10 bits of n
    # Mask = 0x3FF = binary 11 1111 1111
    n &= np.uint32(0x000003FF)

    # Step 2: First spreading stage
    # Shift left by 16 and OR with original.
    # This spreads the bits apart, but introduces overlaps.
    # Mask 0xff0000ff clears out the unwanted overlaps.
    n = (n | (n << np.uint32(16))) & np.uint32(0xFF0000FF)

    # Step 3: Second spreading stage
    # Similar idea: shift left by 8, OR, then mask.
    # Now the bits are further separated.
    n = (n | (n << np.uint32(8))) & np.uint32(0x0300F00F)

    # Step 4: Third spreading stage
    # Shift by 4, OR, mask again.
    # At this point, there are 1 or 2 zeros between many of the bits.
    n = (n | (n << np.uint32(4))) & np.uint32(0x030C30C3)

    # Step 5: Final spreading stage
    # Shift by 2, OR, mask.
    # After this, each original bit is isolated with exactly two zeros
    # between it and the next bit, ready for 3D Morton interleaving.
    n = (n | (n << np.uint32(2))) & np.uint32(0x09249249)

    # Return the dilated value.
    return n


def _encode_morton3d(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Quantize (x, y, z) to 10 bits each (0..1023), dilate the bits so there are
    two zeros between successive bits, and interleave them into a 3D Morton code.

    Notes
    -----
    - Works with scalars or NumPy arrays (broadcasting applies).
    - Output is up to 30 bits; we return np.uint32 (or np.uint64 if you prefer).
    - Requires `part1by2` defined as in your previous snippet.
    """
    # Convert inputs to ndarray for consistent dtype/ufunc behavior.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # --- 1) Normalize each coordinate to [0, 1] over its bounding box. ---
    # Compute denominators once (avoid division by zero if bounds equal).
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    # Normalize to [0,1]; if a range is degenerate, map to 0 to avoid NaN/inf.
    xn = np.where(dx != 0, (x - xmin) / dx, 0.0)
    yn = np.where(dy != 0, (y - ymin) / dy, 0.0)
    zn = np.where(dz != 0, (z - zmin) / dz, 0.0)

    # --- 2) Quantize to 10 bits (0..1023). ---
    # Multiply by 1023, round down, and clip to be safe against slight overshoot.
    xq = np.clip((xn * 1023.0).astype(np.uint32), 0, 1023)
    yq = np.clip((yn * 1023.0).astype(np.uint32), 0, 1023)
    zq = np.clip((zn * 1023.0).astype(np.uint32), 0, 1023)

    # --- 3) Bit-dilate each 10-bit number so each bit is separated by two zeros. ---
    # _dilate_bits maps:  b9..b0  ->  b9 0 0 b8 0 0 ... b0 0 0
    dx3 = _dilate_bits(xq).astype(np.uint64)
    dy3 = _dilate_bits(yq).astype(np.uint64)
    dz3 = _dilate_bits(zq).astype(np.uint64)

    # --- 4) Interleave the dilated bits into a single Morton code. ---
    # Bit layout (from LSB upward): x0,y0,z0, x1,y1,z1, ..., x9,y9,z9
    # We shift z's bits by 2, y's by 1, x stays at 0, then OR them together.
    # Cast to a wide type before shifting/OR to be safe when arrays are used.
    code = (dz3 << 2) | (dy3 << 1) | dx3

    # If you want a compact type, it fits in 30 bits; uint32 is enough.
    return code.astype(np.uint32)
