import numpy as np

from parcels._core.index_search import (
    GRID_SEARCH_ERROR,
    _latlon_rad_to_xyz,
    curvilinear_point_in_cell,
    uxgrid_point_in_cell,
)
from parcels._python import isinstance_noimport


class SpatialHash:
    """Custom data structure that is used for performing grid searches using Spatial Hashing. This class constructs an overlying
    uniformly spaced rectilinear grid, called the "hash grid" on top parcels.XGrid. It is particularly useful for grid searching
    on curvilinear grids. Faces in the Xgrid are related to the cells in the hash grid by determining the hash cells the bounding box
    of the unstructured face cells overlap with.

    Parameters
    ----------
    grid : parcels.XGrid
        Source grid used to construct the hash grid and hash table

    Note
    ----
    Does not currently support queries on periodic elements.
    """

    def __init__(
        self,
        grid,
        bitwidth=1023,
    ):
        if isinstance_noimport(grid, "XGrid"):
            self._point_in_cell = curvilinear_point_in_cell
        elif isinstance_noimport(grid, "UxGrid"):
            self._point_in_cell = uxgrid_point_in_cell
        else:
            raise ValueError("Expected `grid` to be a parcels.XGrid or parcels.UxGrid")

        self._source_grid = grid
        self._bitwidth = bitwidth  # Max integer to use per coordinate in quantization (10 bits = 0..1023)

        if isinstance_noimport(grid, "XGrid"):
            self._coord_dim = 2  # Number of computational coordinates is 2 (bilinear interpolation)
            if self._source_grid._mesh == "spherical":
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
                self._xlow = np.min(_xbound, axis=-1)
                self._xhigh = np.max(_xbound, axis=-1)
                self._ylow = np.min(_ybound, axis=-1)
                self._yhigh = np.max(_ybound, axis=-1)
                self._zlow = np.min(_zbound, axis=-1)
                self._zhigh = np.max(_zbound, axis=-1)

            else:
                # Boundaries of the hash grid are the bounding box of the source grid
                self._xmin = self._source_grid.lon.min()
                self._xmax = self._source_grid.lon.max()
                self._ymin = self._source_grid.lat.min()
                self._ymax = self._source_grid.lat.max()
                # setting min and max below is needed for mesh="flat"
                self._zmin = 0.0
                self._zmax = 0.0
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
                # Compute bounding box of each face
                self._xlow = np.min(_xbound, axis=-1)
                self._xhigh = np.max(_xbound, axis=-1)
                self._ylow = np.min(_ybound, axis=-1)
                self._yhigh = np.max(_ybound, axis=-1)
                self._zlow = np.zeros_like(self._xlow)
                self._zhigh = np.zeros_like(self._xlow)

        elif isinstance_noimport(grid, "UxGrid"):
            self._coord_dim = grid.uxgrid.n_max_face_nodes  # Number of barycentric coordinates
            if self._source_grid._mesh == "spherical":
                # Boundaries of the hash grid are the unit cube
                self._xmin = -1.0
                self._ymin = -1.0
                self._zmin = -1.0
                self._xmax = 1.0
                self._ymax = 1.0
                self._zmax = 1.0  # Compute the cell centers of the source grid (for now, assuming Xgrid)
                # Reshape node coordinates to (nfaces, nnodes_per_face)
                nids = self._source_grid.uxgrid.face_node_connectivity.values
                lon = self._source_grid.uxgrid.node_lon.values[nids]
                lat = self._source_grid.uxgrid.node_lat.values[nids]
                x, y, z = _latlon_rad_to_xyz(np.deg2rad(lat), np.deg2rad(lon))
                _xbound, _ybound, _zbound = _latlon_rad_to_xyz(np.deg2rad(lat), np.deg2rad(lon))

                # Compute bounding box of each face
                self._xlow = np.atleast_2d(np.min(_xbound, axis=-1))
                self._xhigh = np.atleast_2d(np.max(_xbound, axis=-1))
                self._ylow = np.atleast_2d(np.min(_ybound, axis=-1))
                self._yhigh = np.atleast_2d(np.max(_ybound, axis=-1))
                self._zlow = np.atleast_2d(np.min(_zbound, axis=-1))
                self._zhigh = np.atleast_2d(np.max(_zbound, axis=-1))

            else:
                # Boundaries of the hash grid are the bounding box of the source grid
                self._xmin = self._source_grid.uxgrid.node_lon.min().values
                self._xmax = self._source_grid.uxgrid.node_lon.max().values
                self._ymin = self._source_grid.uxgrid.node_lat.min().values
                self._ymax = self._source_grid.uxgrid.node_lat.max().values
                # setting min and max below is needed for mesh="flat"
                self._zmin = 0.0
                self._zmax = 0.0
                # Reshape node coordinates to (nfaces, nnodes_per_face)
                nids = self._source_grid.uxgrid.face_node_connectivity.values
                lon = self._source_grid.uxgrid.node_lon.values[nids]
                lat = self._source_grid.uxgrid.node_lat.values[nids]

                # Compute bounding box of each face
                self._xlow = np.atleast_2d(np.min(lon, axis=-1))
                self._xhigh = np.atleast_2d(np.max(lon, axis=-1))
                self._ylow = np.atleast_2d(np.min(lat, axis=-1))
                self._yhigh = np.atleast_2d(np.max(lat, axis=-1))
                self._zlow = np.zeros_like(self._xlow)
                self._zhigh = np.zeros_like(self._xlow)

        # Generate the mapping from the hash indices to unstructured grid elements
        self._hash_table = self._initialize_hash_table()

    def _initialize_hash_table(self):
        """Create a mapping that relates unstructured grid faces to hash indices by determining
        which faces overlap with which hash cells
        """
        # Quantize the bounding box in each direction
        xqlow, yqlow, zqlow = quantize_coordinates(
            self._xlow,
            self._ylow,
            self._zlow,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            self._bitwidth,
        )

        xqhigh, yqhigh, zqhigh = quantize_coordinates(
            self._xhigh,
            self._yhigh,
            self._zhigh,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            self._bitwidth,
        )
        xqlow = xqlow.ravel().astype(np.int32, copy=False)
        yqlow = yqlow.ravel().astype(np.int32, copy=False)
        zqlow = zqlow.ravel().astype(np.int32, copy=False)
        xqhigh = xqhigh.ravel().astype(np.int32, copy=False)
        yqhigh = yqhigh.ravel().astype(np.int32, copy=False)
        zqhigh = zqhigh.ravel().astype(np.int32, copy=False)
        nx = (xqhigh - xqlow + 1).astype(np.int32, copy=False)
        ny = (yqhigh - yqlow + 1).astype(np.int32, copy=False)
        nz = (zqhigh - zqlow + 1).astype(np.int32, copy=False)
        num_hash_per_face = (nx * ny * nz).astype(
            np.int32, copy=False
        )  # Since nx, ny, nz are in the 10-bit range, their product fits in int32
        total_hash_entries = int(num_hash_per_face.sum())

        # Preallocate output arrays
        morton_codes = np.zeros(total_hash_entries, dtype=np.uint32)

        # Compute the j, i indices corresponding to each hash entry
        nface = np.size(self._xlow)
        face_ids = np.repeat(np.arange(nface, dtype=np.int32), num_hash_per_face)
        offsets = np.concatenate(([0], np.cumsum(num_hash_per_face))).astype(np.int32)[:-1]

        valid = num_hash_per_face != 0
        if not np.any(valid):
            # nothing to do
            pass
        else:
            # Grab only valid faces to avoid empty arrays
            nx_v = np.asarray(nx[valid], dtype=np.int32)
            ny_v = np.asarray(ny[valid], dtype=np.int32)
            nz_v = np.asarray(nz[valid], dtype=np.int32)
            xlow_v = np.asarray(xqlow[valid], dtype=np.int32)
            ylow_v = np.asarray(yqlow[valid], dtype=np.int32)
            zlow_v = np.asarray(zqlow[valid], dtype=np.int32)
            starts_v = np.asarray(offsets[valid], dtype=np.int32)

            # Count of elements per valid face (should match num_hash_per_face[valid])
            counts = (nx_v * ny_v * nz_v).astype(np.int32)
            total = int(counts.sum())

            # Map each global element to its face and output position
            start_for_elem = np.repeat(starts_v, counts)  # shape (total,)

            # Intra-face linear index for each element (0..counts_i-1)
            # Offsets per face within the concatenation of valid faces:
            face_starts_local = np.cumsum(np.r_[0, counts[:-1]])
            intra = np.arange(total, dtype=np.int32) - np.repeat(face_starts_local, counts)

            # Derive (zi, yi, xi) from intra using per-face sizes
            ny_nz = np.repeat(ny_v * nz_v, counts)
            nz_rep = np.repeat(nz_v, counts)

            xi = intra // ny_nz
            rem = intra % ny_nz
            yi = rem // nz_rep
            zi = rem % nz_rep

            # Add per-face lows
            x0 = np.repeat(xlow_v, counts)
            y0 = np.repeat(ylow_v, counts)
            z0 = np.repeat(zlow_v, counts)

            xq = x0 + xi
            yq = y0 + yi
            zq = z0 + zi

            # Vectorized morton encode for all elements at once
            codes_all = _encode_quantized_morton3d(xq, yq, zq)

            # Scatter into the preallocated output using computed absolute indices
            out_idx = start_for_elem + intra
            morton_codes[out_idx] = codes_all

        # Sort face indices by morton code
        order = np.argsort(morton_codes)
        morton_codes_sorted = morton_codes[order]
        face_sorted = face_ids[order]
        j_sorted, i_sorted = np.unravel_index(face_sorted, self._xlow.shape)

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

    def query(self, y, x):
        """
        Queries the hash table and finds the closes face in the source grid for each coordinate pair.

        Parameters
        ----------
        y : array_like
            y-coordinates in degrees (lat) to query of shape (N,) where N is the number of queries.
        x : array_like
            x-coordinates in degrees (lon) to query of shape (N,) where N is the number of queries.

        Returns
        -------
        j : ndarray, shape (N,)
            j-indices of the located face in the source grid for each query. If no face was found, GRID_SEARCH_ERROR is returned.
        i : ndarray, shape (N,)
            i-indices of the located face in the source grid for each query. If no face was found, GRID_SEARCH_ERROR is returned.
        coords : ndarray, shape (N, 2)
            The local coordinates (xsi, eta) of the located face in the source grid for each query.
            If no face was found, (-1.0, -1.0)
        """
        keys = self._hash_table["keys"]
        starts = self._hash_table["starts"]
        counts = self._hash_table["counts"]
        i = self._hash_table["i"]
        j = self._hash_table["j"]

        y = np.asarray(y)
        x = np.asarray(x)
        if self._source_grid._mesh == "spherical":
            # Convert coords to Cartesian coordinates (x, y, z)
            lat = np.deg2rad(y)
            lon = np.deg2rad(x)
            qx, qy, qz = _latlon_rad_to_xyz(lat, lon)
        else:
            # For Cartesian grids, use the coordinates directly
            qx = x
            qy = y
            qz = np.zeros_like(qx)

        query_codes = _encode_morton3d(
            qx, qy, qz, self._xmin, self._xmax, self._ymin, self._ymax, self._zmin, self._zmax
        ).ravel()
        num_queries = query_codes.size

        # Locate each query in the unique key array
        pos = np.searchsorted(keys, query_codes)  # pos is shape (num_queries,)

        # Valid hits: inside range with finite query coordinates and query codes give exact morton code match.
        valid = (pos < len(keys)) & np.isfinite(x) & np.isfinite(y)
        # Clip pos to valid range to avoid out-of-bounds indexing
        pos = np.clip(pos, 0, len(keys) - 1)
        # Further filter out false positives from searchsorted by checking for exact code match
        valid[valid] &= query_codes[valid] == keys[pos[valid]]

        # Pre-allocate i and j indices of the best match for each query
        # Default values to -1 (no match case)
        j_best = np.full(num_queries, GRID_SEARCH_ERROR, dtype=np.int32)
        i_best = np.full(num_queries, GRID_SEARCH_ERROR, dtype=np.int32)

        # How many matches each query has; hit_counts[i] is the number of hits for query i
        hit_counts = np.where(valid, counts[pos], 0).astype(np.int32)  # has shape (num_queries,)
        if hit_counts.sum() == 0:
            return (
                j_best.reshape(query_codes.shape),
                i_best.reshape(query_codes.shape),
                np.full((num_queries, self._coord_dim), -1.0, dtype=np.float32),
            )

        # Now, for each query, we need to gather the candidate (j,i) indices from the hash table
        # Each j,i pair needs to be repeated hit_counts[i] times, only when there are hits.

        # Boolean array for keeping track of which queries have candidates
        has_hits = hit_counts > 0  # shape (num_queries,), True for queries that had candidates

        # A quick lookup array that maps all candindates back to its query index
        q_index_for_candidate = np.repeat(
            np.arange(num_queries, dtype=np.int32), hit_counts
        )  # shape (hit_counts.sum(),)
        # Map all candidates to positions in the hash table
        hash_positions = pos[q_index_for_candidate]  # shape (hit_counts.sum(),)

        # Now that we have the positions in the hash table for each table, we can gather the (j,i) pairs for each candidate
        # We do this in a vectorized way by using a CSR-like approach
        # starts[pos[q_index_for_candidate]] gives the starting point in the hash table for each candidate
        # hit_counts gives the number of candidates for each query

        # We need to build an array that gives the offset within each query's candidates
        offsets = np.concatenate(([0], np.cumsum(hit_counts))).astype(np.int32)  # shape (num_queries+1,)
        total = int(offsets[-1])  # total number of candidates across all queries

        # Now, for each candidate, we need a simple array that tells us its "local candidate id" within its query
        # This way, we can easily take the starts[pos[q_index_for_candidate]] and add this local id to get the absolute index
        # We calculate this by computing the "global candidate number" (0..total-1) and subtracting the offsets of the corresponding query
        # This gives us an array that goes from 0..hit_counts[i]-1 for each query i
        intra = np.arange(total, dtype=np.int32) - np.repeat(offsets[:-1], hit_counts)  # shape (hit_counts.sum(),)

        # starts[pos[q_index_for_candidate]] + intra gives a list of positions in the hash table that we can
        # use to quickly gather the (i,j) pairs for each query
        source_idx = starts[hash_positions].astype(np.int32) + intra

        # Gather all candidate (j,i) pairs in one shot
        j_all = j[source_idx]
        i_all = i[source_idx]

        # Now we need to construct arrays that repeats the y and x coordinates for each candidate
        # to enable vectorized point-in-cell checks
        y_rep = np.repeat(y, hit_counts)  # shape (hit_counts.sum(),)
        x_rep = np.repeat(x, hit_counts)  # shape (hit_counts.sum(),)

        # For each query we perform a point in cell check.
        is_in_face, coordinates = self._point_in_cell(self._source_grid, y_rep, x_rep, j_all, i_all)

        coords_best = np.full((num_queries, coordinates.shape[1]), -1.0, dtype=np.float32)

        # For each query that has hits, we need to find the first candidate that was inside the face
        f_indices = np.flatnonzero(is_in_face)  # Indices of all faces that contained the point
        # For each true position, find which query it belongs to by searching offsets
        # Query index q satisfies offsets[q] <= pos < offsets[q+1].
        q = np.searchsorted(offsets[1:], f_indices, side="right")

        uniq_q, q_idx = np.unique(q, return_index=True)
        keep = has_hits[uniq_q]

        if keep.any():
            uniq_q = uniq_q[keep]
            pos_first = f_indices[q_idx[keep]]

            # Directly scatter: the code wants the first True inside each slice
            j_best[uniq_q] = j_all[pos_first]
            i_best[uniq_q] = i_all[pos_first]
            coords_best[uniq_q] = coordinates[pos_first]

        return (
            j_best.reshape(query_codes.shape),
            i_best.reshape(query_codes.shape),
            coords_best.reshape((num_queries, coordinates.shape[1])),
        )


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


def quantize_coordinates(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth=1023):
    """
    Normalize (x, y, z) to [0, 1] over their bounding box, then quantize to 10 bits each (0..1023).

    Parameters
    ----------
    x, y, z : array_like
        Input coordinates to quantize. Can be scalars or arrays (broadcasting applies).
    xmin, xmax : float
        Minimum and maximum bounds for x coordinate.
    ymin, ymax : float
        Minimum and maximum bounds for y coordinate.
    zmin, zmax : float
        Minimum and maximum bounds for z coordinate.

    Returns
    -------
    xq, yq, zq : ndarray, dtype=uint32
        The quantized coordinates, each in range [0, 1023], same shape as the broadcasted input coordinates.
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
    with np.errstate(invalid="ignore"):
        xn = np.where(dx != 0, (x - xmin) / dx, 0.0)
        yn = np.where(dy != 0, (y - ymin) / dy, 0.0)
        zn = np.where(dz != 0, (z - zmin) / dz, 0.0)

    # --- 2) Quantize to (0..bitwidth). ---
    # Multiply by bitwidth, round down, and clip to be safe against slight overshoot.
    xq = np.clip((xn * bitwidth).astype(np.uint32), 0, bitwidth)
    yq = np.clip((yn * bitwidth).astype(np.uint32), 0, bitwidth)
    zq = np.clip((zn * bitwidth).astype(np.uint32), 0, bitwidth)

    return xq, yq, zq


def _encode_quantized_morton3d(xq, yq, zq):
    xq = np.asarray(xq)
    yq = np.asarray(yq)
    zq = np.asarray(zq)

    # --- 3) Bit-dilate each 10-bit number so each bit is separated by two zeros. ---
    # _dilate_bits maps:  b9..b0  ->  b9 0 0 b8 0 0 ... b0 0 0
    dx3 = _dilate_bits(xq).astype(np.uint32)
    dy3 = _dilate_bits(yq).astype(np.uint32)
    dz3 = _dilate_bits(zq).astype(np.uint32)

    # --- 4) Interleave the dilated bits into a single Morton code. ---
    # Bit layout (from LSB upward): x0,y0,z0, x1,y1,z1, ..., x9,y9,z9
    # We shift z's bits by 2, y's by 1, x stays at 0, then OR them together.
    # Cast to a wide type before shifting/OR to be safe when arrays are used.
    code = (dz3 << 2) | (dy3 << 1) | dx3

    # Since our compact type fits in 30 bits, uint32 is enough.
    return code.astype(np.uint32)


def _encode_morton3d(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth=1023):
    """
    Quantize (x, y, z) to 10 bits each (0..1023), dilate the bits so there are
    two zeros between successive bits, and interleave them into a 3D Morton code.

    Parameters
    ----------
    x, y, z : array_like
        Input coordinates to encode. Can be scalars or arrays (broadcasting applies).
    xmin, xmax : float
        Minimum and maximum bounds for x coordinate.
    ymin, ymax : float
        Minimum and maximum bounds for y coordinate.
    zmin, zmax : float
        Minimum and maximum bounds for z coordinate.

    Returns
    -------
    code : ndarray, dtype=uint32
        The resulting Morton codes, same shape as the broadcasted input coordinates.

    Notes
    -----
    - Works with scalars or NumPy arrays (broadcasting applies).
    - Output is up to 30 bits returned as uint32.
    """
    # Convert inputs to ndarray for consistent dtype/ufunc behavior.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    xq, yq, zq = quantize_coordinates(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth)

    # --- 3) Bit-dilate each 10-bit number so each bit is separated by two zeros. ---
    # _dilate_bits maps:  b9..b0  ->  b9 0 0 b8 0 0 ... b0 0 0
    dx3 = _dilate_bits(xq).astype(np.uint32)
    dy3 = _dilate_bits(yq).astype(np.uint32)
    dz3 = _dilate_bits(zq).astype(np.uint32)

    # --- 4) Interleave the dilated bits into a single Morton code. ---
    # Bit layout (from LSB upward): x0,y0,z0, x1,y1,z1, ..., x9,y9,z9
    # We shift z's bits by 2, y's by 1, x stays at 0, then OR them together.
    # Cast to a wide type before shifting/OR to be safe when arrays are used.
    code = (dz3 << 2) | (dy3 << 1) | dx3

    # Since our compact type fits in 30 bits, uint32 is enough.
    return code.astype(np.uint32)
