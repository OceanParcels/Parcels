from typing import Any

import numpy as np
import pandas as pd
import zarr

from parcels._constants import DATATYPES_TO_FILL_VALUES


def initialize_zarr_dataset(  # TODO: Update to work on a list of variables (access to name, dtype, and metadata) instead of metadata and dtypes
    store, *, n_particles: int, chunks: tuple[int, int], metadata: dict[str, Any], dtypes: dict[str, np.dtype]
):
    root = zarr.group(store=store, overwrite=True)

    for key in metadata.keys():
        zarr_array = root.create_dataset(
            key,
            shape=(n_particles, chunks[1]),
            chunks=chunks,
            dtype=dtypes[key],
            fill_value=DATATYPES_TO_FILL_VALUES[dtypes[key]],
        )

        # Add metadata attributes
        zarr_array.attrs.update(metadata[key])
    zarr.consolidate_metadata(store)

    return root


def write_particle_data(root: zarr.hierarchy.Group, particle_data: pd.DataFrame, mask: np.ndarray):
    for key in root:
        arr = root[key]
        obs_to_write = particle_data["obs_written"][mask]
        ids_to_write = particle_data["trajectory"][mask]

        allowed_tries = 2
        while allowed_tries > 0:
            try:
                arr.vindex[ids_to_write, obs_to_write] = particle_data[key][mask]
                break  # Success
            except zarr.errors.BoundsCheckError as e:
                allowed_tries -= 1
                if allowed_tries == 0:
                    raise e
                bump_array_size_by_chunksize(arr, axis=1)

    particle_data.loc[mask, "obs_written"] += 1
    return root


def bump_array_size_by_chunksize(arr: zarr.core.Array, axis: int, consolidate: bool = True):
    store = arr.store
    chunks = arr.chunks

    new_size = list(arr.shape)
    new_size[axis] += chunks[axis]
    arr.resize(new_size)

    if consolidate:
        zarr.consolidate_metadata(store)
    return arr
