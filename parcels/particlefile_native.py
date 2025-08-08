import numpy as np
import zarr

from parcels._constants import DATATYPES_TO_FILL_VALUES
from parcels.particle import Variable


def initialize_zarr_dataset(store, *, n_particles: int, variables: list[Variable], chunks: tuple[int, int]):
    root = zarr.group(store=store, overwrite=True)

    for var in variables:
        assert var.to_write is not False
        zarr_array = root.create_dataset(
            var.name,
            shape=(n_particles, chunks[1]),
            chunks=chunks,
            dtype=var.dtype,
            fill_value=DATATYPES_TO_FILL_VALUES[var.dtype],
        )

        # Add metadata
        zarr_array.attrs.update(
            {
                "_ARRAY_DIMENSIONS": ["trajectory", "obs"],
                **var.attrs,
            }
        )
    zarr.consolidate_metadata(store)

    return root


def write_particle_data(root: zarr.hierarchy.Group, particle_data: dict[str, np.ndarray], mask: np.ndarray):
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

    particle_data["obs_written"][mask] += 1
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
