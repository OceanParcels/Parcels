from parcels.particle import Particle
from parcels.particlefile import _create_variables_attribute_dict


def test_particlefile_init(): ...


def test_particlefile_init_read_only_store(): ...


def test_particlefile_init_no_zarr_extension(): ...


def test_create_variables_attribute_dict():
    _create_variables_attribute_dict(Particle)
