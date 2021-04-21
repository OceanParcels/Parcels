import numpy as np
import pytest

from datetime import timedelta

from parcels import FieldSet
from parcels import ParticleSet
from parcels import Variable
from parcels import ScipyParticle
from parcels import AdvectionRK4
from parcels import NearestNeighbourWithinRange
from parcels import MergeWithNearestNeighbour


@pytest.mark.parametrize('mode', ['scipy'])  # JIT not (yet) supported
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_merge_example(mode, mesh, npart):
    # Define a fieldset with an easterly flow
    fieldset = FieldSet.from_data({'U': 1, 'V': 0}, {'lon': 0, 'lat': 0}, mesh=mesh)

    runtime = timedelta(days=1)

    # Create custom particle class with extra variables that indicate
    # mass and nearest neighbour.
    class MergeParticle(ScipyParticle):
        nearest_neighbour = Variable('nearest_neighbour', dtype=np.int64, to_write=False)
        mass = Variable('mass', dtype=np.float32)

    pset = ParticleSet(fieldset=fieldset, pclass=MergeParticle,
                       lon=np.random.uniform(low=-2, high=2, size=(npart,)),
                       lat=np.random.uniform(low=-2, high=2, size=(npart,)),
                       interaction_distance=2,
                       mass=np.ones(npart))

    output_file = pset.ParticleFile(name="MergeParticles.nc",
                                    outputdt=timedelta(hours=1))

    pset.execute(pyfunc=AdvectionRK4,
                 pyfunc_inter=pset.InteractionKernel(NearestNeighbourWithinRange) + MergeWithNearestNeighbour,
                 runtime=runtime, dt=timedelta(hours=1),
                 output_file=output_file)

    output_file.export()

    # TODO: add asserts later to make this a proper test


if __name__ == "__main__":
    test_merge_example('scipy', 'flat', 10)