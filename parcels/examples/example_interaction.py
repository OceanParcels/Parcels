import numpy as np
import pytest

from datetime import timedelta

from parcels import FieldSet
from parcels import ParticleSet
from parcels import Variable
from parcels import ScipyParticle
from parcels import AdvectionRK4
from parcels import AsymmetricAttraction


@pytest.mark.parametrize('mode', ['scipy'])  # JIT not (yet) supported
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_interaction_example(mode, mesh):
    # Define a fieldset with an easterly flow
    fieldset = FieldSet.from_data({'U': 1, 'V': 0}, {'lon': 0, 'lat': 0}, mesh=mesh)

    runtime = timedelta(days=1)

    # Create custom particle class with extra variable that indicates
    # whether the interaction kernel should be executed on this particle.
    class InteractingParticle(ScipyParticle):
        attractor = Variable('attractor', dtype=np.bool_, to_write=False)

    pset = ParticleSet(fieldset=fieldset, pclass=InteractingParticle,
                       lon=np.zeros(5), lat=np.array([0.0, 1.0, 0.5, -0.75, 0.25]),
                       interaction_distance=2,
                       attractor=np.array([True, False, False, False, False]))

    output_file = pset.ParticleFile(name="InteractingParticles.nc",
                                    outputdt=timedelta(minutes=4))

    pset.execute(pyfunc=AdvectionRK4, pyfunc_inter=AsymmetricAttraction,
                 runtime=runtime, dt=timedelta(minutes=4),
                 output_file=output_file)

    output_file.export()

    # TODO: add asserts later to make this a proper test


if __name__ == "__main__":
    test_interaction_example('scipy', 'flat')