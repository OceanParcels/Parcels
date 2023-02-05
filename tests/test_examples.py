from os import system
import pytest


@pytest.mark.parametrize('example', ['brownian', 'decaying_moving_eddy', 'moving_eddies', 'nemo_curvilinear',
                                     'peninsula', 'radial_rotation', 'stommel'])
def test_mains(example):
    system(f'python parcels/examples/example_{example}.py')
