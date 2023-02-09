import pytest


def test_decaying_moving_eddy():
    from parcels.examples.example_decaying_moving_eddy import main
    main()


def test_moving_eddies():
    from parcels.examples.example_moving_eddies import main
    main(['jit'])


@pytest.mark.xfail(reason="example data not available from within unit-tests")
def test_nemo_curvilinear():
    from parcels.examples.example_nemo_curvilinear import main
    main(['jit'])


def test_peninsula():
    from parcels.examples.example_peninsula import main
    main(['jit'])


def test_radial_rotation():
    from parcels.examples.example_radial_rotation import main
    main()


def test_stommel():
    from parcels.examples.example_stommel import main
    main(['jit'])
