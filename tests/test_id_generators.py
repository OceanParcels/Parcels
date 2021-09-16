import pytest
import numpy as np

from parcels.tools import GenerateID_Service, SequentialIdGenerator, SpatialIdGenerator, SpatioTemporalIdGenerator

generator_type = ['sequential', 'spatial', 'spatiotemporal']
generators = {'sequential': SequentialIdGenerator,
              'spatial': SpatialIdGenerator,
              'spatiotemporal': SpatioTemporalIdGenerator}

# print("====== Test Spatial ID generator ======")
# package_globals.spat_idgen.setDepthLimits(0.0, 75)
# package_globals.spat_idgen.setTimeLine(0.0, 365.0)
# id1 = package_globals.spat_idgen.getID(0.0, 0.0, 20.0, 0.0)
# id2 = package_globals.spat_idgen.getID(0.0, 0.0, 20.0, 0.0)
# id3 = package_globals.spat_idgen.getID(4.895168, 52.370216, 12.0, 0.0)  # Amsterdam
# id4 = package_globals.spat_idgen.getID(-43.172897, -22.906847, 12.0, 0.0)  # Rio de Janeiro
# id5 = package_globals.spat_idgen.getID(0.0, 0.0, 20.0, 0.0)
# package_globals.spat_idgen.releaseID(id5)
# id6 = package_globals.spat_idgen.getID(0.0, 0.0, 20.0, 0.0)
# print("Test-ID 1:         {}".format(numpy.binary_repr(id1, width=64)))
# print("Test-ID 2:         {}".format(numpy.binary_repr(id2, width=64)))
# print("Test-ID 5:         {}".format(numpy.binary_repr(id5, width=64)))
# print("Test-ID 6:         {}".format(numpy.binary_repr(id6, width=64)))
# print("Test-ID Amsterdam: {}".format(numpy.binary_repr(id3, width=64)))
# print("Test-ID Rio:       {}".format(numpy.binary_repr(id4, width=64)))
# print("===========================================================================")

@pytest.mark.parametrize('gentype', generator_type)
def test_spherical_neighbors(gentype):
    generator = generators[gentype]()
    positions = [(1.0, 0., 0., 0.),
           (180.0, 0., 0., 0.),
           (-1.0, 0., 0., 1.0),
           (0., 0., 0., 1.0),
           (0., 0., 0., 0.0)]
    ids = []
    for pos in positions:
        id = generator.getID(pos[0], pos[1], pos[2], pos[3])
        ids.append(id)
    uids = np.unique(ids)
    assert np.alltrue([id in uids for id in ids])
    if gentype == 'sequential':
        assert np.alltrue([type(id) in [np.int64, np.uint64] for id in ids])
        assert ids[0] == 0
        assert ids[0] < ids[1]
        assert ids[1] == 1
        assert ids[1] < ids[2]
        assert ids[2] == 2
        assert ids[2] < ids[3]
        assert ids[3] == 3
    elif gentype == 'spatial':
        assert ids[2] < ids[3]
        assert ids[2] < ids[0]
        assert ids[1] == max(ids)
        assert ids[2] == min(ids)
        assert ids[3] < ids[4]
    elif gentype == 'spatiotemporal':
        assert ids[2] < ids[3]
        assert ids[2] < ids[0]
        assert ids[1] == max(ids)
        assert ids[2] == min(ids)
        assert ids[4] < ids[3]