import pytest
import numpy as np
import sys
from parcels.tools import logger
from parcels.tools import SequentialIdGenerator, SpatialIdGenerator, SpatioTemporalIdGenerator, GenerateID_Service  # noqa

generator_type = ['sequential', 'spatial', 'spatiotemporal']
generators = {'sequential': SequentialIdGenerator,
              'spatial': SpatialIdGenerator,
              'spatiotemporal': SpatioTemporalIdGenerator}


@pytest.mark.parametrize('gentype', generator_type)
def test_idgenerator_initial(gentype):
    if sys.platform == 'win32' and gentype in ['spatial', 'spatiotemporal']:
        logger.warning("Not testing ID-generator type '{}' on Win32 as requested ID-map memory sizes, which is at 7.91GB, exceeds the GitHub quota.".format(gentype))
        return 0
    generator = generators[gentype]()
    positions = [(1.0, 0., 0., 0.),
                 (180.0, 0., 0., 0.),
                 (-1.0, 0., 0., 1.0),
                 (0., 0., 0., 1.0),
                 (0., 0., 0., 0.0)]
    pos_ref = np.array(positions)
    ids = []
    for pos in positions:
        id = generator.getID(pos[0], pos[1], pos[2], pos[3])
        ids.append(id)
    uids = np.unique(ids)
    assert np.all(np.array(positions) == pos_ref)
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


@pytest.mark.parametrize('gentype', generator_type)
@pytest.mark.parametrize('depth_bound', [(0., 1.0), (0, 25.), (-25.0, 0.), (0., 1000.0)])
@pytest.mark.parametrize('time_bound', [(0., 1.0), (0., 86400.0), (0., 316224000.0)])
def test_idgenerator_settimedepth(gentype, depth_bound, time_bound):
    if sys.platform == 'win32' and gentype in ['spatial', 'spatiotemporal']:
        logger.warning("Not testing ID-generator type '{}' on Win32 as requested ID-map memory sizes, which is at 7.91GB, exceeds the GitHub quota.".format(gentype))
        return 0
    generator = generators[gentype]()
    generator.setDepthLimits(depth_bound[0], depth_bound[1])
    generator.setTimeLine(time_bound[0], time_bound[1])
    positions = [(1.0, 0., 0., 0.),
                 (180.0, 0., 0., 0.),
                 (-1.0, 0., 0., 1.0),
                 (0., 0., 0., 1.0),
                 (0., 0., 0., 0.0),
                 (-90.0, 0., 0., 1.0),
                 (1.0, 1.0, 0., 0.),
                 (10.5, 12.5, 10, 0),
                 (10.5, 12.5, 10, 64800),
                 (-90.0, 0., 0., 43200.0)]
    ids = []
    for pos in positions:
        id = generator.getID(pos[0], pos[1], pos[2], pos[3])
        ids.append(id)
    uids = np.unique(ids)
    assert np.alltrue([id in uids for id in ids])
    if gentype == 'sequential':
        assert np.alltrue([type(id) in [np.int64, np.uint64] for id in ids])
        assert np.all(ids == np.array(np.arange(start=0, stop=10, step=1), dtype=np.int64))
    elif gentype in ['spatial', 'spatiotemporal']:
        assert ids[5] == min(ids)
        assert ids[1] == max(ids)
        assert ids[4] > ids[5]
        assert ids[4] < ids[1]


@pytest.mark.parametrize('binranges', [(360, 180, 8192), (1000, 1000, 1024), (5000, 4000, 32)])
@pytest.mark.parametrize('depth_bound', [(0., 1.0), (0, 25.), (-25.0, 0.), (0., 1000.0)])
@pytest.mark.parametrize('time_bound', [(0., 1.0), (0., 86400.0), (0., 316224000.0)])
def test_idgenerator_changing_bitallocation(binranges, depth_bound, time_bound):
    if sys.platform == 'win32':
        logger.warning("Not testing ID-generator type 'SpatialIdGenerator' on Win32 as requested ID-map memory sizes, which is at 7.91GB, exceeds the GitHub quota.")
        return 0
    generator = SpatialIdGenerator(binranges[0], binranges[1], binranges[2])
    generator.setDepthLimits(depth_bound[0], depth_bound[1])
    generator.setTimeLine(time_bound[0], time_bound[1])
    positions = [(1.0, 0., 0., 0.),
                 (180.0, 0., 0., 0.),
                 (-1.0, 0., 0., 1.0),
                 (0., 0., 0., 1.0),
                 (0., 0., 0., 0.0),
                 (-90.0, 0., 0., 1.0),
                 (1.0, 1.0, 0., 0.),
                 (10.5, 12.5, 10, 0),
                 (10.5, 12.5, 10, 64800),
                 (-90.0, 0., 0., 43200.0)]
    ids = []
    for pos in positions:
        id = generator.getID(pos[0], pos[1], pos[2], pos[3])
        ids.append(id)
    uids = np.unique(ids)
    assert np.alltrue([id in uids for id in ids])


@pytest.mark.parametrize('gentype', generator_type)
@pytest.mark.parametrize('release_ids', [True, False])
def test_idgenerator_idrelease(gentype, release_ids):
    if sys.platform == 'win32' and gentype in ['spatial', 'spatiotemporal']:
        logger.warning("Not testing ID-generator type '{}' on Win32 as requested ID-map memory sizes, which is at 7.91GB, exceeds the GitHub quota.".format(gentype))
        return 0
    generator = generators[gentype]()
    if release_ids:
        generator.enable_ID_recovery()
    positions = [(-0.1, 0., 0., 0.),
                 (180.0, 0., 0., 0.),
                 (1.0, 0., 0., 1.0),
                 (0., 0., 0., 1.0),
                 (0., 0., 0., 0.0)]
    ids = []
    for pos in positions:
        id = generator.getID(pos[0], pos[1], pos[2], pos[3])
        ids.append(id)
    generator.releaseID(ids[0])
    ids.pop(0)
    ids.append(generator.getID(-0.1, 0., 0., 0.))

    id3 = ids[3]
    id4 = ids[4]
    if release_ids:
        assert id4 < id3
    else:
        assert id3 < id4
    uids = np.unique(ids)
    assert np.alltrue([id in uids for id in ids])


@pytest.mark.parametrize('gentype', generator_type)
def test_idgenerator_service(gentype):
    if sys.platform == 'win32' and gentype in ['spatial', 'spatiotemporal']:
        logger.warning("Not testing ID-generator type '{}' on Win32 as requested ID-map memory sizes, which is at 7.91GB, exceeds the GitHub quota.".format(gentype))
        return 0
    generator = GenerateID_Service(generators[gentype])
    positions = [(1.0, 0., 0., 0.),
                 (180.0, 0., 0., 0.),
                 (-1.0, 0., 0., 1.0),
                 (0., 0., 0., 1.0),
                 (0., 0., 0., 0.0)]
    pos_ref = np.array(positions)
    ids = []
    for pos in positions:
        id = generator.getID(pos[0], pos[1], pos[2], pos[3])
        ids.append(id)
    uids = np.unique(ids)
    assert np.all(np.array(positions) == pos_ref)
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

# TODO (in later PR):test GenerateID_Service with MPI
