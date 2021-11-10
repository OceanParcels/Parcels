from parcels import (ScipyParticle, JITParticle, Node, NodeJIT, DoubleLinkedNodeList)
from parcels import GenerateID_Service, SequentialIdGenerator, LibraryRegisterC  # noqa
import pytest
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
ntype = {'scipy': Node, 'jit': NodeJIT}


@pytest.fixture(name="c_lib_register")
def c_lib_register_fixture():
    c_lib_register = LibraryRegisterC()
    return c_lib_register


@pytest.fixture(name="idgen")
def idgen_fixture():
    idgen = GenerateID_Service(SequentialIdGenerator)
    idgen.setDepthLimits(0., 1.0)
    idgen.setTimeLine(0.0, 1.0)
    return idgen


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_create_nodes(mode, c_lib_register, idgen):
    nodevalue = ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen)
    return nodevalue


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_create_list_nodes(mode, c_lib_register, idgen):
    nlist = []
    n1 = ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen)
    nlist.append(n1)
    n2 = ntype[mode](prev=n1, data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen)
    nlist.append(n2)
    n3 = ntype[mode](prev=n2, data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen)
    nlist.append(n3)
    assert n2.prev == n1
    assert n2.next == n3
    del nlist[:]


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_create_nodelist(mode, c_lib_register, idgen):
    nlist = DoubleLinkedNodeList(dtype=ntype[mode], c_lib_register=c_lib_register)
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    n1 = nlist[0]
    n2 = nlist[1]
    n3 = nlist[2]
    assert n2.prev == n1
    assert n2.next == n3
    assert len(nlist) == 3
    del nlist


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_remove_nodelist(mode, c_lib_register, idgen):
    nlist = DoubleLinkedNodeList(dtype=ntype[mode], c_lib_register=c_lib_register)
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    n1 = nlist[0]
    n2 = nlist[1]
    n3 = nlist[2]
    n2.unlink()
    nlist.remove(n2)
    assert n3.prev == n1
    assert n1.next == n3
    del nlist


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_clear_nodelist(mode, c_lib_register, idgen):
    nlist = DoubleLinkedNodeList(dtype=ntype[mode], c_lib_register=c_lib_register)
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    nlist.add(ntype[mode](data=ptype[mode](lon=0, lat=0, pid=idgen.getID(0, 0, 0, 0)), c_lib_register=c_lib_register, idgen=idgen))
    n1 = nlist[0]
    n2 = nlist[1]
    n3 = nlist[2]
    assert n2.prev == n1
    assert n2.next == n3
    nlist.clear()
    assert len(nlist) == 0
    del nlist
