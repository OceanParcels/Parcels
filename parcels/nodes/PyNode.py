import ctypes
import sys
import os
from parcels.compilation import InterfaceC, GNUCompiler_SS, GNUCompiler_MS  # noqa: F401
from parcels.tools import get_cache_dir, get_package_dir

# ======================================================================================================= #
# Due to a delay in compilation and loading of the C-library, it is possible to request the creation      #
# of a node while the C-library is still not ready. Therefore, when creating the first node, we need to   #
# continuously trial-and-error link the node to the C-library. After a certain max. number of tries,      #
# we can conclude that the C-library has not been created (either because it the nodelist was used        #
# inappropriately or because compilation went wrong), so after this max. number of tries, we throw an     #
# error and abort further processing.                                                                     #
# ======================================================================================================= #
LIB_LOAD_MAX_REPEAT = 10

# ======================================================================================================= #
# filename "PyNode.py" is given because the wrap-compilation of "node.c" and "node.h" will result in      #
# an auto-generated "node.py", which would then clash with this manually-defined superclass that uses it. #
# ======================================================================================================= #


class Node(object):
    """
    A simple nodal object, storing its connectivity to its predecessor and its successor, and (a reference to) its object (as payload)
    """
    prev = None
    next = None
    idgen = None
    data = None
    registered = False

    def __init__(self, prev=None, next=None, id=None, data=None, c_lib_register=None, idgen=None):
        """
        Node - Constructor
        :arg prev: predecessor Node object
        :arg next: successor Node object
        :arg id: optional, legacy parameter to set this object's ID
        :arg data: data payload of this Node
        :arg c_lib_register: a LibraryRegisterC object; available for compliance reasons with the NodeJIT sibling class
        :arg idgen: an ID generator object used for ID registration- and de-registration for the Particle data payload
        """
        if prev is not None:
            assert (isinstance(prev, Node))
            self.set_prev(prev)
        else:
            self.reset_prev()
        if next is not None:
            assert (isinstance(next, Node))
            self.set_next(next)
        else:
            self.reset_next()
        if data is not None:
            self.set_data(data)
        else:
            self.reset_data()
        self.link()

        assert idgen is not None, "Using Nodes requires to specify an ID generator (in order to release the ID on delete). See https://github.com/OceanParcels/parcels/tree/2.x/tests/test_nodes.py"
        self.idgen = idgen
        self.registered = True

    def __del__(self):
        """
        Node - Destructor
        """
        if self.data is not None:
            try:
                self.idgen.releaseID(self.data.id)
            except:
                pass
        del self.data
        self.unlink()
        self.reset_data()

    def __deepcopy__(self, memodict={}):
        """
        :returns a deepcopy of this very object
        """
        result = type(self)(prev=None, next=None, id=-1, data=None)
        result.registered = True
        result.next = self.next
        result.prev = self.prev
        result.data = self.data
        return result

    def link(self):
        """
        links this node to its neighbours, i.e. sets this object to be the successor of its predecessor and
        sets this object to be the predecessor of its successor, so that we have a mutual connection in the form of:
        prev <-> this <-> next
        """
        if self.prev is not None and self.prev.next != self:
            self.prev.set_next(self)
        if self.next is not None and self.next.prev != self:
            self.next.set_prev(self)

    def unlink(self):
        """
        removes this object from the linked chain of nodes, i.e. we set this predecessor's successor to be this successor
        and set this successor's predecessor to this predecessor, so that the new connection is in the form of:
        prev <-> next
        """
        if self.registered:
            if self.prev is not None:
                if self.next is not None:
                    self.prev.set_next(self.next)
                else:
                    self.prev.reset_next()
            if self.next is not None:
                if self.prev is not None:
                    self.next.set_prev(self.prev)
                else:
                    self.next.reset_prev()
        self.reset_prev()
        self.reset_next()
        self.registered = False

    def isvalid(self):
        """
        Function is required as Nodes can be unlinked (i.e. not having data, next- and previous links)
        but still part of a list or other collection, not being called on __del__()
        :returns if a node is still valid (True) or invalid (False)
        """
        result = True
        result &= ((self.next is not None) or (self.prev is not None))
        result |= (self.data is not None)
        return result

    def __iter__(self):
        """
        :returns forward iterator through the interconnected chain of double-linked nodes fromout this node
        """
        return self

    def __next__(self):
        """
        :returns next node in the interconnected chain of double-linked nodes
        """
        # ==== we need to skip here deleted nodes that have been queued for deletion, but are still bound in memory ==== #
        next_node = self.next
        while next_node is not None and not next_node.isvalid():
            next_node = next_node.next
        if next_node is None:
            raise StopIteration
        return next_node

    def __eq__(self, other):
        """
        :arg other: another Node object
        :returns boolean if :arg other and this object are equal
        """
        if type(self) is not type(other):
            return False
        if (self.data is not None) and (other.data is not None):
            result = (self.data.id == other.data.id)
            return result

    def __ne__(self, other):
        """
        :arg other: another Node object
        :returns boolean if :arg other and this object are not equal
        """
        return not (self == other)

    def __lt__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered before the position of this
        """
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.data.id < other.data.id

    def __le__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered before-or-at the position of this
        """
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.data.id <= other.data.id

    def __gt__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered after the position of this
        """
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.data.id > other.data.id

    def __ge__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered after-or-at the position of this
        """
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.data.id >= other.data.id

    def __repr__(self):
        """
        :returns a byte-like representation of a Node
        """
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def __str__(self):
        """
        returns a text-like representation of a Node
        """
        return "Node(prev: {}, next: {}, id: {}, data: {})".format(repr(self.prev), repr(self.next), self.data.id, repr(self.data))

    def __sizeof__(self):
        """
        :returns the byte size of this object, INCLUDING the size of its containing object
        """
        obj_size = sys.getsizeof(object)+sys.getsizeof(object)
        if self.data is not None:
            obj_size += sys.getsizeof(self.data)
        return obj_size

    def set_prev(self, prev):
        """
        (mandatory) setter-function for the previous Node object
        """
        self.prev = prev

    def set_next(self, next):
        """
        (mandatory) setter-function for the next Node object
        """
        self.next = next

    def set_data(self, data):
        """
        (mandatory) setter-function for the data payload of this Node
        """
        self.data = data

    def reset_data(self):
        """
        this function resets (i.e. nullifies / nonifies) the the Node's data payload
        """
        self.data = None

    def reset_prev(self):
        """
        this function resets (i.e. nullifies / nonifies) the the Node's predecessor
        """
        self.prev = None

    def reset_next(self):
        """
        this function resets (i.e. nullifies / nonifies) the the Node's successor
        """
        self.next = None


class NodeJIT(Node, ctypes.Structure):
    """
    A nodal object, storing its connectivity to its predecessor and its successor, and (a reference to) its object (as payload).
    This type is connected to a direct ctypes representation, which can be used in ctypes-bound JIT functions.
    """
    _fields_ = [('_c_prev_p', ctypes.c_void_p),
                ('_c_next_p', ctypes.c_void_p),
                ('_c_data_p', ctypes.c_void_p),
                ('_c_pu_affinity', ctypes.c_int)]

    init_node_c = None
    set_prev_ptr_c = None
    set_next_ptr_c = None
    set_data_ptr_c = None
    reset_prev_ptr_c = None
    reset_next_ptr_c = None
    reset_data_ptr_c = None
    c_lib_register_ref = None

    def __init__(self, prev=None, next=None, id=None, data=None, c_lib_register=None, idgen=None):
        """
        NodeJIT - Constructor
        :arg prev: predecessor Node object
        :arg next: successor Node object
        :arg id: optional, legacy parameter to set this object's ID
        :arg data: data payload of this Node
        :arg c_lib_register: a LibraryRegisterC object, used to register this object to the ctypes JIT C-library
        :arg idgen: an ID generator object used for ID registration- and de-registration for the Particle data payload
        """
        super().__init__(prev=None, next=None, id=id, data=None, idgen=idgen)
        libname = "node"
        if not c_lib_register.iscreated(libname) or not c_lib_register.iscompiled(libname) or not c_lib_register.isloaded(libname):
            cppargs = []
            src_dir = os.path.dirname(os.path.abspath(__file__))
            ccompiler = GNUCompiler_SS(cppargs=cppargs, incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
            c_lib_register.add_entry(libname, InterfaceC(libname, ccompiler, src_dir))
            c_lib_register.load(libname)
        c_lib_register.register(libname, close_callback=self.close_c_funcs)
        self.c_lib_register_ref = c_lib_register
        self.registered = True
        parent_c_interface = self.c_lib_register_ref.get(libname)

        c_funcs = None
        repeat_load_iteration = 0
        while c_funcs is None and repeat_load_iteration < LIB_LOAD_MAX_REPEAT:
            c_funcs = parent_c_interface.load_functions(NodeJIT_func_params())
            repeat_load_iteration += 1
            if c_funcs is None:
                c_lib_register.deregister(libname=libname)
                c_lib_register.unload(libname=libname)
                c_lib_register.load(libname)
                c_lib_register.register(libname, close_callback=self.close_c_funcs)
        assert c_funcs is not None, "Loading 'node' library failed."
        self.link_c_functions(c_funcs)
        self.init_node_c(self)

        if prev is not None and isinstance(prev, NodeJIT):
            self.set_prev(prev)
        else:
            self.reset_prev()
        if next is not None and isinstance(next, NodeJIT):
            self.set_next(next)
        else:
            self.reset_next()

        if data is not None:
            self.set_data(data)
        else:
            self.reset_data()
        self.link()

    def __del__(self):
        """
        NodeJIT - Destructor
        """
        super(NodeJIT, self).__del__()

    def __deepcopy__(self, memodict={}):
        """
        :returns a deepcopy of this very object
        The deepcopy operation includes individual registration of the object with the LibraryRegisterC interface.
        """
        result = type(self)(prev=None, next=None, id=-1, data=None)
        if self.c_lib_register_ref is not None:
            libname = "node"
            result.c_lib_register_ref = self.c_lib_register_ref
            result.c_lib_register_ref.register(libname, close_callback=self.close_c_funcs)
            result.registered = True
            parent_c_interface = self.c_lib_register_ref.get(libname)
            c_funcs = None
            repeat_load_iteration = 0
            while c_funcs is None and repeat_load_iteration < LIB_LOAD_MAX_REPEAT:
                c_funcs = parent_c_interface.load_functions(NodeJIT_func_params())
                repeat_load_iteration += 1
            assert c_funcs is not None, "Loading 'node' library failed."
            result.link_c_functions(c_funcs)
            result.init_node_c(self)

        result.set_prev(self.prev)
        result.set_next(self.next)
        result.set_data(self.data)
        return result

    def link(self):
        """
        links this node to its neighbours, i.e. sets this object to be the successor of its predecessor and
        sets this object to be the predecessor of its successor, so that we have a mutual connection in the form of:
        prev <-> this <-> next

        In NodeJIT, this linking requires a C-interface, and hence does not execute without the object being registered
        in C.
        """
        if not self.registered or self.c_lib_register_ref is None:
            return
        super(NodeJIT, self).link()

    def unlink(self):
        """
        removes this object from the linked chain of nodes, i.e. we set this predecessor's successor to be this successor
        and set this successor's predecessor to this predecessor, so that the new connection is in the form of:
        prev <-> next

        In NodeJIT, this unlinking requires a C-interface, and hence does not execute without the object being registered
        in C.
        """
        super(NodeJIT, self).unlink()
        if self.c_lib_register_ref is not None:
            self.unlink_c_functions()
            self.c_lib_register_ref.deregister("node")
        self.c_lib_register_ref = None

    def __repr__(self):
        """
        :returns a byte-like representation of a Node
        """
        return super().__repr__()

    def __str__(self):
        """
        returns a text-like representation of a Node
        """
        return "NodeJIT(p: {}, n: {}, id: {}, d: {})".format(repr(self.prev), repr(self.next), self.data.id, repr(self.data))

    def __sizeof__(self):
        """
        :returns the byte size of this object, INCLUDING the size of its containing object
        """
        return super().__sizeof__()+sys.getsizeof(self._fields_)

    def __eq__(self, other):
        """
        :arg other: another Node object
        :returns boolean if :arg other and this object are equal
        """
        return super().__eq__(other)

    def __ne__(self, other):
        """
        :arg other: another Node object
        :returns boolean if :arg other and this object are not equal
        """
        return super().__ne__(other)

    def __lt__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered before the position of this
        """
        return super().__lt__(other)

    def __le__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered before-or-at the position of this
        """
        return super().__le__(other)

    def __gt__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered after the position of this
        """
        return super().__gt__(other)

    def __ge__(self, other):
        """
        :arg other: another Node object
        :returns boolean, if :arg other is ordered after-or-at the position of this
        """
        return super().__ge__(other)

    def close_c_funcs(self):
        """
        This functions releases the object's links to the C-interface (i.e. deregisters itself in C), and
        deregisters itself from the InterfaceC object of the LibraryRegisterC.
        """
        if self.registered:
            try:
                self.reset_prev_ptr_c()
                self.reset_next_ptr_c()
                self.reset_data_ptr_c()
                self.unlink_c_functions()
                if self.c_lib_register_ref is not None:
                    self.c_lib_register_ref.deregister("node")
            except:
                pass
        self.registered = False
        self.c_lib_register_ref = None

    def unlink_c_functions(self):
        """
        unlinks (i.e. resets / nullifies / nonifies) all C-function member variables.
        """
        self.init_node_c = None
        self.set_prev_ptr_c = None
        self.set_next_ptr_c = None
        self.set_data_ptr_c = None
        self.reset_prev_ptr_c = None
        self.reset_next_ptr_c = None
        self.reset_data_ptr_c = None

    def link_c_functions(self, c_func_dict):
        """
        Links all C-function member variables to their ctypes function interface.
        :arg c_func_dict: dictionary of 'function name' (str) -> CDLL function
        """
        self.init_node_c = c_func_dict['init_node']
        self.set_prev_ptr_c = c_func_dict['set_prev_ptr']
        self.set_next_ptr_c = c_func_dict['set_next_ptr']
        self.set_data_ptr_c = c_func_dict['set_data_ptr']
        self.reset_prev_ptr_c = c_func_dict['reset_prev_ptr']
        self.reset_next_ptr_c = c_func_dict['reset_next_ptr']
        self.reset_data_ptr_c = c_func_dict['reset_data_ptr']

    def set_data(self, data):
        """
        (mandatory) setter-function for the data payload of this Node
        """
        super().set_data(data)
        if self.registered:
            self.update_data()

    def set_prev(self, prev):
        """
        (mandatory) setter-function for the previous Node object
        """
        super().set_prev(prev)
        if self.registered:
            self.update_prev()

    def set_next(self, next):
        """
        (mandatory) setter-function for the next Node object
        """
        super().set_next(next)
        if self.registered:
            self.update_next()

    def reset_data(self):
        """
        this function resets (i.e. nullifies / nonifies) the the Node's data payload
        """
        super().reset_data()
        if self.registered and self.reset_data_ptr_c is not None:
            self.reset_data_ptr_c(self)

    def reset_prev(self):
        """
        this function resets (i.e. nullifies / nonifies) the the Node's predecessor
        """
        super().reset_prev()
        if self.registered and self.reset_prev_ptr_c is not None:
            self.reset_prev_ptr_c(self)

    def reset_next(self):
        """
        this function resets (i.e. nullifies / nonifies) the the Node's successor
        """
        super().reset_next()
        if self.registered and self.reset_next_ptr_c is not None:
            self.reset_next_ptr_c(self)

    def update_prev(self):
        """
        This function checks the variable types of the 'prev' member and then updates the C-pointer of
        this object's prior element using the 'prev' object.
        """
        if self.set_prev_ptr_c is None or self.reset_prev_ptr_c is None:
            return
        if self.prev is not None and isinstance(self.prev, NodeJIT):
            self.set_prev_ptr_c(self, self.prev)
        else:
            self.reset_prev_ptr_c(self)

    def update_next(self):
        """
        This function checks the variable types of the 'next' member and then updates the C-pointer of
        this object's succeeding element using the 'next' object.
        """
        if self.set_next_ptr_c is None or self.reset_next_ptr_c is None:
            return
        if self.next is not None and isinstance(self.next, NodeJIT):
            self.set_next_ptr_c(self, self.next)
        else:
            self.reset_next_ptr_c(self)

    def update_data(self):
        """
        This function checks the variable types of the 'data' member and then updates the C-pointer of
        this object's payload container element using the 'data' object.
        """
        if self.set_data_ptr_c is None or self.reset_data_ptr_c is None:
            return
        if self.data is not None:
            try:
                self.set_data_ptr_c(self, self.data.cdata())
            except AttributeError:
                self.set_data_ptr_c(self, ctypes.cast(self.data, ctypes.c_void_p))
        else:
            self.reset_data_ptr_c(self)


def NodeJIT_func_params():
    """
    This static function
    :returns a list of dictionary records, each entry containg:
     'name' (str)
     'return' (ctypes type of function's return value)
     'arguments' (list of ctypes types of the function's calling arguments)
    """
    return [{"name": 'init_node', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'set_prev_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]},
            {"name": 'set_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]},
            {"name": 'set_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.POINTER(NodeJIT)]},
            {"name": 'set_data_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.c_void_p]},
            {"name": 'set_pu_affinity', "return": None, "arguments": [ctypes.POINTER(NodeJIT), ctypes.c_int]},
            {"name": 'get_pu_affinity', "return": ctypes.c_int, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_prev_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_next_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_data_ptr', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]},
            {"name": 'reset_pu_affinity', "return": None, "arguments": [ctypes.POINTER(NodeJIT)]}]
