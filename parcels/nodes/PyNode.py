import ctypes
import sys
import os
from parcels.compilation import InterfaceC, GNUCompiler_SS, GNUCompiler_MS  # noqa: F401
from parcels.tools import get_cache_dir, get_package_dir
# from numpy import int32, int64, uint32, uint64
# from parcels.tools import logger

LIB_LOAD_MAX_REPEAT = 10

# ======================================================================================================= #
# filename "PyNode.py" is given because the wrap-compilation of "node.c" and "node.h" will result in      #
# an auto-generated "node.py", which would then clash with this manually-defined superclass that uses it. #
# ======================================================================================================= #


class Node(object):
    prev = None
    next = None
    # id = None  # not required anymore
    idgen = None
    data = None
    registered = False

    def __init__(self, prev=None, next=None, id=None, data=None, c_lib_register=None, idgen=None):
        if prev is not None:
            assert (isinstance(prev, Node))
            # self.prev = prev
            self.set_prev(prev)
        else:
            # self.prev = None
            self.reset_prev()
        if next is not None:
            assert (isinstance(next, Node))
            # self.next = next
            self.set_next(next)
        else:
            # self.next = None
            self.reset_next()
        # if id is not None and (isinstance(id, int) or type(id) in [int32, uint32, int64, uint64]) and (id >= 0):
        #     self.id = id
        # elif id is None:
        #     # TODO: change the depth here to a init-function parameter called "max_depth" (here: geographic depth, not depth cells, not 'tree depth' or so
        #     self.id = idgen.nextID(random.uniform(-180.0, 180.0), random.uniform(-90.0, 90.0), random.uniform(0., 75.0), 0.)
        # else:
        #     self.id = None

        # self.data = data
        if data is not None:
            self.set_data(data)
        else:
            self.reset_data()

        self.link()

        assert idgen is not None, "Using Nodes requires to specify an ID generator (in order to release the ID on delete)."
        self.idgen = idgen
        self.registered = True

    def __deepcopy__(self, memodict={}):
        result = type(self)(prev=None, next=None, id=-1, data=None)
        result.registered = True
        # result.id = self.id
        result.next = self.next
        result.prev = self.prev
        result.data = self.data
        return result

    def __del__(self):
        if self.data is not None:
            try:
                self.idgen.releaseID(self.data.id)
            except:
                pass
        del self.data
        # idgen.releaseID(self.id)
        self.unlink()
        self.reset_data()

    def link(self):
        # if not self.registered:
        #     return
        if self.prev is not None and self.prev.next != self:
            self.prev.set_next(self)
        if self.next is not None and self.next.prev != self:
            self.next.set_prev(self)

    def unlink(self):
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
        # self.reset_data()
        self.registered = False

    def is_valid(self):
        """
        Function is required as Nodes can be unlinked (i.e. not having data, next- and previous links)
        but still part of a list or other collection, not being called on __del__()
        """
        result = True
        result &= ((self.next is not None) or (self.prev is not None))
        # result &= (self.data is not None)
        result |= (self.data is not None)
        return result

    def __iter__(self):
        return self

    def __next__(self):
        # ==== we need to skip here deleted nodes that have been queued for deletion, but are still bound in memory ==== #
        next_node = self.next
        while next_node is not None and not next_node.is_valid():
            next_node = next_node.next
        if next_node is None:
            raise StopIteration
        return next_node

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        # if (self.data is not None) and (other.data is not None):
        #     return self.data == other.data
        # else:
        #     return self.id == other.id
        # if (self.data is not None) and (other.data is not None):
        # if self.is_valid() and other.is_valid():
        if (self.data is not None) and (other.data is not None):
            result = (self.data.id == other.data.id)
            # result |=  (self.data == other.data)
            return result

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        # return self.id < other.id
        return self.data.id < other.data.id
        # if self.is_valid() and other.is_valid():
        #     return self.data.id < other.data.id
        # elif not self.is_valid() and other.is_valid():
        #     return False
        # elif self.is_valid() and not other.is_valid():
        #     return True
        # return False

    def __le__(self, other):
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        # return self.id <= other.id
        return self.data.id <= other.data.id
        # if self.is_valid() and other.is_valid():
        #     return self.data.id <= other.data.id
        # elif not self.is_valid() and other.is_valid():
        #     return False
        # elif self.is_valid() and not other.is_valid():
        #     return True
        # return False

    def __gt__(self, other):
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        # return self.id > other.id
        return self.data.id > other.data.id
        # if self.is_valid() and other.is_valid():
        #     return self.data.id > other.data.id
        # elif not self.is_valid() and other.is_valid():
        #     return False
        # elif self.is_valid() and not other.is_valid():
        #     return True
        # return False

    def __ge__(self, other):
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        # return self.id >= other.id
        return self.data.id >= other.data.id
        # if self.is_valid() and other.is_valid():
        #     return self.data.id >= other.data.id
        # elif not self.is_valid() and other.is_valid():
        #     return False
        # elif self.is_valid() and not other.is_valid():
        #     return True
        # return False

    def __repr__(self):
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def __str__(self):
        return "Node(p: {}, n: {}, id: {}, d: {})".format(repr(self.prev), repr(self.next), self.data.id, repr(self.data))

    def __sizeof__(self):
        obj_size = sys.getsizeof(object)+sys.getsizeof(object)  # +sys.getsizeof(self.id)
        if self.data is not None:
            obj_size += sys.getsizeof(self.data)
        return obj_size

    def set_prev(self, prev):
        self.prev = prev

    def set_next(self, next):
        self.next = next

    def set_data(self, data):
        self.data = data

    def reset_data(self):
        self.data = None

    def reset_prev(self):
        self.prev = None

    def reset_next(self):
        self.next = None


class NodeJIT(Node, ctypes.Structure):
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
        # super().__init__(prev=prev, next=next, id=id, data=data, idgen=idgen)
        super().__init__(prev=None, next=None, id=id, data=None, idgen=idgen)
        libname = "node"
        if not c_lib_register.is_created(libname) or not c_lib_register.is_compiled(libname) or not c_lib_register.is_loaded(libname):
            cppargs = []
            src_dir = os.path.dirname(os.path.abspath(__file__))
            ccompiler = GNUCompiler_SS(cppargs=cppargs, incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
            c_lib_register.add_entry(libname, InterfaceC(libname, ccompiler, src_dir))
            c_lib_register.load(libname, src_dir=src_dir)
        c_lib_register.register(libname, close_callback=self.close_c_funcs)
        self.c_lib_register_ref = c_lib_register
        self.registered = True
        parent_c_interface = self.c_lib_register_ref.get(libname)  # ["node"]

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
            # self.set_prev_ptr_c(self, self.prev)
            self.set_prev(prev)
        else:
            # self.reset_prev_ptr_c(self)
            self.reset_prev()

        if next is not None and isinstance(next, NodeJIT):
            # self.set_next_ptr_c(self, self.next)
            self.set_next(next)
        else:
            # self.reset_next_ptr_c(self)
            self.reset_next()

        if data is not None:
            # try:
            #     self.set_data_ptr_c(self, self.data.cdata())
            # except AttributeError:
            #     logger.warn("Node's data container casting error - output of data.cdata(): {}".format(self.data.cdata()))
            #     self.set_data_ptr_c(self, ctypes.cast(self.data, ctypes.c_void_p))
            self.set_data(data)
        else:
            # self.reset_data_ptr_c(self)
            self.reset_data()

        self.link()

    def __deepcopy__(self, memodict={}):
        result = type(self)(prev=None, next=None, id=-1, data=None)
        # result.id = self.id
        # result.next = self.next
        # result.prev = self.prev
        # result.data = self.data
        if self.c_lib_register_ref is not None:  # and self.registered:
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

        # if result.prev is not None and isinstance(result.prev, NodeJIT):
        #     result.set_prev_ptr_c(result, result.prev)
        # else:
        #     result.reset_prev_ptr_c(result)
        # if result.next is not None and isinstance(result.next, NodeJIT):
        #     result.set_next_ptr_c(result, result.next)
        # else:
        #     result.reset_next_ptr_c(result)
        result.set_prev(self.prev)
        result.set_next(self.next)

        # if result.data is not None:
        #     result.set_data_ptr_c(result, ctypes.cast(result.data, ctypes.c_void_p))
        # else:
        #     result.reset_data_ptr_c(result)
        result.set_data(self.data)
        return result

    def __del__(self):
        # nid = -1
        # if self.data is not None:
        #     nid = self.data.id
        # logger.info("NodeJIT.del() {} is called.".format(nid))
        # self.unlink()

        # if self.data is not None:
        #     try:
        #         self.idgen.releaseID(self.data.id)
        #     except:
        #         pass
        # del self.data
        super(NodeJIT, self).__del__()
        # logger.info("NodeJIT {} deleted.".format(nid))

    def link(self):
        if not self.registered or self.c_lib_register_ref is None:
            return
        # if self.prev is not None and self.prev.next != self:
        #     self.prev.set_next(self)
        # if self.next is not None and self.next.prev != self:
        #     self.next.set_prev(self)
        super(NodeJIT, self).link()

    def unlink(self):
        # nid = -1
        # if self.data is not None:
        #     nid = self.data.id
        # logger.info("NodeJIT.unlink() {} is called.".format(nid))
        super(NodeJIT, self).unlink()

        # if self.prev is not None:
        #     if self.next is not None:
        #         # self.prev.set_next(self.next)
        #         self.prev.set_next_ptr_c(self.next)
        #     else:
        #         # self.prev.reset_next()
        #         self.prev.reset_next_ptr_c()
        # if self.next is not None:
        #     if self.prev is not None:
        #         # self.next.set_prev(self.prev)
        #         self.next.set_prev_ptr_c(self.prev)
        #     else:
        #         # self.next.reset_prev()
        #         self.next.reset_prev_ptr_c()

        # # self.reset_prev()
        # self.reset_prev_ptr_c()
        # # self.reset_next()
        # self.reset_next_ptr_c()
        # # self.reset_data()
        # self.reset_data_ptr_c()
        if self.c_lib_register_ref is not None:  # and self.registered:
            self.unlink_c_functions()
            self.c_lib_register_ref.deregister("node")
        # self.registered = False
        self.c_lib_register_ref = None
        # self.prev = None
        # self.next = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def __sizeof__(self):
        return super().__sizeof__()+sys.getsizeof(self._fields_)

    def __eq__(self, other):
        return super().__eq__(other)

    def __ne__(self, other):
        return super().__ne__(other)

    def __lt__(self, other):
        return super().__lt__(other)

    def __le__(self, other):
        return super().__le__(other)

    def __gt__(self, other):
        return super().__gt__(other)

    def close_c_funcs(self):
        # logger.info("NodeJIT.close_c_func() called.")
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
        self.init_node_c = None
        self.set_prev_ptr_c = None
        self.set_next_ptr_c = None
        self.set_data_ptr_c = None
        self.reset_prev_ptr_c = None
        self.reset_next_ptr_c = None
        self.reset_data_ptr_c = None

    def link_c_functions(self, c_func_dict):
        self.init_node_c = c_func_dict['init_node']
        self.set_prev_ptr_c = c_func_dict['set_prev_ptr']
        self.set_next_ptr_c = c_func_dict['set_next_ptr']
        self.set_data_ptr_c = c_func_dict['set_data_ptr']
        self.reset_prev_ptr_c = c_func_dict['reset_prev_ptr']
        self.reset_next_ptr_c = c_func_dict['reset_next_ptr']
        self.reset_data_ptr_c = c_func_dict['reset_data_ptr']

    def set_data(self, data):
        # logger.info("NodeJIT.set_data() is called.")
        super().set_data(data)
        if self.registered:
            self.update_data()

    def set_prev(self, prev):
        # logger.info("NodeJIT.set_prev() is called.")
        super().set_prev(prev)
        if self.registered:
            self.update_prev()

    def set_next(self, next):
        # logger.info("NodeJIT.set_next() is called.")
        super().set_next(next)
        if self.registered:
            self.update_next()

    def reset_data(self):
        # logger.info("NodeJIT.reset_data() is called.")
        super().reset_data()
        if self.registered and self.reset_data_ptr_c is not None:
            self.reset_data_ptr_c(self)

    def reset_prev(self):
        # logger.info("NodeJIT.reset_prev() is called.")
        super().reset_prev()
        if self.registered and self.reset_prev_ptr_c is not None:
            self.reset_prev_ptr_c(self)

    def reset_next(self):
        # logger.info("NodeJIT.reset_next() is called.")
        super().reset_next()
        if self.registered and self.reset_next_ptr_c is not None:
            self.reset_next_ptr_c(self)

    def update_prev(self):
        if self.set_prev_ptr_c is None or self.reset_prev_ptr_c is None:
            return
        if self.prev is not None and isinstance(self.prev, NodeJIT):
            self.set_prev_ptr_c(self, self.prev)
        else:
            self.reset_prev_ptr_c(self)

    def update_next(self):
        if self.set_next_ptr_c is None or self.reset_next_ptr_c is None:
            return
        if self.next is not None and isinstance(self.next, NodeJIT):
            self.set_next_ptr_c(self, self.next)
        else:
            self.reset_next_ptr_c(self)

    def update_data(self):
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
