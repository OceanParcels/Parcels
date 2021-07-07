from parcels.nodes.PyNode import *
from sortedcontainers import SortedList
from numpy import int32, int64, uint32, uint64
from copy import deepcopy
from parcels.tools import logger
import gc

class DoubleLinkedNodeList(SortedList):
    dtype = None
    _c_lib_register = None

    def __init__(self, iterable=None, dtype=Node, c_lib_register=None):
        super(DoubleLinkedNodeList, self).__init__(iterable)
        self.dtype = dtype
        self._c_lib_register = c_lib_register

    def __del__(self):
        # logger.info("DoubleLinkedNodeList.del() called.")
        self.clear()
        # super(DoubleLinkedNodeList, self).__del__()

    def clear(self):
        """Remove all the elements from the list."""
        n = self.__len__()
        # logger.info("DoubleLinkedNodeList.clear() - # remaining items: {}".format(n))
        if n > 0:
            # logger.info("Deleting {} elements ...".format(n))
            while (n > 0):
                # val = self.pop(); del val
                # super().__delitem__(n-1)
                # self.pop()

                self.__getitem__(-1).unlink()
                # self.__delitem__(-1)
                del self[-1]
                n = self.__len__()
                # logger.info("Deleting {} elements ...".format(n))
        gc.collect()
        # logger.info("DoubleLinkedNodeList.clear() - list empty.")
        super()._clear()

    def __new__(cls, iterable=None, key=None, load=1000, dtype=Node, c_lib_register=None):
        return object.__new__(cls)

    def add(self, val):
        assert type(val) == self.dtype
        if isinstance(val, Node):
            self._add_by_node(val)
        elif isinstance(val, int) or type(val) in [int32, uint32, int64, uint64]:
            self._add_by_id(val)
        else:
            self._add_by_pdata(val)

    def _add_by_node(self, val):
        n = self.__len__()
        index = self.bisect_right(val)
        if index < n:
            next_node = self.__getitem__(index)
        else:
            next_node = None
        if index > 0:
            prev_node = self.__getitem__(index - 1)
        else:
            prev_node = None
        if next_node is not None:
            next_node.set_prev(val)
            val.set_next(next_node)
        if prev_node is not None:
            prev_node.set_next(val)
            val.set_prev(prev_node)
        super().add(val)

    def _add_by_id(self, val):
        n = self.__len__()
        index = self.bisect_right(val)
        if index < n:
            next_node = self.__getitem__(index)
        else:
            next_node = None
        if index > 0:
            prev_node = self.__getitem__(index - 1)
        else:
            prev_node = None
        new_node = self.dtype(prev=prev_node, next=next_node, id=val, c_lib_register=self._c_lib_register)
        super().add(new_node)

    def _add_by_pdata(self, val):
        new_node = self.dtype(data=val, c_lib_register=self._c_lib_register)
        super().add(new_node)
        n = self.__len__()
        index = self.index(new_node)
        if index < (n - 2):
            next_node = self.__getitem__(index + 1)
        else:
            next_node = None
        if index > 0:
            prev_node = self.__getitem__(index - 1)
        else:
            prev_node = None
        if next_node is not None:
            next_node.set_prev(new_node)
            new_node.set_next(next_node)
        if prev_node is not None:
            prev_node.set_next(new_node)
            new_node.set_prev(prev_node)

    def append(self, val):
        self.add(val)

    def pop(self, idx=-1, deepcopy_elem=False):
        """
        Because we expect the return node to be of use,
        the actual node is NOT physically deleted (by calling
        the destructor). pop() only dereferences the object
        in the list. The parameter 'deepcopy_elem' can be set so as
        to physically delete the list object and return a deep copy (unlisted).
        :param idx:
        :param deepcopy_elem:
        :return:
        """
        if deepcopy_elem:
            val = super().pop(idx)
            result = deepcopy(val)
            # result = deepcopy(self.__getitem__(idx))
            del val
            return result
        return super().pop(idx)
