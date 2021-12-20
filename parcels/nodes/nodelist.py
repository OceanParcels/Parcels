from parcels.nodes.PyNode import Node
from sortedcontainers import SortedList
from numpy import int32, int64, uint32, uint64
from copy import deepcopy
import gc


class DoubleLinkedNodeList(SortedList):
    """
    This class stores items in a dynamic, double-linked list using the 'sortedcollections.SortedList' interface.
    This class has the expected properties of double-linked list:
    - sorted (according to an ordering element)
    - ordered insertion and removal
    - forward- and reverse-iterable

    As the collection implementation is interfacing and using Python vectorized lists, it is indexable
    (in contrast to common double-linked lists).
    """

    dtype = None
    _c_lib_register = None

    def __init__(self, iterable=None, dtype=Node, c_lib_register=None):
        """
        DoubleLinkedNodeList - Constructor
        :arg iterable: a Python iterable object, e.g. list, tuple, np.ndarray, ...
        :arg dtype: type of object to be stored in the list. While base functions work on aritrary objects, specific add(...) functions are targetting a 'Node' or 'NodeJIT' dtype
        :arg c_lib_register: a globally-available object of a LibraryRegisterC to forward to individual list items
        """
        super(DoubleLinkedNodeList, self).__init__(iterable)
        self.dtype = dtype
        self._c_lib_register = c_lib_register

    def __del__(self):
        self.clear(do_gc=False)

    def clear(self, do_gc=True):
        """Remove all the elements from the list."""
        n = self.__len__()
        if n > 0:
            while n > 0:
                self.__getitem__(-1).unlink()
                del self[-1]
                n = self.__len__()
        if gc is not None and do_gc:
            try:
                gc.collect()
            except:
                pass
        super()._clear()

    def __new__(cls, iterable=None, key=None, load=1000, dtype=Node, c_lib_register=None):
        """
        A static, globally-available, class-bound function to create a new instance of a DoubleLinkedNodeList
        :returns empty DoubleLinkedNodeList
        """
        return object.__new__(cls)

    def irange(self, minimum=None, maximum=None, inclusive=(True, True), reverse=False):
        """
        Generates a double-linked list with consecutive elements in the interval (:arg minimum ... :arg maximum). If
        :arg inclusive contains a 'False' item, the interval can be half-open or fully-open. A reversed-order list is
        generated using the :arg reverse parameter.
        :arg minimum: minimum element of the list; minimum object must be orderable and countable
        :arg maximum: maximum element of the list; maximum object must be orderable and countable
        :arg inclusive: tuple-of-booleans; determines if either the minimum and/or the maximum are included in the collections (i.e. 'True) or excluded (i.e. False);
                        controls the interval of the list being closed, half-open or fully-open
        :arg reverse: boolean; reverses the element order
        :returns filled DoubleLinkedNodeList
        """
        return super(DoubleLinkedNodeList, self).irange(minimum=minimum, maximum=maximum, inclusive=inclusive, reverse=reverse)

    def islice(self, start=None, stop=None, reverse=False):
        """
        Retrieves a subset of the list between :arg start and :arg stop.
        :arg start: index of the first member of the slice (i.e. included in slice)
        :arg stop: index of the last member of the slide (i.e. included in slice)
        :arg reverse: boolean; reverses the element order
        :returns: a Python vectorized list of requested elements
        """
        return super(DoubleLinkedNodeList, self).islice(start=start, stop=stop, reverse=reverse)

    def __iter__(self):
        """
        :returns a forward iterator through the list
        """
        return super(DoubleLinkedNodeList, self).__iter__()

    def __reversed__(self):
        """
        :returns a backward iterator through the collection
        """
        return super(DoubleLinkedNodeList, self).__reversed__()

    def __len__(self):
        """
        :returns the number of elements in the list
        """
        return super(DoubleLinkedNodeList, self).__len__()

    def add(self, val):
        """
        Adds a new Node(JIT) or a new Particle to the list
        :arg val: an instance of a Node (or derived subclass), an instance of a Particle, or an object ID (as integer value)
        """
        assert type(val) == self.dtype
        if isinstance(val, Node):
            self._add_by_node(val)
        elif isinstance(val, int) or type(val) in [int32, uint32, int64, uint64]:
            self._add_by_id(val)
        else:
            self._add_by_pdata(val)

    def _add_by_node(self, val):
        """
        Adds a Node(JIT)-object to the list
        :arg val: a Node(JIT) object
        """
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
        """
        Creates a new Node(JIT) object from a predefined ID, and adds it to the list.
        :arg val: an ID (integer value)
        """
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
        """
        Creates a new Node(JIT) object as container for the provided Particle object
        :arg val: a Particle object
        """
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
        """
        Appends (i.e. adds) an item to the list
        :arg val: an instanced object of type DoubleLinkedNodeList::dtype
        """
        self.add(val)

    def pop(self, idx=-1, deepcopy_elem=False):
        """
        Because we expect the return node to be of use,
        the actual node is NOT physically deleted (by calling
        the destructor). pop() only dereferences the object
        in the list. The parameter 'deepcopy_elem' can be set so as
        to physically delete the list object and return a deep copy (unlisted).
        :param idx: index of the item to be returned
        :param deepcopy_elem: boolean, telling if the returned item is a shallow copy (i.e. reference) or a new, deep copy of the object
        :returns item of type DoubleLinkedNodeList::dtype from this list, previously positioned at :arg idx
        """
        if deepcopy_elem:
            val = super().pop(idx)
            result = deepcopy(val)
            del val
            return result
        return super().pop(idx)
