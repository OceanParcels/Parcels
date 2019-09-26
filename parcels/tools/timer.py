from __future__ import print_function

import datetime
import time
try:
    from mpi4py import MPI
except:
    MPI = None


class Timer():
    def __init__(self, name, parent=None, start=True):
        self._start = None
        self._t = 0
        self._name = name
        self._children = []
        self._parent = parent
        if self._parent:
            self._parent._children.append(self)
        if start:
            self.start()

    def start(self):
        if self._parent:
            assert(self._parent._start), ("Timer '%s' cannot be started. Its parent timer does not run" % self._name)
        if self._start is not None:
            raise RuntimeError('Timer %s cannot start since it is already running' % self._name)
        self._start = time.time()

    def stop(self):
        assert(self._start), ("Timer '%s' was stopped before being started" % self._name)
        self._t += time.time() - self._start
        self._start = None

    def print_local(self):
        if self._start:
            print("Timer '%s': %g s (process running)" % (self._name, self._t + time.time() - self._start))
        else:
            print("Timer '%s': %g s" % (self._name, self._t))

    def local_time(self):
        return self._t + time.time() - self._start if self._start else self._t

    def print_tree_sequential(self, step=0, root_time=0, parent_time=0):
        time = self.local_time()
        if step == 0:
            root_time = time
        print(('(%3d%%)' % round(time/root_time*100)), end='')
        for i in range(step+1):
            print('  ', end='')
        if step > 0:
            print('(%3d%%) ' % round(time/parent_time*100), end='')
        t_str = '%1.3e s' % time if root_time < 300 else datetime.timedelta(seconds=time)
        print("Timer %s: %s" % ((self._name).ljust(20 - 2*step + 7*(step == 0)), t_str))
        for child in self._children:
            child.print_tree_sequential(step+1, root_time, time)

    def print_tree(self, step=0, root_time=0, parent_time=0):
        if MPI is None:
            self.print_tree_sequential(step, root_time, parent_time)
        else:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()
            if mpi_size == 1:
                self.print_tree_sequential(step, root_time, parent_time)
            else:
                for iproc in range(mpi_size):
                    if iproc == mpi_rank:
                        print("Proc %d/%d - Timer tree" % (mpi_rank, mpi_size))
                        self.print_tree_sequential(step, root_time, parent_time)
                    mpi_comm.Barrier()
