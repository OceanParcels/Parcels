from __future__ import print_function
import time


global_timers = {}


class Timer():
    def __init__(self, name, parent='', start=True):
        self._start = -1
        self._end = -1
        self._t = 0
        self._name = name
        assert(name not in global_timers), ("Name '%s' is already defined for another timer" % name)
        global_timers[name] = self
        self._children = []
        if parent:
            assert(parent in global_timers), ("Parent '%s' is not defined" % parent)
            self._parent = global_timers[parent]
            self._parent._children.append(self)
        else:
            self._parent = None
        if start:
            self.start()

    def start(self):
        if self._parent:
            assert(self._parent._start > 0), ("Timer '%s' cannot be started. Its parent timer does not run" % self._name)
        self._start = time.time()

    def stop(self):
        assert(self._start > 0), ("Timer '%s' was stopped before being started" % self._name)
        self._t += time.time() - self._start
        self._start = -1

    def print_local(self):
        if self._start > 0:
            print("Timer '%s': %g s (process running)" % (self._name, self._t + time.time() - self._start))
        else:
            print("Timer '%s': %g s" % (self._name, self._t))

    def local_time(self):
        if self._start > 0:
            return self._t + time.time() - self._start
        else:
            return self._t

    def print_tree(self, step=0, root_time=0, parent_time=0):
        time = self.local_time()
        if step == 0:
            root_time = time
        print(('(%3d%%)' % round(time/root_time*100)), end='')
        for i in range(step+1):
            print('  ', end='')
        print("Timer '%s': %1.1e s" % (self._name, time), end='')
        if step == 0:
            print('')
        else:
            print(('  (%3d%%)' % round(time/parent_time*100)))
        for child in self._children:
            child.print_tree(step+1, root_time, time)
