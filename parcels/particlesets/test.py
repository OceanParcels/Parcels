class SomeIterator(object):
    def __init__(self, obj, reverse=False):
        self._reverse = reverse
        self.obj = obj
        self.max_len = len(obj.list)
        self._index = 0
        if reverse:
            self._index = self.max_len - 1

    def __iter__(self):
        return self

    def __next__(self):
        if not self._reverse and self._index < self.max_len:
            result = self.obj.list[self._index]
            self._index += 1
            return result

        if self._reverse and self._index >= 0:
            result = self.obj.list[self._index]
            self._index -= 1
            return result
        raise StopIteration

    def __repr__(self):
        direction_str = 'Backward' if self._reverse else 'Forward'
        return f"{direction_str} iteration at index {self._index} of {self.max_len}."

class SomeObject(object):
    def __init__(self, list):
        self.list = list

    def __iter__(self):
        return SomeIterator(self)

    def __reversed__(self):
        return SomeIterator(self, True)


if __name__ == "__main__":
    it = SomeObject(["a", "B", "c", "D"])
    for i, j in enumerate(it):
        print(f"{i} : {j}")

    for i in reversed(it):
        print(i)