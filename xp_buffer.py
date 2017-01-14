import random


class XpBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._size = 0
        self._index = 0
        self._s = [None] * self._capacity
        self._a = [None] * self._capacity
        self._r = [None] * self._capacity

    def append(self, s, a, r):
        self._index += 1
        if self._index == self._capacity:
            self._index = 0
        if self._size < self._capacity:
            self._size += 1
        self._s[self._index] = s
        self._a[self._index] = a
        self._r[self._index] = r

    def samples(self, n):

        # states, actions, rewards
        s = []
        a = []
        r = []

        # sample randomly
        for _ in range(0, n):
            i = random.randrange(0, len(self._s))
            s += [self._s[i]]
            a += [self._a[i]]
            r += [self._r[i]]

        return s, a, r

    @property
    def size(self):
        return self._size
