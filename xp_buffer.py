import random


class XpBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._size = 0
        self._index = 0
        self._s = [None] * self._capacity
        self._a = [None] * self._capacity
        self._r = [None] * self._capacity
        self._t = [None] * self._capacity

    def append(self, s, a, r, t):
        self._s[self._index] = s
        self._a[self._index] = a
        self._r[self._index] = r
        self._t[self._index] = t
        self._index += 1
        if self._index == self._capacity:
            self._index = 0
        if self._size < self._capacity:
            self._size += 1

    def samples(self, n):

        # states, actions, rewards
        s = [None] * n
        a = [None] * n
        r = [None] * n

        # sample randomly
        for i in range(0, n):
            s[i], a[i], r[i] = self.sample()

        return s, a, r

    def sample(self):
        valid = False
        while not valid:
            i = random.randrange(0, self.size)
            # no terminal states
            valid = not self._t[i]
        return self._s[i], self._a[i], self._r[i]

    @property
    def size(self):
        return self._size
