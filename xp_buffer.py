import random


class XpBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._s = []
        self._a = []
        self._r = []

    def append(self, states, actions, rewards):
        self._s += states
        self._a += actions
        self._r += rewards
        overfill = len(self._s) - self._capacity
        if overfill > 0:
            del self._s[:overfill]
            del self._a[:overfill]
            del self._r[:overfill]

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
        return len(self._s)
