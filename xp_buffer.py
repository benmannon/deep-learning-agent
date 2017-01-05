import random


class XpBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._buf = []

    def append(self, xps):
        self._buf += xps
        overfill = len(self._buf) - self._capacity
        if overfill > 0:
            del self._buf[:overfill]

    def samples(self, n):
        elements = []
        for _ in range(0, n):
            element = self._buf[random.randrange(0, len(self._buf))]
            elements.append(element)
        return elements
