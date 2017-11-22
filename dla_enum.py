class Enum:
    def __init__(self):
        self._val = -1

    def next(self):
        self._val += 1
        return self._val