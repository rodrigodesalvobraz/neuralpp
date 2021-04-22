class EveryKTimes:
    """Provides an object that runs a given thunk (0-args function) every k-th time it is invoked"""

    def __init__(self, thunk, k):
        self.thunk = thunk
        self.k = k
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter == self.k:
            self.counter = 0
            self.thunk()
