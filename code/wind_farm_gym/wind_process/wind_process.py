from abc import ABC, abstractmethod


class WindProcess(ABC):

    def __init__(self):
        self._t = 0

    def step(self):
        self._t += 1

    def reset(self):
        self._t = 0

    def close(self):
        pass
