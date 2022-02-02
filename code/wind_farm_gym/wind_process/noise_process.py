import os
from csv import DictWriter, DictReader
import math
import random

from gym.utils import seeding
from scipy.linalg import expm
import numpy as np
from numpy.linalg import pinv

from . import WindProcess, CSVProcess


class NoiseProcess(WindProcess):

    def __init__(self, n, seed=None):
        """
        :param episodes: how many episodes to simulate
        :param time_delta: time interval, in seconds, between two consecutive time steps
        :param properties: a dictionary describing the underlying MVOU process
        :param kwargs: extra parameters
        """
        super().__init__()
        self._seed = seed
        self._n = n
        self.reset()

    def step(self):
        return self._np_random.normal(size=self._n)

    def reset(self):
        super().reset()
        self._np_random, _ = seeding.np_random(self._seed)

    def save(self, file, timesteps=10000):
        keys = [f'obs_{i}' for i in range(self._n)]
        with open(file, 'w') as output_file:
            dict_writer = DictWriter(output_file, keys)
            dict_writer.writeheader()
            self.reset()
            for i in range(timesteps):
                dict_writer.writerow(self.step())
