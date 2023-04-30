from gym.utils import seeding

from . import WindProcess


class MVGaussianNoiseProcess(WindProcess):
    """
    MVGaussianNoiseProcess is a multi-variate Gaussian noise process.
    """

    def __init__(self, n: int, seed=None):
        """
        Initialize the process
        :param n: number of dimensions
        :param seed: random seed
        """
        super().__init__()
        self._seed = seed
        self._n = n
        self._np_random = None
        self.reset()

    def step(self):
        return {'value': self._np_random.normal(size=self._n)}

    def reset(self):
        super().reset()
        self._np_random, _ = seeding.np_random(self._seed)
