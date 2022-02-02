import os
from csv import DictWriter, DictReader
import math
import random

from gym.utils import seeding
from scipy.linalg import expm
import numpy as np
from numpy.linalg import pinv

from . import WindProcess, CSVProcess

# This is a data provider based on a Multi-Variate Ornstein-Uhlenbeck process. It is a continuous-time stochastic
# process described by a differential equation
# dX = theta (mu - X) dt + sigma dW,
# where X is a vector of variables, theta is a drift matrix, mu is a mean vector, and sigma is a diffusion matrix.
# There interpretation is as follows:
# Names:     names of the variables in the same order they are used in vectors/matrices
# Logs:      whether a logarithmic transformation has been taken or not. The transformation is usually required for
#            non-negative variables, such as wind speed, to convert their co-domain from (0, inf) to (-inf, inf).
#            in this example, the vector X = [X_1, X_2] represents logarithm of the wind speed M, X_1 = ln(M),
#            and wind direction phi, X_2 = phi.
# Mean:      long-term mean values of the variables. Wind speed M is exp(2) (because it is 2 for ln(M)), and
#            direction is 270 degrees.
# Drift:     A drift matrix shows how fast the variables revert to the mean values after randomly drifting away.
#            A zero matrix means no drift, and the variables are changing according to a brownian motion process.
# Diffusion: Determines the scale of a random noise process. In this case the random noise is a 2-dimensional
#            Brownian motion [W_1, W_2], where W_1 drives the randomness in the wind speed and W_2 in the direction.
#            The diffusion matrix governs the scale and dependencies on these two processes, so the wind direction
#            (second line of the matrix) depends on W_2 only, but wind speed is influenced by both W_1 and W_2,
#            i.e., random fluctuations in wind speed depend on random fluctuations in wind direction as well.


DEFAULT_PROPERTIES = {
    'names': ['wind_speed', 'wind_direction'],
    'logs': [True, False],
    'mean': [2.219, 0.0],
    'drift': [[0.032, 0.00],
              [-3.35, 0.002]],
    'diffusion': [[0.055, 0.0],
                  [0.069, 4.037]],
    'mean_wind_direction': 270.0
}


class MVOUWindProcess(WindProcess):

    def __init__(self, time_delta=1, properties=None, seed=None):
        """
        :param episodes: how many episodes to simulate
        :param time_delta: time interval, in seconds, between two consecutive time steps
        :param properties: a dictionary describing the underlying MVOU process
        :param kwargs: extra parameters
        """
        super().__init__()
        self._seed = seed
        if properties is None:
            properties = DEFAULT_PROPERTIES
        self._dt = time_delta
        self._names = properties.get('names', [])
        self._theta = np.array(properties.get('drift', []))
        self._sigma = np.array(properties.get('diffusion', []))
        self._mu = np.array(properties.get('mean', []))
        assert len(self._theta.shape) == 2 and len(self._sigma.shape) == 2,\
            'Need square drift and diffusion matrices'
        assert self._theta.shape[0] == self._theta.shape[1] == self._sigma.shape[0] == self._sigma.shape[1],\
            'Matrices have incompatible dimensions'
        self._n = self._theta.shape[0]
        self._logs = properties.get('logs', [False for _ in range(self._n)])
        if self._mu is None:
            self._mu = np.zeros(self._n)
        else:
            assert len(self._mu.shape) == 1 and self._mu.size == self._n, f'Mean must be a vector of length {n}'
        self._mean_wind_direction = properties.get('mean_wind_direction', 270.0)
        self.reset()

        # Based on
        # Meucci, A. (2009). Review of statistical arbitrage, cointegration, and multivariate Ornstein-Uhlenbeck.
        # http://symmys.com/node/132. Working Paper
        # and https://doi.org/10.1186/s13662-019-2214-1
        i_n = np.eye(self._n)
        self._exp_theta = expm(self._theta * (-self._dt))
        self._eps_mean = (i_n - self._exp_theta) @ self._mu
        kron_sum = np.kron(self._theta, i_n) + np.kron(i_n, self._theta)
        sigma_square = self._sigma @ self._sigma.transpose()
        self._eps_cov = pinv(kron_sum) @ (np.eye(self._n*self._n) - expm(kron_sum * (-self._dt)))
        self._eps_cov = self._eps_cov @ sigma_square.flatten('F')
        self._eps_cov = self._eps_cov.reshape((self._n, self._n), order='F')

    def step(self):
        super().step()
        eps = self._np_random.multivariate_normal(mean=self._eps_mean, cov=self._eps_cov)
        self._x = self._exp_theta @ self._x + eps
        return self._get_vars_dictionary()

    def reset(self):
        super().reset()
        self._np_random, _ = seeding.np_random(self._seed)
        self._x = self._mu

    def _get_vars_dictionary(self):
        x_dict = {self._names[i]: (math.exp(self._x[i]) if self._logs[i] else self._x[i]) for i in range(len(self._x))}
        if 'wind_direction' in self._names:
            x_dict['wind_direction'] = (x_dict['wind_direction'] + self._mean_wind_direction) % 360
        return x_dict

    def save(self, file, timesteps=10000):
        keys = self._names
        with open(file, 'w') as output_file:
            dict_writer = DictWriter(output_file, keys)
            dict_writer.writeheader()
            self.reset()
            for i in range(timesteps):
                dict_writer.writerow(self.step())

    @staticmethod
    def switch_to_csv(data_file, time_steps, time_delta, properties, seed):
        if not os.path.exists(data_file):
            wind_process = MVOUWindProcess(time_delta=time_delta, properties=properties, seed=seed)
            wind_process.save(data_file, time_steps)
        return CSVProcess(data_file)
