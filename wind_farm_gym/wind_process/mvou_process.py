import os
from csv import DictWriter
import math
from gymnasium.utils import seeding
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
#            in this example, the vector X = [X_1, X_2, X_3] represents logarithm of the turbulence intensity and
#            wind speed M, X_1 = ln(TI), X_2 = ln(M), and wind direction phi, X_3 = phi.
# Mean:      long-term mean values of the variables. E.g., wind speed M is exp(2.2769937) (i.e., approximately 9.75),
#            and direction is 0 degrees from the mean wind direction (270).
# Drift:     A drift matrix shows how fast the variables revert to the mean values after randomly drifting away.
#            A zero matrix means no drift, and the variables are changing according to a brownian motion process.
# Diffusion: Determines the scale of a random noise process. In this case the random noise is a 3-dimensional
#            Brownian motion [W_1, W_2, W_3], where W_3 drives the randomness in the wind direction.
#            The diffusion matrix governs the scale and dependencies on these two processes, so the wind direction
#            (third line of the matrix) depends on W_3 only, but wind speed is influenced by both W_2 and W_3,
#            i.e., random fluctuations in wind speed depend on random fluctuations in wind direction as well.
# Mean Wind Direction: wind direction can be rotated arbitrarily. It is easier to simulate the wind with the mean
#            direction of 0.0, and then rotate it by a given angle in degrees.
DEFAULT_PROPERTIES = {
    'names': ['turbulence_intensity', 'wind_speed', 'wind_direction'],
    'logs': [True, True, False],
    'mean': [-2.1552094, 2.2769937, 0.0],
    'drift': [[0.0024904,      5.4994818e-04, -2.3334057e-06],
              [-2.1413137e-05, 4.7972649e-05,  5.2700795e-07],
              [3.0910895e-03, -3.57165e-03,    0.01]],
    'diffusion': [[0.0125682, -0.0002111, -0.0004371],
                  [0.0,        0.0021632,  0.0002508],
                  [0.0,        0.0,        0.1559985]],
    'mean_wind_direction': 270.0
}


class MVOUWindProcess(WindProcess):
    """
    MVOUWindProcess is a multi-variate Ornstein--Uhlenbeck process.
    """

    def __init__(self, time_delta=1, properties=None, seed=None):
        """
        Initializes a MVOU process.

        :param time_delta: time interval between two consecutive time steps, in seconds
        :param properties: a dictionary of properties of the process, includes:
            `names`: a list of atmospheric conditions, e.g., ['wind_speed', 'wind_direction'];
            `logs`: a list of boolean values of the same length, showing whether logarithmic transformation is needed
                for a particular atmospheric measurement;
            `mean`: a list of mean values to which (log)variables tend to revert to; for wind_direction usually 0.0
            `drift`: the drift matrix
            `diffusion`: the diffusion matrix
            `mean_wind_direction`: wind direction is additionally rotated by this angle after the simulation; this makes
                it possible to turn the wind without needing to re-estimate the drift and diffusion
        :param seed: random seed
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
            assert len(self._mu.shape) == 1 and self._mu.size == self._n, f'Mean must be a vector of length {self._n}'
        self._mean_wind_direction = properties.get('mean_wind_direction', 270.0)
        self._np_random, self._x = None, None
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
        """
        Saves the MVOU process into a `.csv` file, and returns a CSVProcess that reads data from that file.
        Do this when the same process needs to be used multiple times to ensure that the same data is used.
        :param data_file: file to save the generated data into
        :param time_steps: number of time steps to save
        :param time_delta: time increment between two consecutive time steps, in seconds
        :param properties: a property dictionary for the MVOU process
        :param seed: random seed
        :return: a `CSVProcess` that reads data from the file
        """
        if not os.path.exists(data_file):
            wind_process = MVOUWindProcess(time_delta=time_delta, properties=properties, seed=seed)
            wind_process.save(data_file, time_steps)
        return CSVProcess(data_file)
