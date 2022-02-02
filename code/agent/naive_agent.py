from wind_farm_gym import WindFarmEnv
from .agent import Agent
import numpy as np


class NaiveAgent(Agent):

    def __init__(self, name, env: WindFarmEnv):
        super().__init__(name, 'Naive', env)
        self._stored_representation = env.action_representation
        env.action_representation = 'wind'
        self._opt_action = list(np.zeros(self.action_shape))

    def find_action(self, observation, in_eval=False):
        return self._opt_action

    def learn(self, observation, action, reward, next_observation, global_step):
        pass

    def get_log_dict(self):
        return {}

    def close(self):
        self._env.action_representation = self._stored_representation
        super().close()
