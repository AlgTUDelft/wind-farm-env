

from .multi_agent import MultiAgent, AgentChild
from gym import Env
import numpy as np


class MultiAgentNaive(MultiAgent):
    def __init__(self, name, env: Env, turbine_layout, distance_threshold):
        super().__init__(
            name, "MultiAgentNaive", env, turbine_layout,
            distance_threshold=distance_threshold,
            agent_constructor=AgentNaiveChild)

    def get_log_dict(self):
        return {}


class AgentNaiveChild(AgentChild):
    def find_action(self, observation, in_eval=False):
        # empty action
        return list(np.zeros(self.action_shape))

    def learn(self, observation, action, reward, next_observation, global_step):
        # no learning
        pass
