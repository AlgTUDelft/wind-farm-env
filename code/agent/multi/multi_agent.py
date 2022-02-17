from abc import ABC, abstractmethod

import scipy.spatial.distance
from gym import Env
from ..agent import Agent
import scipy
import numpy as np


class MultiAgent(Agent, ABC):
    def __init__(self, name, type, env: Env, turbine_layout, distance_threshold, agent_constructor, **kwargs):
        super().__init__(name, type, env)

        # compute for each turbine what other turbines are within distance and store in matrix
        self.turbines_in_range_list = is_within_distance_mapping(
            distance_matrix(turbine_layout),
            distance_threshold
        )

        # create an agent for each turbine
        def create_agent(row):
            state_space_size_per_agent = 1
            number_of_agents_within_distance = len(row)
            state_size = number_of_agents_within_distance * state_space_size_per_agent
            # state size depends on the amount of other agents.
            # Action size = 1, as a single agent coordinates a single turbine
            return agent_constructor(state_shape=(state_size, 1), action_shape=(1, 1), **kwargs)

        self.child_agents = list(map(create_agent, self.turbines_in_range_list))

    def find_action(self, observation, in_eval=False):
        # todo: split observation and call find_action() child agents
        return

    def learn(self, observation, action, reward, next_observation, global_step):
        # todo: split observation and call learn() on child agents
        return


class AgentChild(ABC):
    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape

    @abstractmethod
    def find_action(self, observation, in_eval=False):
        pass

    @abstractmethod
    def learn(self, observation, action, reward, next_observation, global_step):
        pass


def distance_matrix(turbine_layout):
    turbine_coordinates = list(zip(turbine_layout["x"], turbine_layout["y"]))
    return scipy.spatial.distance.cdist(turbine_coordinates, turbine_coordinates)


def is_within_distance_mapping(distance_matrix, threshold):
    return list(map(lambda row: [i for i, x in enumerate(row) if x < threshold], distance_matrix))


if __name__ == "__main__":
    turbine_x = [0, 1, 2, 5]
    turbine_y = [5, 0, 2, 8]
    turbine_coordinates = list(zip(turbine_x, turbine_y))
    dist = distance_matrix(turbine_coordinates)
    within_distance = is_within_distance_mapping(dist, 4)
