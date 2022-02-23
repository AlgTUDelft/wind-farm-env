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
            state_space_size_per_agent = 3
            number_of_agents_within_distance = len(row)
            # state size depends on the amount of other agents.
            # We add 1 to the state size since turbulence intensity is a global observation added for all turbines
            state_size = number_of_agents_within_distance * state_space_size_per_agent + 1
            return agent_constructor(state_shape=(state_size, 1), **kwargs)

        self.child_agents = list(map(create_agent, self.turbines_in_range_list))

    def find_action(self, observation, in_eval=False):
        """
        Retrieves an action per agent.
        """
        return [
            child_agent.find_action(
                self.get_state_for_agent_from_observation(observation, child_agent_index),
                in_eval
            ) for child_agent_index, child_agent in enumerate(self.child_agents)
        ]

    def learn(self, observation, action, reward, next_observation, global_step):
        """
        Split observation and call learn() on child agents
        """
        for agent_index, agent in enumerate(self.child_agents):
            agent.learn(
                observation=self.get_state_for_agent_from_observation(observation, agent_index),
                action=action[agent_index],
                reward=reward,
                next_observation=self.get_state_for_agent_from_observation(next_observation, agent_index),
                global_step=global_step
            )

    def get_corresponding_observations(self, observation, agent_index):
        """
        retrieve those observations that correspond with the agent.
        In practice, it means all sensor data for that particular turbine.
        Retrieves yaw, speed and direction
        """
        yaw = observation[agent_index]
        speed = observation[len(self.child_agents) + 2 * agent_index + 1]
        direction = observation[len(self.child_agents) + 2 * agent_index + 2]
        return [yaw, speed, direction]

    def get_state_for_agent_from_observation(self, observation, agent_index):
        """
        Retrieves the corresponding state for a single agent.
        This consists of sensor data from surrounding turbines + the global wind turbulence
        """
        return flatten([
            self.get_corresponding_observations(observation, nearby_agent_index)
            for nearby_agent_index in self.turbines_in_range_list[agent_index]
        ]) + [observation[len(self.child_agents)]]


class AgentChild(ABC):
    def __init__(self, state_shape):
        self.state_shape = state_shape

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


def flatten(t):
    return [item for sublist in t for item in sublist]