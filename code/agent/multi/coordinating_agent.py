import scipy.spatial.distance
from gym import Env
from ..agent import Agent
import scipy
import numpy as np


class CoordinatingAgent(Agent):
    def __init__(self, name, env: Env, turbine_coordinates, distance_threshold, agent_constructor, state_space_size_per_agent):
        super().__init__(name, "Multi-Agent", env)

        # compute for each turbine what other turbines are within distance and store in matrix
        self.turbines_in_range_list = is_within_distance_mapping(
            distance_matrix(turbine_coordinates),
            distance_threshold
        )

        # create an agent for each turbine
        def create_agent(row):
            number_of_agents_within_distance = len(row)
            state_size = number_of_agents_within_distance * state_space_size_per_agent
            # state size depends on the amount of other agents.
            # Action size = 1, as a single agent coordinates a single turbine
            return agent_constructor(state_size=state_size, action_size=1)

        self.child_agents = list(map(create_agent, self.turbines_in_range_list))

    def find_action(self, observation, in_eval=False):
        # split up the observation such that the correct state space is sent to each agent
        # todo
        pass

    def learn(self, observation, action, reward, next_observation, global_step):
        # split up the observation such that the correct state space is sent to each agent
        # todo
        pass

    def get_log_dict(self):
        # todo
        pass


def distance_matrix(turbine_coordinates):
    return scipy.spatial.distance.cdist(turbine_coordinates, turbine_coordinates)


def is_within_distance_mapping(distance_matrix, threshold):
    return list(map(lambda row: [i for i, x in enumerate(row) if x < threshold], distance_matrix))


if __name__ == "__main__":
    turbine_x = [0, 1, 2, 5]
    turbine_y = [5, 0, 2, 8]
    turbine_coordinates = list(zip(turbine_x, turbine_y))
    dist = distance_matrix(turbine_coordinates)
    within_distance = is_within_distance_mapping(dist, 4)
