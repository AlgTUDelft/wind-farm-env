from typing import Tuple, List, Literal, Union, Optional, Dict

import floris.simulation
from floris.tools.floris_interface import FlorisInterface
from floris.simulation import Farm

from gym import Env
from gym.utils import seeding
from gym.spaces import Box, Discrete

import numpy as np
import os

from .farm_visualization import FarmVisualization
from .wind_process import WindProcess, NoiseProcess

# When normalizing the state vector to [0, 1], we need to know the boundaries for the atmospheric conditions.
# If no such boundaries are supplied, these defaults will be used.
DEFAULT_BOUNDARIES = {
    "wind_speed": (0.0, 20.0),
    "wind_direction": (0.0, 360.0),
    "wind_veer": (-1, 1),
    "wind_shear": (-1, 1),
    "turbulence_intensity": (0.0, 2.0),
    "air_density": (1.1455, 1.4224)  # from 1.1455 at 35°C to 1.4224 at -25°C
}


class WindFarmEnv(Env):
    """
    WindFarmEnv is an OpenAI gym environment that simulates a wind farm as a reinforcement learning problem
    with dynamic atmospheric conditions. It uses FLORIS to simulate the farm at each time step, and a custom
    (stochastic) process for the wind data for transitions between time steps.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(
            self,
            seed: Optional[int] = None,
            floris: Optional[Union[FlorisInterface, str]] = None,
            turbine_layout: Optional[Union[Dict[str, List[float]], Tuple[List[float], List[float]]]] = None,
            mast_layout: Optional[Union[Dict[str, List[float]], Tuple[List[float], List[float]]]] = None,
            time_delta: float = 1.0,
            max_angular_velocity: float = 1.0,
            desired_yaw_boundaries: Tuple[float, float] = (-45.0, 45.0),
            wind_process: Optional[WindProcess] = None,
            observe_yaws: bool = True,
            farm_observations: Optional[Union[str, Tuple[str, ...]]] = None,
            mast_observations: Optional[Union[str, Tuple[str, ...]]] = ('wind_speed', 'wind_direction'),
            lidar_observations: Optional[Union[str, Tuple[str, ...]]] = ('wind_speed', 'wind_direction'),
            lidar_turbines: Optional[Union[str, int, Tuple[int, ...]]] = 'all',
            lidar_range: Optional[float] = 10.0,
            observation_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
            normalize_observations: bool = True,
            random_reset=False,
            action_representation: Literal['wind', 'absolute', 'yaw'] = 'wind',
            perturbed_observations: Optional[Union[str, int, Tuple[int, ...]]] = None,
            perturbation_scale: float = 0.05
    ):
        """
        Initialize an instance of WindFarmEnv.

        :param seed: random seed
        :param floris: either FlorisInterface or a path to an input .json file to initialize FLORIS; if None,
        default_floris_input.json will be used
        :param turbine_layout: positions (in meters) of the turbines in the wind farm; either a dictionary
        {'x': [x_0, x_1, ...], 'y': [y_o, y_1, ...]}, or a tuple of two lists ([x_0, x_1, ...], [y_o, y_1, ...]);
        if None, the positions will be taken from FlorisInterface
        :param mast_layout: positions of the met masts; same format as @turbine_layout; if None, no met masts are
        created
        :param time_delta: time interval between two consecutive time steps, in seconds
        :param max_angular_velocity: maximum angular velocity of the turbines, degrees/sec; determines how much can the
        yaws change between two consecutive time steps
        :param desired_yaw_boundaries: minimum gamma_min and maximum yawing to the wind that the turbines will  try to
        maintain; if the current wind direction is gamma, the turbine will not turn to angles outside of
        [gamma - gamma_min, gamma + gamma_max]; if a yaw is already outside of these limits, the turbine will
        turn to the closest point within the interval it can reach, given its angular velocity
        :param wind_process: a (stochastic) wind process that governs the transitions between time steps; if None
        static wind is used, based on data in the FlorisInterface
        :param observe_yaws: are the yaws included in the observation vector?
        :param farm_observations: atmospheric conditions observed on the farm-level; these are not measured by met masts
        or turbines, but come from an external source and are always the same across the wind farm; any atmospheric
        conditions supported by FLORIS can be used; for FLORIS 2.4, these are "wind_speed", "wind_direction",
        "wind_veer", "wind_shear", "turbulence_intensity", and "air_density"; these atmospheric conditions are included
        in the observation vector
        :param mast_observations: atmospheric conditions measured by met masts, see @farm_observations for the list of
        possible values; these atmospheric conditions are included in the observation vector separately for each mast
        :param lidar_observations: atmospheric conditions measured at turbine locations by nacelle-mounted lidars, see
        @farm_observations for the list of possible values; these atmospheric conditions are included in the observation
        vector separately for each turbine from the @lidar_turbines list
        :param lidar_turbines: turbines that collect atmospheric measurements; either 'all', or a list of turbine
        indices corresponding to @turbine_layout
        :param lidar_range: turbine measurements are collected at a point located this far in front of the turbine rotor
        for a negative value, measurements are collected behind the rotor area, and thus are affected by the turbine's
        wake
        :param observation_boundaries: for normalizing the observations, minimum and maximum values are required; if
        None, the defaults will be used
        :param normalize_observations: should the observations be normalized?
        :param random_reset: if True, when the environment is reset the yaws will be set to random values given by
        @desired_yaw_boundaries; otherwise, the turbines will face the wind upon reset (or the smallest possible yaw,
        if @desired_yaw_boundaries do not allow facing the wind)
        :param action_representation: the way that the actions are encoded into [0, 1]; Either 'yaw', 'absolute', or
        'wind', see Section 3.2 in the paper for details
        :param perturbed_observations: a list of atmospheric conditions with perturbed observations; white noise
        is injected into these measurements
        :param perturbation_scale: perturbation noise scale, relative to the observation scale. If the observation is
        normalized, than zero-mean Gaussian variables with standard deviation of @perturbation_scale are added to
        each of the perturbed observations; for non-normalized observations, this noise is appropriately rescaled
        """
        # random seeding
        self._np_random, self._seed = self.seed(seed)
        self.random_reset = random_reset

        # initialize the floris interface depending on the argument type
        if floris is None:
            # use the default floris parameters
            file_path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(file_path, 'default_floris_input.json')
            self.floris_interface = FlorisInterface(path)
        elif isinstance(floris, FlorisInterface):
            self.floris_interface = floris
        elif isinstance(floris, str):
            self.floris_interface = FlorisInterface(floris)

        # place the turbines in the environment, if a layout is given
        if turbine_layout is not None:
            if isinstance(turbine_layout, dict):
                turbine_layout = (turbine_layout.get('x', [0.0]), turbine_layout.get('y', [0.0]))
            assert len(turbine_layout) == 2 and len(turbine_layout[0]) == len(turbine_layout[1]),\
                "Numbers of turbines in x and y coordinates do not match"
            self.floris_interface.reinitialize_flow_field(layout_array=turbine_layout)
        self.turbine_layout = (self.floris_interface.layout_x, self.floris_interface.layout_y)

        # place wind measurement masts
        if mast_layout is None:
            self.mast_layout = ()
        else:
            if isinstance(mast_layout, dict):
                mast_layout = (mast_layout.get('x', [0.0]), mast_layout.get('y', [0.0]))
            assert len(mast_layout) == 2 and len(mast_layout[0]) == len(mast_layout[1]), \
                "Numbers of turbines in x and y coordinates do not match"
            self.mast_layout = mast_layout

        # store a reference to the farm for easier access
        farm = self.floris_interface.floris.farm
        if isinstance(farm, Farm):
            self._farm = farm
        else:
            # only a single farm is currently supported
            raise NotImplementedError

        # initialize yawing constraints
        self.desired_min_yaw = min(desired_yaw_boundaries)
        self.desired_max_yaw = max(desired_yaw_boundaries)
        self.time_delta = time_delta
        self.max_delta_yaw = max_angular_velocity * self.time_delta

        # initialize the wind data provider
        self.wind_process = wind_process

        # initialize the action space to a normalized interval [-1; 1] for each turbine
        ones = np.array([1.0 for _ in range(self.n_turbines)], dtype=np.float32)
        self.action_space = Box(-ones, ones, dtype=np.float32)
        self.action_space.seed(self._seed)
        self.action_representation = action_representation
        if self.action_representation == 'absolute':
            self._primal_wind_direction = self._farm.wind_direction[0]

        self.current_flow_points = None
        self._current_flow = None

        # initialize the observation space
        self.observed_variables = []
        if observe_yaws:
            self.observed_variables.extend(
                [
                    {
                        'name': f'turbine_{n}_yaw',
                        'min': -180.0,
                        'max': 180.0,
                        'type': 'yaw',
                        'index': n
                    }
                    for n in range(self.n_turbines)
                ]
            )

        if observation_boundaries is None:
            observation_boundaries = {}

        # build a list of farm-wide observations
        if isinstance(farm_observations, str):
            farm_observations = (farm_observations, )
        if farm_observations is not None and len(farm_observations) > 0:
            self.observed_variables.extend(
                [
                    {
                        'name': f'{variable}',
                        'min': observation_boundaries.get(variable, DEFAULT_BOUNDARIES.get(variable, -np.inf))[0],
                        'max': observation_boundaries.get(variable, DEFAULT_BOUNDARIES.get(variable, np.inf))[1],
                        'type': variable
                    }
                    for variable in farm_observations
                ]
            )

        # build a list of per-mast observations
        if isinstance(mast_observations, str):
            mast_observations = (mast_observations, )
        if mast_observations is not None and len(mast_observations) > 0:
            self.observed_variables.extend(
                [
                    {
                        'name': f'mast_{n}_{variable}',
                        'min': observation_boundaries.get(variable, DEFAULT_BOUNDARIES.get(variable, -np.inf))[0],
                        'max': observation_boundaries.get(variable, DEFAULT_BOUNDARIES.get(variable, np.inf))[1],
                        'type': variable,
                        'index': n
                    }
                    for n in range(self.n_masts) for variable in mast_observations
                ]
            )

        # build a list of per-turbine observations
        self._lidar_turbines = []
        if isinstance(lidar_observations, str):
            lidar_observations = (lidar_observations, )
        if lidar_observations is not None and len(lidar_observations) > 0:
            if isinstance(lidar_turbines, int):
                lidar_turbines = (lidar_turbines, )
            elif isinstance(lidar_turbines, str) and lidar_turbines == 'all':
                lidar_turbines = [i for i in range(self.n_turbines)]
            if lidar_turbines is not None and len(lidar_turbines) > 0:
                self._lidar_range = lidar_range
                self.observed_variables.extend(
                    [
                        {
                            'name': f'turbine_{n}_{variable}',
                            'min': observation_boundaries.get(variable, DEFAULT_BOUNDARIES.get(variable, -np.inf))[0],
                            'max': observation_boundaries.get(variable, DEFAULT_BOUNDARIES.get(variable, np.inf))[1],
                            'type': variable,
                            'index': i + self.n_masts
                        }
                        for n, i in zip(lidar_turbines, range(len(lidar_turbines))) for variable in lidar_observations
                    ]
                )
                self._lidar_turbines = list(lidar_turbines)

        # if nothing is observed, the problem has no states; in MDPs this is modeled as a single-state problem
        self._has_states = len(self.observed_variables) > 0

        if self._has_states:
            self.low = np.array([d['min'] for d in self.observed_variables], dtype=np.float32)
            self.high = np.array([d['max'] for d in self.observed_variables], dtype=np.float32)

            self._normalize_observations = normalize_observations
            if self._normalize_observations:

                self.state_delta = self.high - self.low
                self.observation_space = Box(np.zeros_like(self.low, dtype=np.float32),
                                             np.ones_like(self.high, dtype=np.float32),
                                             dtype=np.float32)
            else:
                self.observation_space = Box(self.low, self.high, dtype=np.float32)

            self._perturbed_observations = None
            if perturbed_observations is not None:
                if isinstance(perturbed_observations, (float, int)):
                    perturbed_observations = [perturbed_observations]
                elif isinstance(perturbed_observations, str):
                    perturbed_observations = [i for i in range(len(self.observed_variables))]
                self._perturbed_observations = list(perturbed_observations)
                self._perturbation_scale = [(x['max'] - x['min']) * perturbation_scale for x in self.observed_variables]
                self._noise = NoiseProcess(len(self._perturbed_observations))
        else:
            self.observation_space = Discrete(1)  # single-state problem
        self.observation_space.seed(self._seed)

        # initialize the reward range
        min_reward = 0.0
        max_reward = 0.0

        if self.desired_min_yaw <= 0.0 <= self.desired_max_yaw:
            best_angle = 0.0
            if abs(self.desired_min_yaw) <= self.desired_max_yaw:
                worst_angle = self.desired_min_yaw
            else:
                worst_angle = self.desired_max_yaw
        else:
            if self.desired_min_yaw > 0:
                best_angle, worst_angle = self.desired_max_yaw, self.desired_min_yaw
            else:
                best_angle, worst_angle = self.desired_min_yaw, self.desired_max_yaw

        # To compute the reward range, we check the power curve for the best and worst power output.
        # To do so, we first need to know the maximum and minimum wind speed for which the power curve is defined in
        # FlorisInterface
        if wind_process is None:
            max_speed = np.max(self.floris_interface.floris.farm.wind_speed)
            min_speed = np.min(self.floris_interface.floris.farm.wind_speed)
        else:
            min_speed, max_speed = observation_boundaries.get('wind_speed', DEFAULT_BOUNDARIES.get('wind_speed'))

        # In theory, the turbines may not be identical, so we cannot just multiply by the number of turbines
        for turbine in self.turbines:
            # remember the original values
            yaw = turbine.yaw_angle
            velocities = turbine.velocities

            # calculate the output under the best conditions
            x = np.where(np.logical_and(turbine.powInterp.x <= max_speed, turbine.powInterp.x >= min_speed))[0]
            best_speed = x[np.argmax(turbine.powInterp.y[x])]
            turbine.velocities = np.repeat(turbine.powInterp.x[best_speed],
                                           turbine.grid_point_count)
            turbine.yaw_angle = best_angle
            max_reward += turbine.power

            # calculate the output under the worst conditions
            worst_speed = x[np.argmin(turbine.powInterp.y[x])]
            turbine.velocities = np.repeat(turbine.powInterp.x[worst_speed],
                                           turbine.grid_point_count)
            turbine.yaw_angle = worst_angle
            min_reward += turbine.power

            # return the values to the original ones
            turbine.yaw_angle = yaw
            turbine.velocities = velocities

        self._reward_scaling_factor = 1.0e-6 * self.time_delta / 3600  # from power to energy, convert to MWh
        self.reward_range = (min_reward * self._reward_scaling_factor, max_reward * self._reward_scaling_factor)

        self.visualization = None
        self.state = self._get_state()

    @property
    def turbines(self) -> List[floris.simulation.Turbine]:
        """
        All turbines in the farm
        """
        return self._farm.turbines

    @property
    def n_turbines(self) -> int:
        """
        Number of turbines in the farm
        """
        return len(self.turbines)

    @property
    def n_masts(self) -> int:
        """
        Number of meteorological masts in the farm
        """
        return 0 if len(self.mast_layout) == 0 else len(self.mast_layout[0])

    @property
    def yaws_from_wind(self) -> List[float]:
        """
        Turbine yaws relative to the wind
        """
        return [turbine.yaw_angle for turbine in self.turbines]

    @property
    def wind_directions_at_turbines(self) -> List[float]:
        """
        Wind directions at turbine locations
        """
        return self._farm.wind_direction

    @property
    def yaws_from_north(self) -> List[float]:
        """
        Turbine yaws from the north
        """
        return [x - y for x, y in zip(self.wind_directions_at_turbines, self.yaws_from_wind)]

    @property
    def hub_height(self) -> float:
        """
        Hub height of the turbines; currently assumed to be the same
        """
        return self.turbines[0].hub_height

    def _get_measurement_point_data(self, d):
        i, measurement = d.get('index'), d['type']
        if i is None:
            res = self._farm.__getattribute__(measurement)
            if not isinstance(res, (int, float)):
                res = res[0]
        else:
            if measurement == 'yaw':
                res = self.yaws_from_wind[i]
            else:
                x, y, z = self.current_flow_points[0][i], self.current_flow_points[1][i], self.current_flow_points[2][i]
                flow = self._current_flow
                flow = flow[(flow.x == x) & (flow.y == y) & (flow.z == z)]
                u, v = flow.u.mean(), flow.v.mean()
                if measurement == 'wind_speed':
                    res = (u ** 2 + v ** 2) ** 0.5
                elif measurement == 'wind_direction':
                    res = (np.degrees(np.arctan2(v, u)) + self._farm.wind_map.input_direction[0]) % 360
                elif measurement == 'turbulence_intensity':
                    res = self._farm.wind_map.input_ti[0]
                else:
                    res = self._farm.__getattribute__(measurement)
        return res

    @staticmethod
    def _get_farm_measurement(self, measurement):
        return self._farm.__getattribute__(measurement)

    def _flow_points(self):
        if self.n_masts > 0:
            x, y = self.mast_layout[0], self.mast_layout[1]
        else:
            x, y = [], []
        angles = np.radians(self.yaws_from_north)
        if len(self._lidar_turbines) > 0:
            angles = [angles[i] for i in self._lidar_turbines]
            dx = np.round(self._lidar_range * np.sin(angles), 5)
            dy = np.round(self._lidar_range * np.cos(angles), 5)
            dx += [self.turbine_layout[0][i] for i in self._lidar_turbines]
            dy += [self.turbine_layout[1][i] for i in self._lidar_turbines]
            x = x + list(dx)
            y = y + list(dy)
        z = [self.hub_height] * len(x)
        return x, y, z

    def _get_state(self):
        if self._has_states:
            self.current_flow_points = self._flow_points()
            if len(self.current_flow_points[0]) > 0:
                self._current_flow = self.floris_interface.get_set_of_points(*self.current_flow_points)

            state = [self._get_measurement_point_data(d) for d in self.observed_variables]

            # inject noise
            if self._perturbed_observations is not None:
                added_noise = np.zeros_like(state)
                added_noise.put(self._perturbed_observations, self._noise.step(), mode='raise')
                added_noise = added_noise * self._perturbation_scale
                state = state + added_noise

            # rescale and clip off
            if self._normalize_observations:
                state = (np.array(state) - self.low) / self.state_delta
                state = np.clip(state, np.zeros_like(self.low), np.ones_like(self.high))
            else:
                state = np.clip(state, self.low, self.high)
            return list(state)
        else:
            return 0

    def _generate_noise(self):
        pass  # [self._get_measurement_point_data(d) for d in self._observed_variables]

    def seed(self, seed=None):
        return seeding.np_random(seed)

    def step(self, action):
        """

        :type action: np.array
        """
        action = np.array(action, dtype=np.float32)
        if not self.action_space.contains(action):
            action = np.minimum(np.maximum(action, -1.0), 1.0)

        if self.wind_process is not None:
            new_wind_data = self.wind_process.step()
            old_wind_direction = np.array(self._farm.__getattribute__('wind_direction'))
            # adjust the flow field
            self.floris_interface.reinitialize_flow_field(**new_wind_data)
            self.floris_interface.calculate_wake()

            new_wind_direction = np.array(self._farm.__getattribute__('wind_direction'))
            wind_direction_change = (new_wind_direction - old_wind_direction + 180) % 360 - 180
            # print(f"delta: {wind_direction_change}")
        else:
            old_wind_direction = new_wind_direction = np.array(self._farm.__getattribute__('wind_direction'))

        # take an action
        self._adjust_yaws(action, new_wind_direction, old_wind_direction)

        done = False
        # self._floris_interface.reinitialize_flow_field()
        self.state = self._get_state()
        self.floris_interface.calculate_wake()
        reward = np.sum(self.floris_interface.get_turbine_power())
        if np.isnan(reward):
            reward = 0
        reward *= self._reward_scaling_factor
        return self.state, reward, done, {}

    def _adjust_yaws(self, action, new_wind_direction, old_wind_direction):
        """
        Changes the turbine yaws given an action vector.
        """
        action = np.array(action)
        wind_direction_change = (new_wind_direction - old_wind_direction + 180) % 360 - 180
        # get how far the turbine is allowed to turn in each direction in one time step
        y_max = [min(yaw + self.max_delta_yaw, self.desired_max_yaw) for yaw in self.yaws_from_wind]
        y_min = [max(yaw - self.max_delta_yaw, self.desired_min_yaw) for yaw in self.yaws_from_wind]
        # sometimes a turbine is too far out of the desired yawing zone. In this case it should
        # move towards the wind direction. This ensures that by setting equal lower and upper bounds
        # for the next step yaw range
        y_max, y_min = [y if y > x == self.desired_max_yaw else x for x, y in zip(y_max, y_min)],\
                       [x if x < y == self.desired_min_yaw else y for x, y in zip(y_max, y_min)]
        # depending on the representation, compute new yaws from the action
        if self.action_representation == 'yaw':          # see Section 3.2.1 in the paper
            new_yaws = ((self.yaws_from_wind + action * self.max_delta_yaw) + 180) % 360 - 180
        elif self.action_representation == 'wind':       # see Section 3.2.3 in the paper
            new_yaws = (action + 1.0) / 2.0 * (self.desired_max_yaw - self.desired_min_yaw) + self.desired_min_yaw
        elif self.action_representation == 'absolute':   # see Section 3.2.2 in the paper
            absolute_yaws = self._primal_wind_direction - action * 180.0
            new_yaws = (old_wind_direction - absolute_yaws + 180) % 360 - 180
        else:
            # If you want to add extra representations, here goes the code that computes new yaws from actions
            raise NotImplementedError
        # ensure that the new yaw is within the limits
        for yaw, i in zip(np.clip(new_yaws, y_min, y_max), range(self.n_turbines)):
            self.turbines[i].yaw_angle = yaw + wind_direction_change[i]

    # Implementing a method from the base class. This method resets the environment to begin a new experiment
    def reset(self):
        if self.random_reset:
            for turbine in self._turbines:
                turbine.yaw_angle = self._np_random.uniform(self.desired_min_yaw, self.desired_max_yaw)
        else:
            default_yaw = min(self.desired_max_yaw, max(self.desired_min_yaw, 0.0))
            for turbine in self.turbines:
                turbine.yaw_angle = default_yaw
        if self.wind_process is not None:
            self.wind_process.reset()
        self.state = self._get_state()
        return self.state

    # Implementing a method from the base class. This method renders the environment for the user.
    def render(self, mode='human'):

        if self.state is None:
            return None

        if self.visualization is None:
            self.visualization = FarmVisualization(self.floris_interface)

        return self.visualization.render(return_rgb_array=mode == 'rgb_array')

    # Implementing a method from the base class. This method finalized the environment when it is not used anymore.
    def close(self):
        if self.visualization is not None:
            self.visualization.close()
        self.wind_process.close()

    def get_log_dict(self) -> Dict[str, Union[float, int, str, bool]]:
        """
        Generates a dictionary of data that can be used for logging purposes.
        """
        yaws = self.yaws_from_wind
        yaw_dict = {f'yaw_{i}': yaws[i] for i in range(len(yaws))}
        return {'yaw': yaw_dict}
