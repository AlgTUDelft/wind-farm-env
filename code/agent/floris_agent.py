from floris.tools.floris_interface import FlorisInterface
from floris.tools.optimization.scipy.yaw import YawOptimization

from wind_farm_gym import WindFarmEnv
from .agent import Agent
import numpy as np


class FlorisAgent(Agent):

    def __init__(self, name, env: WindFarmEnv, floris: FlorisInterface = None):
        super().__init__(name, 'FLORIS', env)
        self._stored_representation = env.action_representation
        env.action_representation = 'wind'
        self._opt_options = {'disp': False}
        self._min_yaw, self._max_yaw = env.desired_min_yaw, env.desired_max_yaw
        if self._env.wind_process is None:
            self._opt_yaws = None
        if floris is None:
            # use FLORIS parameters from the env
            self._floris_interface = FlorisInterface(env.floris_interface.input_file)
        elif isinstance(floris, FlorisInterface):
            self._floris_interface = floris
        elif isinstance(floris, str):
            self._floris_interface = FlorisInterface(floris)
        self.turbine_layout = env.turbine_layout
        self._floris_interface.reinitialize_flow_field(layout_array=self.turbine_layout)
        self._is_learning = False

    def find_action(self, observation, in_eval=False):
        # if not in_eval:
        #     return list(np.zeros(self.action_shape))
        parameters = {}
        for s, desc in zip(observation, self._env.observed_variables):
            val = desc['min'] + s * (desc['max'] - desc['min'])
            if desc['type'] != 'yaw':
                if desc.get('index') is None:
                    # these are global; they take priority
                    parameters[desc['type']] = val
                else:
                    data = parameters.get(desc['type'], [])
                    data.append(val)
                    parameters[desc['type']] = data
        if 'wind_speed' in parameters.keys():
            # the point with maximum wind speed is not affected by wakes; use it for reference
            i = np.argmax(parameters['wind_speed'])
            parameters = {k: v[i] if isinstance(v, list) else v for k, v in parameters.items()}
        else:
            parameters = {k: np.mean(v) for k, v in parameters.items()}
        self._floris_interface.reinitialize_flow_field(**parameters)
        return self.optimal_action()

    def learn(self, observation, action, reward, next_observation, global_step):
        pass

    def get_log_dict(self):
        return {}

    def optimal_action(self):
        if self._env.wind_process is None:
            if self._opt_yaws is None:
                self._opt_yaws = YawOptimization(self._floris_interface,
                                                 self._min_yaw,
                                                 self._max_yaw,
                                                 opt_options=self._opt_options).optimize(False)
            opt_yaws = self._opt_yaws
        else:
            opt_yaws = YawOptimization(self._floris_interface,
                                       self._min_yaw,
                                       self._max_yaw,
                                       opt_options=self._opt_options).optimize(False)
        return self.yaws_to_action(opt_yaws)

    def yaws_to_action(self, opt_yaws):
        return [
            (min(self._max_yaw, max(self._min_yaw, yaw)) - self._min_yaw) * 2 / (self._max_yaw - self._min_yaw) - 1
            for yaw in opt_yaws
        ]

    def close(self):
        self._env.action_representation = self._stored_representation
        super().close()
