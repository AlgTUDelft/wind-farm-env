from abc import ABC, abstractmethod

from gym import Env
from typing import Optional, Tuple, List

import numpy as np
import progressbar
from gym.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter


class Agent(ABC):

    def __init__(self, name, type, env: Env):
        self.name = name
        self.type = type
        assert env is not None, f'Agent {name} is not provided with an environment'
        self._env = env
        if isinstance(self._env.observation_space, Box):
            self.observation_shape = self._env.observation_space.shape[0]
        elif isinstance(self._env.observation_space, Discrete):
            self.observation_shape = self._env.observation_space.n
        self.action_shape = self._env.action_space.shape[0]

    @abstractmethod
    def find_action(self, observation, in_eval=False):
        pass

    @abstractmethod
    def learn(self, observation, action, reward, next_observation, global_step):
        pass

    def get_parameters(self) -> dict:
        return {}

    @abstractmethod
    def get_log_dict(self):
        return {}

    def run(
            self,
            total_steps: int = 10000,
            render: bool = False,
            rescale_rewards: bool = True,
            reward_range: Optional[Tuple[float, float]] = None,
            log: bool = True,
            log_every: int = 1,
            log_directory: Optional[str] = None,
            eval_envs: Optional[List[Env]] = None,
            eval_steps: Optional[int] = 1000,
            eval_every: Optional[int] = 1000,
            eval_once: bool = False,  # for non-learning agents,
            eval_only: bool = False  # will repeat evaluations but skip training
    ):
        if rescale_rewards:
            if reward_range is None:
                reward_range = self._env.reward_range
            lowest_reward = min(reward_range)
            reward_delta = max(reward_range) - lowest_reward
        if log_directory is None or log_every <= 0:
            log = False
        if log:
            writer = SummaryWriter(log_directory)
            parameter_dictionary = self.get_parameters()
            self._log_dict(parameter_dictionary, writer)

        do_eval = (eval_envs is not None) and (eval_steps is not None) and (eval_every is not None) and log
        eval_rewards = None

        total_reward = 0.0
        log_reward = 0.0
        observation = self._env.reset()
        env_log_exists = hasattr(self._env, 'get_log_dict') and callable(getattr(self._env, 'get_log_dict'))
        for global_step in progressbar.progressbar(range(total_steps)):
            if not eval_only:
                action = self.find_action(observation)
                do_logging = log and (global_step + 1) % log_every == 0

                if do_logging:
                    if env_log_exists:
                        self._log_dict(self._env.get_log_dict(), writer, global_step)
                    self._log_dict(self.get_log_dict(), writer, global_step)
                    self._log_value('observation', observation, writer, global_step)
                    self._log_value('action', action, writer, global_step)

                old_observation = observation
                observation, reward, _, _ = self._env.step(action)

                if rescale_rewards:
                    reward = (reward - lowest_reward) / reward_delta

                log_reward += reward

                if do_logging:
                    total_reward += log_reward
                    self._log_value('reward/instantaneous', log_reward, writer, global_step)
                    self._log_value('reward/cumulative', total_reward, writer, global_step)
                    if rescale_rewards:
                        self._log_value('reward/MWh',
                                        log_reward * reward_delta + lowest_reward * log_every,
                                        writer, global_step)
                        self._log_value('reward/MWh_cumulative',
                                        total_reward * reward_delta + lowest_reward * log_every,
                                        writer, global_step)
                    log_reward = 0.0

                self.learn(old_observation, action, reward, observation, global_step)

                if render:
                    self._env.render()

            if do_eval and (global_step + 1) % eval_every == 0:
                # evaluate the agent
                if (not eval_once) or (eval_rewards is None):
                    eval_rewards = np.array([self._eval(env, eval_steps) for env in eval_envs])
                    if rescale_rewards:
                        eval_rewards = (eval_rewards - lowest_reward) / reward_delta
                self._log_value('eval/mean', np.mean(eval_rewards), writer, global_step)
                self._log_value('eval/std', np.std(eval_rewards), writer, global_step)
                for reward, i in zip(eval_rewards, range(len(eval_rewards))):
                    self._log_value(f'eval/reward_{i}', reward, writer, global_step)

    def _eval(self, env: Env, eval_steps):
        observation = env.reset()
        eval_reward = 0.0
        print('\nEvaluating...')
        for _ in progressbar.progressbar(range(eval_steps)):
            action = self.find_action(observation, in_eval=True)
            observation, reward, _, _ = env.step(action)
            eval_reward += reward
        return eval_reward

    def _log_value(self, tag, value, writer, global_step=0):
        if isinstance(value, dict):
            for k, v in value.items():
                writer.add_scalar(f'{tag}/{k}', v, global_step)
        elif isinstance(value, (list, np.ndarray)):
            for i in range(len(value)):
                writer.add_scalar(f'{tag}/{i}', value[i], global_step)
        else:
            writer.add_scalar(tag, value, global_step)

    def _log_dict(self, dictionary: dict, writer, global_step=0):
        for tag, value in dictionary.items():
            self._log_value(tag, value, writer, global_step)

    def close(self):
        self._env.close()
