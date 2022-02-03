from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym import Env
from gym.spaces import Box

from ..agent import Agent
from . import ReplayBuffer
from .actor import Actor
from .critic import Critic
from .polyak_update import polyak_update


class TD3Agent(Agent):

    def __init__(self, name, env: Env,
                 discounting_factor: float = 0.99,
                 batch_size: int = 32,
                 buffer_size: int = 50000,
                 start_learning: int = 1000,
                 learning_rate_actor: float = 0.0005,
                 learning_rate_critic: float = 0.001,
                 polyak_tau: float = 0.01,
                 hidden_sizes_s: Union[int, Tuple[int, ...]] = 128,
                 hidden_sizes_a: Union[int, Tuple[int, ...]] = 128,
                 hidden_sizes_shared: Union[int, Tuple[int, ...]] = 256,
                 hidden_sizes_actor: Union[int, Tuple[int, ...]] = (128, 128),
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 max_grad_norm: float = 0.5,
                 exploration_noise: float = 0.1,
                 policy_update_frequency: int = 10,
                 target_update_frequency: int = 10
                 ):
        super().__init__(name, 'TD3', env)
        assert isinstance(self._env.action_space, Box), "Action space must be of type Box"
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._gamma = discounting_factor
        self._memory = ReplayBuffer(buffer_size, self._device)
        self.q1 = Critic(self.observation_shape,
                         self.action_shape,
                         hidden_sizes_s,
                         hidden_sizes_a,
                         hidden_sizes_shared,
                         self._device)
        self.q2 = Critic(self.observation_shape,
                         self.action_shape,
                         hidden_sizes_s,
                         hidden_sizes_a,
                         hidden_sizes_shared,
                         self._device)
        self.q1_target = Critic(self.observation_shape,
                                self.action_shape,
                                hidden_sizes_s,
                                hidden_sizes_a,
                                hidden_sizes_shared,
                                self._device)
        self.q2_target = Critic(self.observation_shape,
                                self.action_shape,
                                hidden_sizes_s,
                                hidden_sizes_a,
                                hidden_sizes_shared,
                                self._device)
        self.pi = Actor(self.observation_shape,
                        self.action_shape,
                        hidden_sizes_actor,
                        self._device)
        self.pi_target = Actor(self.observation_shape,
                        self.action_shape,
                        hidden_sizes_actor,
                        self._device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.pi_target.load_state_dict(self.pi.state_dict())
        self.q1_target.train(False)
        self.q2_target.train(False)
        self.pi_target.train(False)
        self._q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate_critic)
        self._pi_optimizer = optim.Adam(list(self.pi.parameters()), lr=learning_rate_actor)
        self._batch_size = batch_size
        self._start_learning = max(start_learning, batch_size)
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._max_grad_norm = max_grad_norm
        self._exploration_noise = exploration_noise
        self._policy_update_frequency = policy_update_frequency
        self._target_update_frequency = target_update_frequency
        self._tau = polyak_tau
        self._q_loss = torch.Tensor([0.0], device=self._device)
        self._pi_loss = torch.Tensor([0.0], device=self._device)
        self._a_limits = torch.Tensor(self._env.action_space.low, device=self._device),\
                         torch.Tensor(self._env.action_space.high, device=self._device)

    def find_action(self, observation, in_eval=False):
        with torch.no_grad():
            a = self.pi(torch.tensor(observation, dtype=torch.float, device=self._device)).detach().numpy()
            if not in_eval:
                a += np.random.normal(0, self._exploration_noise, size=self.action_shape)
                a = a.clip(self._env.action_space.low, self._env.action_space.high)
        return a.tolist()

    def learn(self, observation, action, reward, next_observation, global_step):
        self._memory.put((observation, action, reward, next_observation))

        if self._memory.size() > self._start_learning:
            s, a, r, s_prime = self._memory.sample(self._batch_size)
            with torch.no_grad():
                clipped_noise = torch.randn_like(a, device=self._device) * self._policy_noise
                clipped_noise = clipped_noise.clamp(-self._noise_clip, self._noise_clip)
                a_prime = self.pi_target(s_prime) + clipped_noise
                a_prime = a_prime.clamp(*self._a_limits)
                qf1_next_target = self.q1_target(s_prime, a_prime)
                qf2_next_target = self.q2_target(s_prime, a_prime)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = r + self._gamma * min_qf_next_target

            q1_l = F.mse_loss(self.q1(s, a), next_q_value)
            q2_l = F.mse_loss(self.q2(s, a), next_q_value)
            self._q_loss = 0.5 * (q1_l + q2_l)
            # optimize the model
            self._q_optimizer.zero_grad()
            self._q_loss.backward()
            nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self._max_grad_norm)
            self._q_optimizer.step()

            if (global_step + 1) % self._policy_update_frequency == 0:
                self._pi_loss = -self.q1(s, self.pi(s)).mean()
                self._pi_optimizer.zero_grad()
                self._pi_loss.backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()), self._max_grad_norm)
                self._pi_optimizer.step()

            if (global_step + 1) % self._target_update_frequency == 0:
                polyak_update(self.q1.parameters(), self.q1_target.parameters(), self._tau)
                polyak_update(self.q2.parameters(), self.q2_target.parameters(), self._tau)
                polyak_update(self.pi.parameters(), self.pi_target.parameters(), self._tau)

    def get_log_dict(self):
        return {
            'loss/q_loss': self._q_loss.item(),
            'loss/pi_loss': self._pi_loss.item()
        }
