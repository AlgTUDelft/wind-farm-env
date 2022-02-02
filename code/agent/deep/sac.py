from typing import Union, Tuple, Optional

import torch
import torch.optim as optim
import torch.nn.functional as F
from gym import Env

from agent.agent import Agent
from . import ReplayBuffer
from .polyak_update import polyak_update
from .soft_actor import SoftActor
from .critic import Critic


class SACAgent(Agent):

    def __init__(self, name, env: Env,
                 discounting_factor: float = 0.99,
                 batch_size: int = 32,
                 buffer_size: int = 50000,
                 start_learning: int = 1000,
                 learning_rate_actor: float = 0.0005,
                 learning_rate_critic: float = 0.001,
                 polyak_tau: float = 0.05,
                 hidden_sizes_s: Union[int, Tuple[int, ...]] = 128,
                 hidden_sizes_a: Union[int, Tuple[int, ...]] = 128,
                 hidden_sizes_shared: Union[int, Tuple[int, ...]] = 256,
                 hidden_sizes_actor: Union[int, Tuple[int, ...]] = (128, 128),
                 init_alpha: float = 1.0,
                 tune_alpha: bool = True,
                 learning_rate_alpha: float = 0.001,
                 target_entropy: Optional[float] = None,
                 target_update_frequency: int = 2
                 ):
        super().__init__(name, 'SAC', env)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._gamma = discounting_factor
        self._memory = ReplayBuffer(buffer_size, self._device)

        # action rescaling
        self._action_scale = torch.Tensor(
            (self._env.action_space.high - self._env.action_space.low) / 2., device=self._device)
        self._action_bias = torch.Tensor(
            (self._env.action_space.high + self._env.action_space.low) / 2., device=self._device)

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
        self.pi = SoftActor(self.observation_shape,
                            self.action_shape,
                            self._action_scale,
                            self._action_bias,
                            hidden_sizes_actor,
                            self._device)
        self.pi_target = SoftActor(self.observation_shape,
                                   self.action_shape,
                                   self._action_scale,
                                   self._action_bias,
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
        self._tune = tune_alpha
        if self._tune:
            if target_entropy is None:
                self._target_entropy = -torch.prod(torch.Tensor(env.action_space.shape, device=self._device)).item()
            else:
                self._target_entropy = target_entropy
            self._log_alpha = torch.log(torch.Tensor([init_alpha], device=self._device)).requires_grad_(True)
            self._alpha_optimizer = optim.Adam([self._log_alpha], lr=learning_rate_alpha)
        else:
            self._alpha = torch.Tensor([init_alpha], device=self._device)

        self._target_update_frequency = target_update_frequency
        self._tau = polyak_tau

        self._q_loss = torch.Tensor([0.0], device=self._device)
        self._pi_loss = torch.Tensor([0.0], device=self._device)
        self._alpha_loss = torch.Tensor([0.0], device=self._device)

    def find_action(self, observation, in_eval=False):
        with torch.no_grad():
            a, _ = self.pi(
                torch.Tensor(observation, device=self._device),
                deterministic=in_eval, with_logprob=False
            )
        return a.tolist()

    def learn(self, observation, action, reward, next_observation, global_step):
        self._memory.put((observation, action, reward, next_observation))

        if self._memory.size() > self._start_learning:
            s, a, r, s_prime = self._memory.sample(self._batch_size)
            with torch.no_grad():
                a_prime, a_prime_log_pi = self.pi_target(s_prime)
                qf1_next_target = self.q1_target(s_prime, a_prime)
                qf2_next_target = self.q2_target(s_prime, a_prime)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                min_qf_next_target -= self._get_alpha() * a_prime_log_pi.unsqueeze(1)
                next_q_value = r + self._gamma * min_qf_next_target

            q1_l = F.mse_loss(self.q1(s, a), next_q_value)
            q2_l = F.mse_loss(self.q2(s, a), next_q_value)
            self._q_loss = 0.5 * (q1_l + q2_l)
            # optimize the model
            self._q_optimizer.zero_grad()
            self._q_loss.backward()
            self._q_optimizer.step()

            a_pi, log_a_pi = self.pi(s)
            qf1 = self.q1(s, a_pi)
            qf2 = self.q2(s, a_pi)
            min_qf = torch.min(qf1, qf2)
            self._pi_loss = (self._get_alpha() * log_a_pi.unsqueeze(1) - min_qf).mean()
            self._pi_optimizer.zero_grad()
            self._pi_loss.backward()
            self._pi_optimizer.step()

            if self._tune:
                self._alpha_loss = -(self._log_alpha * (log_a_pi + self._target_entropy).detach()).mean()
                self._alpha_optimizer.zero_grad()
                self._alpha_loss.backward()
                self._alpha_optimizer.step()

            if (global_step + 1) % self._target_update_frequency == 0:
                polyak_update(self.q1.parameters(), self.q1_target.parameters(), self._tau)
                polyak_update(self.q2.parameters(), self.q2_target.parameters(), self._tau)
                polyak_update(self.pi.parameters(), self.pi_target.parameters(), self._tau)

    def get_log_dict(self):
        return {
            'loss/q_loss': self._q_loss.item(),
            'loss/pi_loss': self._pi_loss.item(),
            'loss/alpha_loss': self._alpha_loss.item(),
            'loss/alpha': self._get_alpha().item()
        }

    def _get_alpha(self):
        return torch.exp(self._log_alpha.detach()) if self._tune else self._alpha
