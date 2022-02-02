from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_shape,
                 action_shape,
                 hidden_sizes_s: Union[int, Tuple[int, ...]] = 128,
                 hidden_sizes_a: Union[int, Tuple[int, ...]] = 128,
                 hidden_sizes_shared: Union[int, Tuple[int, ...]] = 256,
                 device=None):
        super().__init__()
        self._device = device
        if isinstance(hidden_sizes_s, int):
            hidden_sizes_s = [hidden_sizes_s]
        else:
            hidden_sizes_s = list(hidden_sizes_s)
        hidden_sizes_s.insert(0, state_shape)
        if len(hidden_sizes_s) > 1:
            self._fcs_s = [
                nn.Linear(hidden_sizes_s[i-1], hidden_sizes_s[i], device=self._device)
                for i in range(1, len(hidden_sizes_s))
            ]
        else:
            self._fcs_s = []
        if isinstance(hidden_sizes_a, int):
            hidden_sizes_a = [hidden_sizes_a]
        else:
            hidden_sizes_a = list(hidden_sizes_a)
        hidden_sizes_a.insert(0, action_shape)
        if len(hidden_sizes_a) > 1:
            self._fcs_a = [
                nn.Linear(hidden_sizes_a[i-1], hidden_sizes_a[i], device=self._device)
                for i in range(1, len(hidden_sizes_a))
            ]
        else:
            self._fcs_a = []
        if isinstance(hidden_sizes_shared, int):
            hidden_sizes_shared = [hidden_sizes_shared]
        else:
            hidden_sizes_shared = list(hidden_sizes_shared)
        hidden_sizes_shared.insert(0, hidden_sizes_s[-1] + hidden_sizes_a[-1])
        if len(hidden_sizes_shared) > 1:
            self._fcs_shared = [
                nn.Linear(hidden_sizes_shared[i-1], hidden_sizes_shared[i], device=self._device)
                for i in range(1, len(hidden_sizes_shared))
            ]
        else:
            self._fcs_shared = []
        self.fc_out = nn.Linear(hidden_sizes_shared[-1], 1, device=self._device)

    def forward(self, x, a):
        for fc_s in self._fcs_s:
            x = F.relu(fc_s(x))
        for fc_a in self._fcs_a:
            a = F.relu(fc_a(a))
        q = torch.cat([x, a], dim=1)
        for fc in self._fcs_shared:
            q = F.relu(fc(q))
        q = self.fc_out(q)
        return q

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self._tau) + param.data * self._tau)
