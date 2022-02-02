from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self,
                 state_shape,
                 action_shape,
                 hidden_sizes: Union[int, Tuple[int]] = (128, 128),
                 device=None):
        super().__init__()
        self._device = device
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        else:
            hidden_sizes = list(hidden_sizes)
        hidden_sizes.insert(0, state_shape)
        if len(hidden_sizes) > 1:
            self._fcs = [
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i], device=self._device)
                for i in range(1, len(hidden_sizes))
            ]
        else:
            self._fcs = []
        self._fc_out = nn.Linear(hidden_sizes[-1], action_shape, device=self._device)

    def forward(self, x):
        for fc in self._fcs:
            x = F.relu(fc(x))
        x = self._fc_out(x)
        return x
