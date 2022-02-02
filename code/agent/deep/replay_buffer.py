import torch
import collections, random


class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self._buffer = collections.deque(maxlen=buffer_size)
        self._device = device

    def put(self, transition):
        self._buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self._buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)

        return torch.tensor(s_lst, dtype=torch.float, device=self._device),\
               torch.tensor(a_lst, dtype=torch.float, device=self._device),\
               torch.tensor(r_lst, dtype=torch.float, device=self._device),\
               torch.tensor(s_prime_lst, dtype=torch.float, device=self._device)

    def size(self):
        return len(self._buffer)
