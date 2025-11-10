import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor for DISCRETE actions: outputs raw logits (no softmax)."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        super().__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self._hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self._hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits

    @staticmethod
    def _hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1.0 / (fan_in ** 0.5)
        return (-lim, lim)


class Critic(nn.Module):
    """
    Centralized Critic for MADDPG:
      Input = concat(all agents' states), concat(all agents' one-hot actions)
      Output = scalar Q-value.
    """

    def __init__(self, n_agents, state_size, action_size, seed, fcs1_units=256, fc2_units=256):
        super().__init__()
        torch.manual_seed(seed)

        self.n_agents = n_agents
        self.total_state = n_agents * state_size
        self.total_action = n_agents * action_size

        self.fcs1 = nn.Linear(self.total_state, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + self.total_action, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*self._hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*self._hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states_all_flat, actions_all_flat):
        """
        states_all_flat : (B, n_agents*state_size)
        actions_all_flat: (B, n_agents*action_size)  (one-hot per agent)
        """
        xs = F.relu(self.fcs1(states_all_flat))
        x = torch.cat([xs, actions_all_flat], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    @staticmethod
    def _hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1.0 / (fan_in ** 0.5)
        return (-lim, lim)
