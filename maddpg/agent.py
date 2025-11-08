import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from maddpg.model import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

class MADDPGAgent:
    """MADDPG Agent with Actor-Critic and exploration noise."""

    def __init__(self, state_size, action_size, random_seed, lr_actor=1e-4, lr_critic=1e-3,
                 weight_decay=0, gamma=0.99, tau=1e-2):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.gamma = gamma
        self.tau = tau

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Initialize targets same as local
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        # print(f"State dtype: {state.dtype}, type: {type(state)}, shape: {state.shape}")
        # print(f"State sample elements: {state[:10]}")  # print first 10 elements
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dim
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)



    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        if actions.dim() == 3 and actions.size(1) == 1:
            actions = actions.squeeze(1)  # Convert [batch, 1, action_dim] to [batch, action_dim]
        elif actions.dim() == 1:
            actions = actions.unsqueeze(1)  # reshape to 2D if needed
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(device)

        # Fix dimension mismatch for actions
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # Add batch dimension if missing

        if states.dim() == 1:
            states = states.unsqueeze(0)

        if next_states.dim() == 1:
            next_states = next_states.unsqueeze(0)

        # ---------------------------- update critic ---------------------------- #
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        return actor_loss.item(), critic_loss.item()


    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self, target, source):
        """Hard update model parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
