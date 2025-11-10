import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from maddpg.model import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPGAgent:
    """One MADDPG agent (local actor, centralized critic shared structure)."""

    def __init__(self, n_agents, state_size, action_size, agent_id, seed=0,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=1e-2):

        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau

        # Actor
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Centralized Critic
        self.critic_local = Critic(n_agents, state_size, action_size, seed).to(device)
        self.critic_target = Critic(n_agents, state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)

        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

    # -------- ACTING --------
    def act(self, state, epsilon=0.1):
        """Epsilon-greedy over Actor logits."""
        state_t = torch.from_numpy(state).float().to(device).unsqueeze(0)

        self.actor_local.eval()
        with torch.no_grad():
            logits = self.actor_local(state_t)
            if np.random.rand() < epsilon:
                action = np.random.randint(0, self.action_size)
            else:
                # argmax(logits) == argmax(softmax(logits))
                action = torch.argmax(logits, dim=-1).item()
        self.actor_local.train()

        return int(action)

    def reset(self):
        pass

    # -------- LEARNING --------
    def learn(self, experiences, all_agents):
        """
        experiences (from centralized buffer):
           states_all        (B, n_agents, state_size)
           actions_all       (B, n_agents, action_size one-hot)
           rewards_all       (B, n_agents)
           next_states_all   (B, n_agents, state_size)
           dones_all         (B, n_agents)
        """
        (states, actions, rewards, next_states, dones) = experiences

        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

        B = states.size(0)

        # Flatten for centralized critic
        states_flat = states.reshape(B, -1)
        actions_flat = actions.reshape(B, -1)
        next_states_flat = next_states.reshape(B, -1)

        # -------- Critic update --------
        with torch.no_grad():
            next_onehots = []
            for i, ag in enumerate(all_agents):
                logits_i = ag.actor_target(next_states[:, i, :])
                next_a_i = torch.argmax(logits_i, dim=-1)                           # (B,)
                next_oh_i = F.one_hot(next_a_i, self.action_size).float()           # (B, A)
                next_onehots.append(next_oh_i)
            next_actions_flat = torch.cat(next_onehots, dim=1)                       # (B, n*A)

            Q_targets_next = self.critic_target(next_states_flat, next_actions_flat) # (B,1)
            Q_targets = rewards[:, self.agent_id].unsqueeze(1) + \
                        self.gamma * Q_targets_next * (1.0 - dones[:, self.agent_id].unsqueeze(1))

        Q_expected = self.critic_local(states_flat, actions_flat)

        Q_targets = torch.clamp(Q_targets, -10.0, 10.0)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- Actor update (only this agent's action is re-decided) --------
        my_state = states[:, self.agent_id, :]                                        # (B,S)
        logits = self.actor_local(my_state)
        my_action_onehot = F.gumbel_softmax(logits, tau=1.0, hard=True)               # (B,A)

        new_actions = actions.clone()
        new_actions[:, self.agent_id, :] = my_action_onehot
        new_actions_flat = new_actions.reshape(B, -1)

        actor_loss = -self.critic_local(states_flat, new_actions_flat).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------- Target soft-update --------
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    # -------- UTILS --------
    def soft_update(self, local, target):
        for t, s in zip(target.parameters(), local.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def hard_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)
