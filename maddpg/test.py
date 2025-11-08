import torch
import numpy as np
from envs.custom_warehouse import WarehouseEnv
from maddpg.agent import MADDPGAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_maddpg(checkpoint_path, n_episodes=10, max_t=200, render=False):
    """
    Evaluate the trained MADDPG agents on unseen episodes.

    Args:
        checkpoint_path: Path to the saved MADDPG checkpoint
        n_episodes: Number of test episodes
        max_t: Maximum timesteps per episode
        render: Whether to render the environment
    """

    env = WarehouseEnv()
    state_size = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 100
    action_size = env.action_space.n if hasattr(env.action_space, 'n') else 8
    n_agents = env.n_agents

    agents = [MADDPGAgent(state_size, action_size, random_seed=i) for i in range(n_agents)]

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    for i, agent in enumerate(agents):
        agent.actor_local.load_state_dict(checkpoint[f'agent_{i}_actor'])
        agent.critic_local.load_state_dict(checkpoint[f'agent_{i}_critic'])
        agent.actor_target.load_state_dict(checkpoint[f'agent_{i}_actor_target'])
        agent.critic_target.load_state_dict(checkpoint[f'agent_{i}_critic_target'])

    total_rewards = np.zeros(n_agents)

    for episode in range(n_episodes):
        obs = env.reset()
        for agent in agents:
            agent.reset()

        episode_rewards = np.zeros(n_agents)

        for t in range(max_t):
            actions = []
            for i, agent in enumerate(agents):
                state = obs[i] if isinstance(obs, list) else obs
                action = agent.act(state, add_noise=False)
                actions.append(action)

            next_obs, rewards, dones, _ = env.step(actions)

            if render:
                env.render()

            for i in range(n_agents):
                reward = rewards[i] if isinstance(rewards, (list, np.ndarray)) else rewards
                episode_rewards[i] += reward

            obs = next_obs

            if isinstance(dones, (list, np.ndarray)):
                if all(dones):
                    break
            else:
                if dones:
                    break

        print(f'Episode {episode+1}/{n_episodes} Rewards: {episode_rewards}')
        total_rewards += episode_rewards

    avg_rewards = total_rewards / n_episodes
    print(f'\nAverage rewards over {n_episodes} episodes per agent: {avg_rewards}')

    env.close()

if __name__ == "__main__":
    test_maddpg(checkpoint_path='maddpg/checkpoints/best_model.pth', n_episodes=20, render=False)
