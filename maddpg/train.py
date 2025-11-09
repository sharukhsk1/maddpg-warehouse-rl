
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import itertools
import numpy as np
import random
import torch
import csv
from collections import deque
from datetime import datetime

from envs.custom_warehouse import CustomWarehouseEnv
from maddpg.agent import MADDPGAgent
from maddpg.model import Actor, Critic

# Create necessary directories
os.makedirs('maddpg/checkpoints', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_state(s):
    if isinstance(s, tuple):
        processed = []
        for x in s:
            if isinstance(x, dict):
                continue  # Ignore dict elements
            if hasattr(x, 'flatten'):
                arr = x.flatten()
                # flatten again to ensure 1D
                arr = np.ravel(arr)
                processed.append(arr)
            else:
                processed.append(np.array([x]))
        if processed:
            # concatenate all 1D arrays
            flat = np.concatenate(processed)
        else:
            flat = np.array([])
        # flatten whole output once more to ensure 1D
        flat = np.ravel(flat)
        # print(f"flatten_state output shape: {flat.shape}")
        return flat
    else:
        if isinstance(s, dict):
            return np.array([])  # empty if dict
        flat = np.ravel(np.array(s).flatten())  # flatten and ravel to guarantee 1D
        # print(f"flatten_state output shape: {flat.shape}")
        return flat

    

def fully_flatten(obs_part):
    flat_parts = []
    for part in obs_part:
        if isinstance(part, dict):
            continue  # skip dict metadata
        part_array = np.array(part, dtype=np.float32)  # safe conversion to numpy float array
        flat_parts.append(part_array.flatten())
    if flat_parts:
        return np.concatenate(flat_parts)
    else:
        return np.array([], dtype=np.float32)  # empty array if no valid parts

class ReplayBuffer:
    """Fixed size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        assert isinstance(state, np.ndarray) and state.ndim == 1, "State should be 1D numpy array"
        assert isinstance(next_state, np.ndarray) and next_state.ndim == 1, "Next state should be 1D numpy array"
        assert state.shape == next_state.shape, f"State and next_state shapes mismatch: {state.shape} vs {next_state.shape}"
        # print(f"Replay buffer add: state shape {state.shape}, next_state shape {next_state.shape}")

        self.memory.append((state, action, reward, next_state, done))


    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def train_maddpg(n_episodes=1000, max_t=200, print_every=10,
                buffer_size=100000, batch_size=128, gamma=0.99,
                lr_actor=1e-4, lr_critic=1e-3, tau=1e-2,
                checkpoint_path='maddpg/checkpoints/best_model.pth',
                resume_from=None):
    # Initialize environment
    env = CustomWarehouseEnv()

    # Determine state and action sizes
    obs_sample = env.reset()
    # print(f"DEBUG obs_sample type: {type(obs_sample)}")
    # if isinstance(obs_sample, list) or isinstance(obs_sample, tuple):
    #     for i, x in enumerate(obs_sample):
    #         print(f"DEBUG obs_sample[{i}] type: {type(x)} shape: {getattr(x, 'shape', 'no shape')}")
    sample_state = flatten_state(obs_sample[0] if isinstance(obs_sample, list) or isinstance(obs_sample, tuple) else obs_sample)
    # print(f"DEBUG sample_state shape: {sample_state.shape} sample_state: {sample_state}")
    state_size = sample_state.shape[0]


    # Determine action size as before
    action_size = env.action_space[0].n if hasattr(env.action_space, '__getitem__') else env.action_space.n

    print(f"Determined flattened state size: {state_size}")
    print(f"Action size: {action_size}")


    # Initialize agents (multi-agent)
    n_agents = env.n_agents
    agents = [MADDPGAgent(state_size, action_size, random_seed=i,
                          lr_actor=lr_actor, lr_critic=lr_critic,
                          gamma=gamma, tau=tau)
              for i in range(n_agents)]

    replay_buffer = ReplayBuffer(buffer_size, batch_size)

    scores_deque = deque(maxlen=100)
    scores_list = []
    agent_scores_list = [[] for _ in range(n_agents)]
    actor_losses_list = [[] for _ in range(n_agents)]
    critic_losses_list = [[] for _ in range(n_agents)]

    best_score = -np.inf
    start_episode = 0

    # Resume if checkpoint provided
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from)
        for i, agent in enumerate(agents):
            agent.actor_local.load_state_dict(checkpoint[f'agent_{i}_actor'])
            agent.critic_local.load_state_dict(checkpoint[f'agent_{i}_critic'])
            agent.actor_target.load_state_dict(checkpoint[f'agent_{i}_actor_target'])
            agent.critic_target.load_state_dict(checkpoint[f'agent_{i}_critic_target'])
        start_episode = checkpoint.get('episode', 0)
        best_score = checkpoint.get('best_score', -np.inf)
        print(f"Resumed from episode {start_episode} with best score {best_score:.2f}")

    # CSV logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'results/logs/training_log_{timestamp}.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Episode', 'Total_Reward'] + [f'Agent_{i}_Reward' for i in range(n_agents)] + \
                 [f'Agent_{i}_Actor_Loss' for i in range(n_agents)] + \
                 [f'Agent_{i}_Critic_Loss' for i in range(n_agents)]
        writer.writerow(header)

    print(f"Training on {device}")
    print(f"Number of agents: {n_agents}")
    print(f"State size: {state_size}, Action size: {action_size}")

    for episode in range(start_episode, n_episodes):
        obs = env.reset()
        for agent in agents:
            agent.reset()

        episode_rewards = np.zeros(n_agents)
        episode_actor_losses = [[] for _ in range(n_agents)]
        episode_critic_losses = [[] for _ in range(n_agents)]

        for t in range(max_t):
            actions = []
            env_actions = []  # for storing discrete scalar actions to send to env
            for i, agent in enumerate(agents):
                obs_i = obs[i] if isinstance(obs, list) else obs
                state = fully_flatten(obs_i) if isinstance(obs_i, (tuple, list)) else np.array(obs_i).flatten()
                state = state.astype(np.float32)  # Explicit cast to float32
                # print(f"Agent {i} flattened state shape: {state.shape}")
                raw_action = agent.act(state, add_noise=True)

                # Instead of taking argmax or converting to int for replay buffer, keep full vector:
                actions.append(raw_action)  # Store full continuous action vector

                # For passing to env step, if env expects discrete scalar actions,
                # convert separately here:
                if isinstance(raw_action, (np.ndarray, torch.Tensor)):
                    if hasattr(raw_action, 'argmax'):
                        action_scalar = int(raw_action.argmax())
                    elif hasattr(raw_action, 'item'):
                        action_scalar = int(raw_action.item())
                    else:
                        raise ValueError(f"Unsupported action format from agent.act(): {raw_action}")
                else:
                    action_scalar = int(raw_action)

                env_actions.append(action_scalar)  # different list for env discrete actions



            next_obs, rewards, dones, truncated, info = env.step(env_actions)

            for i in range(n_agents):
                state = flatten_state(obs[i] if isinstance(obs, list) else obs)
                next_state = flatten_state(next_obs[i] if isinstance(next_obs, list) else next_obs)
                reward = rewards[i] if isinstance(rewards, (list, np.ndarray)) else rewards
                done = dones[i] if isinstance(dones, (list, np.ndarray)) else dones

                replay_buffer.add(state, raw_action, reward, next_state, done)  # raw_action vector stored
                episode_rewards[i] += reward

            if len(replay_buffer) > batch_size:
                for i, agent in enumerate(agents):
                    experiences = replay_buffer.sample()
                    actor_loss, critic_loss = agent.learn(experiences)
                    episode_actor_losses[i].append(actor_loss)
                    episode_critic_losses[i].append(critic_loss)

            obs = next_obs

            if isinstance(dones, (list, np.ndarray)):
                if all(dones):
                    break
            else:
                if dones:
                    break

        total_score = np.sum(episode_rewards)
        scores_deque.append(total_score)
        scores_list.append(total_score)

        for i in range(n_agents):
            agent_scores_list[i].append(episode_rewards[i])
            actor_losses_list[i].append(np.mean(episode_actor_losses[i]) if episode_actor_losses[i] else 0)
            critic_losses_list[i].append(np.mean(episode_critic_losses[i]) if episode_critic_losses[i] else 0)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [episode, total_score] + list(episode_rewards) + \
                  [np.mean(episode_actor_losses[i]) if episode_actor_losses[i] else 0 for i in range(n_agents)] + \
                  [np.mean(episode_critic_losses[i]) if episode_critic_losses[i] else 0 for i in range(n_agents)]
            writer.writerow(row)

        avg_score = np.mean(scores_deque)
        if avg_score > best_score:
            best_score = avg_score
            checkpoint = {
                'episode': episode,
                'best_score': best_score,
            }
            for i, agent in enumerate(agents):
                checkpoint[f'agent_{i}_actor'] = agent.actor_local.state_dict()
                checkpoint[f'agent_{i}_critic'] = agent.critic_local.state_dict()
                checkpoint[f'agent_{i}_actor_target'] = agent.actor_target.state_dict()
                checkpoint[f'agent_{i}_critic_target'] = agent.critic_target.state_dict()

            torch.save(checkpoint, checkpoint_path)
            print(f'\nNew best model saved at episode {episode} with avg score {best_score:.2f}')

        if episode % print_every == 0:
            print(f"Episode {episode} - Total Avg: {avg_score:.2f}, Total Score: {total_score:.2f}")
            for i in range(n_agents):
                print(f"  Agent_{i} Reward: {agent_scores_list[i][-1]:.3f}, "
                      f"Actor Loss: {actor_losses_list[i][-1]:.3f}, "
                      f"Critic Loss: {critic_losses_list[i][-1]:.3f}")

    env.close()
    print(f'\nTraining completed! Best average score: {best_score:.2f}')
    print(f'Logs saved to {log_file}')

    return scores_list, agent_scores_list, actor_losses_list, critic_losses_list




def run_grid_search():
    # Manually define 5 best configurations
    param_combinations = [
        # Config 0: Your best performing config from previous run
        {'lr_actor': 1e-4, 'lr_critic': 1e-3, 'gamma': 0.95, 'tau': 0.01, 'batch_size': 64, 'buffer_size': 100000},
        
        # Config 1: Slightly higher actor learning rate for faster learning
        {'lr_actor': 5e-4, 'lr_critic': 1e-3, 'gamma': 0.95, 'tau': 0.01, 'batch_size': 64, 'buffer_size': 100000},
        
        # Config 2: Higher gamma for better long-term reward consideration
        {'lr_actor': 1e-4, 'lr_critic': 1e-3, 'gamma': 0.99, 'tau': 0.01, 'batch_size': 64, 'buffer_size': 100000},
        
        # Config 3: Faster target network updates (higher tau)
        {'lr_actor': 1e-4, 'lr_critic': 1e-3, 'gamma': 0.95, 'tau': 0.05, 'batch_size': 64, 'buffer_size': 100000},
        
        # Config 4: Larger batch size for more stable gradients
        {'lr_actor': 1e-4, 'lr_critic': 1e-3, 'gamma': 0.95, 'tau': 0.01, 'batch_size': 128, 'buffer_size': 100000},
    ]

    best_score = -float('inf')
    best_params = None
    best_checkpoint = None

    for i, params in enumerate(param_combinations):
        print(f"Running config {i + 1}/{len(param_combinations)}: {params}")
        checkpoint_path = f"maddpg/checkpoints/best_model_{i}.pth"

        scores, *_ = train_maddpg(
            n_episodes=1000,  # Reduced to 1000 episodes as requested
            max_t=200,
            print_every=200,  # Print more frequently
            lr_actor=params['lr_actor'],
            lr_critic=params['lr_critic'],
            gamma=params['gamma'],
            tau=params['tau'],
            batch_size=params['batch_size'],
            buffer_size=params['buffer_size'],
            checkpoint_path=checkpoint_path
        )

        avg_score = sum(scores[-50:]) / min(50, len(scores))
        print(f"Average score in last 50 episodes: {avg_score:.2f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_checkpoint = checkpoint_path

    print(f"\nBest average score {best_score:.2f} obtained with params: {best_params}")
    print(f"Best checkpoint saved at {best_checkpoint}")



if __name__ == "__main__":
    run_grid_search()
