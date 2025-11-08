import matplotlib.pyplot as plt
import csv
import os
import numpy as np

def plot_training_logs(log_file):
    episodes = []
    total_rewards = []
    per_agent_rewards = {}
    actor_losses = {}
    critic_losses = {}

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Identify agents count from headers
        agent_count = sum('Agent_' in name and '_Reward' in name for name in fieldnames)

        for i in range(agent_count):
            per_agent_rewards[i] = []
            actor_losses[i] = []
            critic_losses[i] = []

        for row in reader:
            episodes.append(int(row['Episode']))
            total_rewards.append(float(row['Total_Reward']))
            for i in range(agent_count):
                per_agent_rewards[i].append(float(row[f'Agent_{i}_Reward']))
                actor_losses[i].append(float(row[f'Agent_{i}_Actor_Loss']))
                critic_losses[i].append(float(row[f'Agent_{i}_Critic_Loss']))

    # Plot average total reward
    plt.figure(figsize=(12,6))
    plt.plot(episodes, total_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/total_reward_curve.png')
    plt.show()

    # Plot rewards per agent
    plt.figure(figsize=(12,6))
    for i in range(agent_count):
        plt.plot(episodes, per_agent_rewards[i], label=f'Agent {i}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Per-Agent Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/per_agent_reward_curve.png')
    plt.show()

    # Plot actor losses per agent
    plt.figure(figsize=(12,6))
    for i in range(agent_count):
        plt.plot(episodes, actor_losses[i], label=f'Agent {i}')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.title('Actor Network Loss per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/actor_loss_curve.png')
    plt.show()

    # Plot critic losses per agent
    plt.figure(figsize=(12,6))
    for i in range(agent_count):
        plt.plot(episodes, critic_losses[i], label=f'Agent {i}')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.title('Critic Network Loss per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/critic_loss_curve.png')
    plt.show()

if __name__ == "__main__":
    # Provide the path of your latest log file
    log_folder = 'results/logs/'
    log_files = sorted([os.path.join(log_folder, f) for f in os.listdir(log_folder) if f.endswith('.csv')])
    if not log_files:
        print("No log files found in results/logs/. Please run training first.")
    else:
        latest_log = log_files[-1]
        print(f"Visualizing log: {latest_log}")
        plot_training_logs(latest_log)
