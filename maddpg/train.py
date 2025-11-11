# File: maddpg/train_grid.py
import os, sys, csv, random, math, time
from collections import deque, defaultdict
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from envs.custom_warehouse import CustomWarehouseEnv
from maddpg.agent import MADDPGAgent

# ------------------------- IO setup ------------------------------
os.makedirs('maddpg/checkpoints', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- reproducibility -----------------------
def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (can slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------- Replay Buffer -------------------------
from collections import deque as _deque
class MultiAgentReplayBuffer:
    def __init__(self, n_agents, state_size, action_size, buffer_size, batch_size, seed=0):
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.memory = _deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, states, actions, rewards, next_states, done):
        onehot_actions = []
        for a in actions:
            oh = np.zeros(self.action_size, dtype=np.float32)
            oh[a] = 1.0
            onehot_actions.append(oh)
        self.memory.append((
            np.stack(states, axis=0),
            np.stack(onehot_actions, axis=0),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.full((self.n_agents,), bool(done), dtype=np.bool_)
        ))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states       = np.stack([b[0] for b in batch], axis=0)
        actions      = np.stack([b[1] for b in batch], axis=0)
        rewards      = np.stack([b[2] for b in batch], axis=0)
        next_states  = np.stack([b[3] for b in batch], axis=0)
        dones        = np.stack([b[4] for b in batch], axis=0)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# ------------------------- Metrics -------------------------------
def action_oscillation(actions_episode):
    """
    actions_episode: list of list[int] length T; each inner list has n_agents actions at step t.
    We compute per-agent switch rate and return mean switch rate over agents.
    """
    if len(actions_episode) <= 1: return 0.0
    actions = np.array(actions_episode)  # (T, n_agents)
    T, nA = actions.shape
    switches = (actions[1:] != actions[:-1]).sum(axis=0)  # per agent
    switch_rate_per_agent = switches / max(1, T-1)       # [0..1]
    return float(np.mean(switch_rate_per_agent))

def safe_get_info(info, key, default=0):
    try:
        if isinstance(info, dict):
            return info.get(key, default)
        return default
    except Exception:
        return default

# ------------------------- Plotting ------------------------------
def plot_training_results(scores_list, agent_scores_list, actor_losses_list, critic_losses_list,
                          osc_list, tasks_list, n_agents, title, save_path):
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35, height_ratios=[1, 1, 1])

    episodes = list(range(len(scores_list)))

    # Total reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, scores_list, linewidth=2, label='Total Reward')
    if len(scores_list) >= 10:
        mv = np.convolve(scores_list, np.ones(10) / 10, mode='valid')
        ax1.plot(range(9, len(scores_list)), mv, linewidth=3, label='Moving Avg (10)')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward'); ax1.set_title('Training Progress: Total Reward')
    ax1.grid(True, alpha=0.3, linestyle='--'); ax1.legend()

    # Per-agent rewards
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(n_agents):
        ax2.plot(episodes, agent_scores_list[i], linewidth=1.5, label=f'Agent {i}')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Reward'); ax2.set_title('Per-Agent Rewards')
    ax2.grid(True, alpha=0.3, linestyle='--'); ax2.legend(ncol=2)

    # Actor losses
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(n_agents):
        ax3.plot(episodes, actor_losses_list[i], linewidth=1.5, label=f'Agent {i}')
    ax3.set_xlabel('Episode'); ax3.set_ylabel('Actor Loss'); ax3.set_title('Actor Losses')
    ax3.grid(True, alpha=0.3, linestyle='--'); ax3.legend(ncol=2)

    # Critic losses
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(n_agents):
        ax4.plot(episodes, critic_losses_list[i], linewidth=1.5, label=f'Agent {i}')
    ax4.set_xlabel('Episode'); ax4.set_ylabel('Critic Loss'); ax4.set_title('Critic Losses')
    ax4.grid(True, alpha=0.3, linestyle='--'); ax4.legend(ncol=2)

    # Oscillation + Tasks
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(episodes, osc_list, linewidth=2, label='Action Oscillation (lower=better)')
    if len(osc_list) >= 10:
        mv = np.convolve(osc_list, np.ones(10)/10, mode='valid')
        ax5.plot(range(9, len(osc_list)), mv, linewidth=3, label='Moving Avg (10)')
    ax5.set_xlabel('Episode'); ax5.set_ylabel('Switch Rate'); ax5.set_title('Movement Smoothness')
    ax5.grid(True, alpha=0.3, linestyle='--'); ax5.legend()

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(episodes, tasks_list, linewidth=2, label='Tasks Completed / Episode')
    ax6.set_xlabel('Episode'); ax6.set_ylabel('Tasks'); ax6.set_title('Productivity (Env-reported)')
    ax6.grid(True, alpha=0.3, linestyle='--'); ax6.legend()

    fig.suptitle(title, fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {save_path}")

def plot_grid_comparison(grid_summary_rows, save_path):
    """
    grid_summary_rows: list of dicts with keys:
        'name', 'avg_reward', 'avg_tasks', 'avg_osc', 'best_avg100', 'episodes'
    """
    if not grid_summary_rows:
        return
    names = [r['name'] for r in grid_summary_rows]
    avg_reward = [r['avg_reward'] for r in grid_summary_rows]
    avg_tasks  = [r['avg_tasks']  for r in grid_summary_rows]
    avg_osc    = [r['avg_osc']    for r in grid_summary_rows]

    x = np.arange(len(names))
    width = 0.28

    fig = plt.figure(figsize=(22, 8))
    # Reward & Tasks bar
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x - width/2, avg_reward, width, label='Avg Reward')
    ax1.bar(x + width/2, avg_tasks,  width, label='Avg Tasks')
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha='right')
    ax1.set_title('Grid: Avg Reward & Avg Tasks'); ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)

    # Oscillation line (lower better)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, avg_osc, marker='o', linewidth=2, label='Avg Oscillation')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha='right')
    ax2.set_title('Grid: Avg Oscillation (lower is better)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Grid comparison saved: {save_path}")

# ------------------------- Training Loop -------------------------
def train_maddpg(config, run_name, resume_from=None):
    """
    config: dict with keys:
      n_episodes, max_t, buffer_size, batch_size, gamma, lr_actor, lr_critic, tau,
      eps_start, eps_end, eps_decay, seed
    """
    set_global_seeds(config.get('seed', 42))
    env = CustomWarehouseEnv(render_mode=None)
    observations, info = env.reset(seed=config.get('seed', 42))

    state_size = int(observations[0].shape[0])
    action_size = env.action_space[0].n
    n_agents = env.n_agents

    agents = [
        MADDPGAgent(
            n_agents=n_agents,
            agent_id=i,
            state_size=state_size,
            action_size=action_size,
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            gamma=config['gamma'],
            tau=config['tau']
        )
        for i in range(n_agents)
    ]

    memory = MultiAgentReplayBuffer(
        n_agents=n_agents,
        state_size=state_size,
        action_size=action_size,
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        seed=config.get('seed', 42)
    )

    eps_start, eps_end, eps_decay = config['eps_start'], config['eps_end'], config['eps_decay']

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = f"results/logs/{run_name}_{timestamp}"
    plot_prefix = f"results/plots/{run_name}_{timestamp}"
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    log_file = f"{log_prefix}.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = (['Episode', 'Total_Reward', 'Tasks', 'Oscillation'] +
                  [f'Agent_{i}_Reward' for i in range(n_agents)] +
                  [f'ActorLoss_{i}' for i in range(n_agents)] +
                  [f'CriticLoss_{i}' for i in range(n_agents)])
        writer.writerow(header)

    scores_deque = deque(maxlen=100)
    scores_list, tasks_list, osc_list = [], [], []
    agent_scores_list = [[] for _ in range(n_agents)]
    actor_losses_list = [[] for _ in range(n_agents)]
    critic_losses_list = [[] for _ in range(n_agents)]

    best_avg100 = -np.inf
    best_ckpt_path = f"maddpg/checkpoints/{run_name}_best.pth"

    # Resume (optional)
    if resume_from and os.path.exists(resume_from):
        print(f"[RESUME] from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        for i, agent in enumerate(agents):
            agent.actor_local.load_state_dict(ckpt[f'agent_{i}_actor'])
            agent.critic_local.load_state_dict(ckpt[f'agent_{i}_critic'])
            agent.actor_target.load_state_dict(ckpt[f'agent_{i}_actor_target'])
            agent.critic_target.load_state_dict(ckpt[f'agent_{i}_critic_target'])
        best_avg100 = ckpt.get('best_score', -np.inf)

    print(f"[RUN] {run_name} | device={device} | agents={n_agents} | S={state_size} | A={action_size}")

    for ep in range(config['n_episodes']):
        observations, info = env.reset(seed=config.get('seed', 42))
        for a in agents: a.reset()

        epsilon = max(eps_end, eps_start * (eps_decay ** ep))

        ep_rewards = np.zeros(n_agents, dtype=np.float32)
        ep_actor_losses = [[] for _ in range(n_agents)]
        ep_critic_losses = [[] for _ in range(n_agents)]
        ep_actions = []
        ep_tasks = 0

        for t in range(config['max_t']):
            # act
            actions = []
            for i, agent in enumerate(agents):
                s_i = np.asarray(observations[i], dtype=np.float32).flatten()
                a = agent.act(s_i, epsilon=epsilon)  # your agent supports epsilon
                actions.append(a)

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = bool(terminated)

            # record actions
            ep_actions.append(actions)

            # track tasks if env exposes it
            ep_tasks = safe_get_info(info, 'completed_tasks', ep_tasks)

            # store transition
            memory.add(observations, actions, rewards, next_obs, terminated)

            # accumulate rewards
            ep_rewards += np.array(rewards, dtype=np.float32)

            # learn (sample one agent per step for stability/cost)
            if len(memory) >= config['batch_size']:
                batch = memory.sample()
                i = np.random.randint(n_agents)
                a_loss, c_loss = agents[i].learn(batch, agents)
                ep_actor_losses[i].append(a_loss)
                ep_critic_losses[i].append(c_loss)

            observations = next_obs
            if terminated or truncated:
                break

        total = float(np.sum(ep_rewards))
        scores_deque.append(total)
        scores_list.append(total)
        tasks_list.append(ep_tasks)
        osc_val = action_oscillation(ep_actions)
        osc_list.append(osc_val)

        for i in range(n_agents):
            agent_scores_list[i].append(ep_rewards[i])
            actor_losses_list[i].append(float(np.mean(ep_actor_losses[i])) if ep_actor_losses[i] else 0.0)
            critic_losses_list[i].append(float(np.mean(ep_critic_losses[i])) if ep_critic_losses[i] else 0.0)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [ep, total, ep_tasks, osc_val] + list(ep_rewards) + \
                  [actor_losses_list[i][-1] for i in range(n_agents)] + \
                  [critic_losses_list[i][-1] for i in range(n_agents)]
            writer.writerow(row)

        avg100 = float(np.mean(scores_deque))
        if avg100 > best_avg100:
            best_avg100 = avg100
            checkpoint = {'episode': ep, 'best_score': best_avg100}
            for i, agent in enumerate(agents):
                checkpoint[f'agent_{i}_actor'] = agent.actor_local.state_dict()
                checkpoint[f'agent_{i}_critic'] = agent.critic_local.state_dict()
                checkpoint[f'agent_{i}_actor_target'] = agent.actor_target.state_dict()
                checkpoint[f'agent_{i}_critic_target'] = agent.critic_target.state_dict()
            torch.save(checkpoint, best_ckpt_path)
            print(f"[CKPT] {run_name} ep {ep} new best Avg100={best_avg100:.2f} -> {best_ckpt_path}")

        if (ep % config.get('print_every', 25)) == 0:
            print(f"[{run_name}] Ep {ep:4d} | Avg100 {avg100:7.2f} | Total {total:7.2f} | "
                  f"Tasks {ep_tasks:3d} | Osc {osc_val:.3f} | eps={epsilon:.3f}")

    env.close()

    plot_path = f"{plot_prefix}.png"
    plot_training_results(scores_list, agent_scores_list, actor_losses_list, critic_losses_list,
                          osc_list, tasks_list, n_agents,
                          title=f"{run_name} (best Avg100={best_avg100:.2f})",
                          save_path=plot_path)

    # Return summary for grid selection
    return {
        'name': run_name,
        'episodes': len(scores_list),
        'avg_reward': float(np.mean(scores_list[-max(1, len(scores_list)//3):])),  # tail avg
        'avg_tasks':  float(np.mean(tasks_list[-max(1, len(tasks_list)//3):])),
        'avg_osc':    float(np.mean(osc_list[-max(1, len(osc_list)//3):])),
        'best_avg100': best_avg100,
        'best_ckpt_path': best_ckpt_path,
        'log_file': log_file,
        'plot_file': plot_path
    }

# ------------------------- Balanced Ranking ----------------------
def zscore(xs):
    xs = np.array(xs, dtype=np.float32)
    m, s = xs.mean(), xs.std()
    return (xs - m) / (s + 1e-8)

def rank_grid(rows, w_reward=0.6, w_tasks=0.3, w_osc=0.4):
    """
    Score = w_reward*z(Reward) + w_tasks*z(Tasks) - w_osc*z(Oscillation)
    Higher is better.
    """
    rew = np.array([r['avg_reward'] for r in rows], dtype=np.float32)
    tsk = np.array([r['avg_tasks']  for r in rows], dtype=np.float32)
    osc = np.array([r['avg_osc']    for r in rows], dtype=np.float32)

    score = w_reward*zscore(rew) + w_tasks*zscore(tsk) - w_osc*zscore(osc)
    for i, r in enumerate(rows):
        r['balanced_score'] = float(score[i])
    rows_sorted = sorted(rows, key=lambda d: d['balanced_score'], reverse=True)
    return rows_sorted

# ------------------------- Deterministic Eval --------------------
def evaluate_checkpoint(ckpt_path, episodes=5, max_t=300, seed=123):
    set_global_seeds(seed)
    env = CustomWarehouseEnv(render_mode=None)
    observations, info = env.reset(seed=seed)
    n_agents = env.n_agents
    state_size = int(observations[0].shape[0]); action_size = env.action_space[0].n

    agents = [
        MADDPGAgent(n_agents=n_agents, agent_id=i, state_size=state_size, action_size=action_size)
        for i in range(n_agents)
    ]
    ckpt = torch.load(ckpt_path, map_location=device)
    for i, agent in enumerate(agents):
        agent.actor_local.load_state_dict(ckpt[f'agent_{i}_actor'])
        agent.critic_local.load_state_dict(ckpt[f'agent_{i}_critic'])
        agent.actor_target.load_state_dict(ckpt[f'agent_{i}_actor_target'])
        agent.critic_target.load_state_dict(ckpt[f'agent_{i}_critic_target'])
        agent.reset()

    total_rewards, tasks_list, osc_list = [], [], []

    for ep in range(episodes):
        observations, info = env.reset(seed=seed + ep)
        ep_rewards = np.zeros(n_agents, dtype=np.float32)
        ep_actions = []
        ep_tasks = 0
        for t in range(max_t):
            actions = []
            for i, agent in enumerate(agents):
                s_i = np.asarray(observations[i], dtype=np.float32).flatten()
                a = agent.act(s_i, epsilon=0.0)  # deterministic (no exploration)
                actions.append(a)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            ep_actions.append(actions)
            ep_tasks = safe_get_info(info, 'completed_tasks', ep_tasks)
            ep_rewards += np.array(rewards, dtype=np.float32)
            observations = next_obs
            if terminated or truncated:
                break
        total_rewards.append(float(np.sum(ep_rewards)))
        tasks_list.append(ep_tasks)
        osc_list.append(action_oscillation(ep_actions))

    env.close()
    print(f"[EVAL] ckpt={ckpt_path}")
    print(f"  Avg Reward: {np.mean(total_rewards):.2f}  |  Avg Tasks: {np.mean(tasks_list):.2f}  |  Avg Osc: {np.mean(osc_list):.3f}")
    return {
        'avg_reward': float(np.mean(total_rewards)),
        'avg_tasks':  float(np.mean(tasks_list)),
        'avg_osc':    float(np.mean(osc_list)),
        'rewards': total_rewards, 'tasks': tasks_list, 'osc': osc_list
    }

# ------------------------- Grid Search ---------------------------
def main():
    # You can bump episodes to 1000–1500 for your final run.
    base = dict(
        n_episodes=1000,
        max_t=500,
        buffer_size=120_000,
        batch_size=256,
        gamma=0.99,
        lr_actor=1e-4,
        lr_critic=1e-3,
        tau=1e-2,
        eps_start=0.5,
        eps_end=0.1,
        eps_decay=0.995,
        seed=42,
        print_every=100
    )

    # 6 compact configs that generally work well for MADDPG on discrete wrappers.
    grid = [
        {**base, 'name': 'G1', 'lr_actor': 1e-4, 'lr_critic': 1e-3, 'gamma': 0.99, 'tau': 1e-2,  'batch_size': 256},
        {**base, 'name': 'G2', 'lr_actor': 3e-4, 'lr_critic': 1e-3, 'gamma': 0.98, 'tau': 5e-3, 'batch_size': 256},
        {**base, 'name': 'G3', 'lr_actor': 1e-4, 'lr_critic': 3e-4, 'gamma': 0.99, 'tau': 5e-3, 'batch_size': 256},
        {**base, 'name': 'G4', 'lr_actor': 3e-4, 'lr_critic': 3e-4, 'gamma': 0.95, 'tau': 1e-2,  'batch_size': 256},
        {**base, 'name': 'G5', 'lr_actor': 1e-4, 'lr_critic': 5e-4, 'gamma': 0.98, 'tau': 1e-2,  'batch_size': 128},
        {**base, 'name': 'G6', 'lr_actor': 2e-4, 'lr_critic': 8e-4, 'gamma': 0.99, 'tau': 7e-3, 'batch_size': 256},
    ]

    all_rows = []
    for cfg in grid:
        run_name = f"maddpg_{cfg['name']}_A{cfg['lr_actor']}_C{cfg['lr_critic']}_g{cfg['gamma']}_t{cfg['tau']}_bs{cfg['batch_size']}"
        summary = train_maddpg(cfg, run_name=run_name)
        all_rows.append({
            'name': summary['name'],
            'avg_reward': summary['avg_reward'],
            'avg_tasks': summary['avg_tasks'],
            'avg_osc': summary['avg_osc'],
            'best_avg100': summary['best_avg100'],
            'best_ckpt_path': summary['best_ckpt_path'],
            'episodes': summary['episodes'],
            'log_file': summary['log_file'],
            'plot_file': summary['plot_file'],
        })

    # Compare & rank
    grid_plot = "results/plots/grid_compare.png"
    plot_grid_comparison(all_rows, grid_plot)

    ranked = rank_grid(all_rows, w_reward=0.6, w_tasks=0.3, w_osc=0.4)
    print("\n==== GRID RANKING (Balanced) ====")
    for r in ranked:
        print(f"{r['name']}: score={r['balanced_score']:.3f} | avgR={r['avg_reward']:.2f} | "
              f"avgTasks={r['avg_tasks']:.2f} | avgOsc={r['avg_osc']:.3f} | bestAvg100={r['best_avg100']:.2f}")

    best = ranked[0]
    print(f"\n[SELECTED BEST] {best['name']}  ->  {best['best_ckpt_path']}")
    # Final deterministic eval (5 episodes)
    evaluate_checkpoint(best['best_ckpt_path'], episodes=5, max_t=base['max_t'])

if __name__ == "__main__":
    main()

