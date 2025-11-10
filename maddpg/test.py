# File: maddpg/test_trained.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from envs.custom_warehouse import CustomWarehouseEnv
from maddpg.agent import MADDPGAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_agents(checkpoint_path, n_agents, state_size, action_size):
    """Load MADDPG agents from saved checkpoint"""

    print("\n✅ Loading checkpoint:", checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device)

    print("\n📌 Keys inside checkpoint:")
    for k in ckpt.keys():
        print("  ", k)

    agents = []
    for i in range(n_agents):
        agents.append(
            MADDPGAgent(
                n_agents=n_agents,
                agent_id=i,
                state_size=state_size,
                action_size=action_size,
            )
        )

    # ✅ Correct key format
    for i, agent in enumerate(agents):
        print(f"\nLoading agent {i}:")
        agent.actor_local.load_state_dict(ckpt[f"agent_{i}_actor"])
        agent.actor_target.load_state_dict(ckpt[f"agent_{i}_actor_target"])
        agent.critic_local.load_state_dict(ckpt[f"agent_{i}_critic"])
        agent.critic_target.load_state_dict(ckpt[f"agent_{i}_critic_target"])

    print("\n✅ All agents loaded successfully.\n")
    return agents


def test_trained_model(checkpoint_path, episodes=2, max_steps=300):

    print("\n====================================")
    print(" ✅ TESTING TRAINED MADDPG AGENTS")
    print("====================================")

    # ✅ Render mode "human" for GUI
    env = CustomWarehouseEnv(render_mode="human")
    obs, info = env.reset()

    n_agents = env.n_agents
    state_size = obs[0].shape[0]
    action_size = env.action_space[0].n

    print(f"\nAgents: {n_agents}, State size: {state_size}, Action size: {action_size}")

    agents = load_agents(checkpoint_path, n_agents, state_size, action_size)

    for ep in range(episodes):

        print(f"\n🏁 Episode {ep + 1}/{episodes}")

        obs, info = env.reset()
        total_rewards = np.zeros(n_agents)

        for step in range(max_steps):

            actions = []

            for i, agent in enumerate(agents):
                state_i = np.array(obs[i], dtype=np.float32)
                # ✅ Small epsilon to avoid NOOP-only behavior
                action = agent.act(state_i, epsilon=0.15)
                actions.append(action)

            obs, rewards, done, truncated, info = env.step(actions)

            total_rewards += np.array(rewards)

            # ✅ Render movement
            env.render()

            print(
                f"Step {step:3d} | Actions={actions} | "
                f"Rewards={[round(r,2) for r in rewards]}"
            )

            if done:
                print("Episode finished due to max steps.")
                break

        print(f"🎯 Episode {ep + 1} Total Rewards: {total_rewards}")

    env.close()
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    test_trained_model(
        checkpoint_path="maddpg/checkpoints/full_150ep.pth",
        episodes=2,
        max_steps=150,
    )
