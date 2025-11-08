from envs.custom_warehouse import CustomWarehouseEnv

env = CustomWarehouseEnv()
obs = env.reset()

print("Environment initialized successfully!")
print(f"Grid size: {env.grid_size}")
print(f"Number of agents: {env.n_agents}")
print(f"Number of shelves: {len(env.shelves)}")
print(f"Number of workstations: {len(env.workstations)}")
print(f"Number of charging stations: {len(env.charging_stations)}")

print(f"Observation per agent: {obs}")
print(f"Observation shape/type per agent: {[o.shape if hasattr(o, 'shape') else type(o) for o in obs]}")

print("Running 10 test environment steps...")
for step in range(10):
    actions = [env.action_space.sample() for _ in range(env.n_agents)]
    obs, rewards, dones, truncated, info = env.step(actions)
    print(f"Step {step+1}, Rewards: {rewards}, Completed tasks: {info.get('completedtasks','N/A')}, Pending tasks: {info.get('pendingtasks','N/A')}")
    if step % 3 == 0:
        env.render()

env.close()
print("Test completed successfully!")
