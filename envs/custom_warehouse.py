# File: environments/custom_warehouse.py
# Complete RL Environment with A* Pathfinding (optional hooks kept), Communication, Charging, Priority System,
# and TRUE movement (FORWARD uses direction; LEFT/RIGHT rotate).

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import heapq
from enum import IntEnum
from typing import List, Tuple, Dict, Optional


class Action(IntEnum):
    """Enhanced action space"""
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    PICKUP = 4
    DELIVERY = 5
    COMMUNICATE = 6
    REQUEST_PRIORITY = 7


class MessageType(IntEnum):
    """Encoded message types"""
    POSITION_SHARE = 0
    BATTERY_WARNING = 1
    TASK_COMPLETE = 2
    COLLISION_ALERT = 3
    PRIORITY_HELP = 4
    CHARGING_URGENT = 5


class Message:
    """Encoded message structure"""
    def __init__(self, sender_id: int, msg_type: int, data: Dict):
        self.sender_id = sender_id
        self.msg_type = msg_type
        self.data = data
        self.timestamp = None

    def encode(self) -> np.ndarray:
        """Encode message to numpy array"""
        encoded = np.zeros(6, dtype=np.float32)
        encoded[0] = self.sender_id
        encoded[1] = self.msg_type

        if self.msg_type == MessageType.POSITION_SHARE:
            encoded[2] = self.data.get('x', 0)
            encoded[3] = self.data.get('y', 0)
            encoded[4] = self.data.get('battery', 0)
        elif self.msg_type == MessageType.BATTERY_WARNING:
            encoded[2] = self.data.get('battery_level', 0)
            encoded[3] = self.data.get('x', 0)
            encoded[4] = self.data.get('y', 0)
        elif self.msg_type == MessageType.PRIORITY_HELP:
            encoded[2] = self.data.get('task_priority', 0)
            encoded[3] = self.data.get('target_x', 0)
            encoded[4] = self.data.get('target_y', 0)

        return encoded

    @staticmethod
    def decode(encoded: np.ndarray) -> 'Message':
        """Decode message from numpy array"""
        sender_id = int(encoded[0])
        msg_type = int(encoded[1])

        data = {}
        if msg_type == MessageType.POSITION_SHARE:
            data = {'x': encoded[2], 'y': encoded[3], 'battery': encoded[4]}
        elif msg_type == MessageType.BATTERY_WARNING:
            data = {'battery_level': encoded[2], 'x': encoded[3], 'y': encoded[4]}
        elif msg_type == MessageType.PRIORITY_HELP:
            data = {'task_priority': encoded[2], 'target_x': encoded[3], 'target_y': encoded[4]}

        return Message(sender_id, msg_type, data)


class AStarPathfinder:
    """A* pathfinding algorithm for warehouse robots (kept for future use)"""
    def __init__(self, layout: np.ndarray, grid_size: Tuple[int, int]):
        self.layout = layout
        self.grid_size = grid_size

    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(self, pos: Tuple[int, int],
                      carrying_task: bool = False) -> List[Tuple[int, int]]:
        """Get valid neighboring cells"""
        x, y = pos
        neighbors = []

        # 4-directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check bounds
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                cell = self.layout[nx][ny]
                # Walls are blocked
                if cell == 'W':
                    continue
                # If carrying task, can't enter shelves
                if carrying_task and cell == 'x':
                    continue
                neighbors.append((nx, ny))
        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  carrying_task: bool = False,
                  agents_positions: List[Tuple[int, int]] = None) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path using A*"""
        if agents_positions is None:
            agents_positions = []
        if start == goal:
            return [start]

        counter = 0
        open_set = [(0, counter, start, [start])]
        open_dict = {start: 0}
        closed_set = set()
        g_score = {start: 0}

        while open_set:
            _, _, current, path = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                return path

            neighbors = self.get_neighbors(current, carrying_task)
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + (1.5 if neighbor in agents_positions else 1.0)
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self.heuristic(neighbor, goal)
                if neighbor not in open_dict or f_score < open_dict[neighbor]:
                    counter += 1
                    open_dict[neighbor] = f_score
                    heapq.heappush(open_set, (f_score, counter, neighbor, path + [neighbor]))
        return None


class AgentType(IntEnum):
    """Heterogeneous robot types"""
    STANDARD = 0
    SCOUT = 1
    HEAVY = 2


class CustomWarehouseEnv(gym.Env):
    """
    Advanced Multi-Agent Warehouse Environment with:
    - TRUE movement via direction + FORWARD
    - Pickup/Delivery actions
    - Encoded message communication
    - Autonomous charging logic
    - Request-priority system
    - (A* hooks retained for future hybrid policies)
    """

    # Gymnasium metadata key is 'render_modes'
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'custom_warehouse_v3'
    }

    def __init__(
        self,
        n_agents: int = 4,
        grid_size: Tuple[int, int] = (15, 24),
        n_shelves: int = 8,
        sensor_range: int = 3,
        request_queue_size: int = 6,
        use_battery: bool = True,
        heterogeneous_agents: bool = True,
        enable_communication: bool = True,
        task_priorities: bool = True,
        render_mode: str = 'human'
    ):
        super().__init__()

        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_shelves = n_shelves
        self.sensor_range = sensor_range
        self.request_queue_size = request_queue_size
        self.use_battery = use_battery
        self.heterogeneous_agents = heterogeneous_agents
        self.enable_communication = enable_communication
        self.task_priorities = task_priorities
        self.render_mode = render_mode

        # Environment state
        self.timestep = 0
        self.step_count = 0
        self.max_steps = 1000

        # Layout + pathfinder
        self._create_custom_layout()
        self.pathfinder = AStarPathfinder(self.layout, self.grid_size)

        # Agents + tasks
        self._initialize_agents()
        self._initialize_tasks()

        # Spaces
        self._setup_spaces()

        # Visualization
        self._init_pygame()
        self.legend_scroll_offset = 0

    # ---------------- Layout & init ----------------

    def _create_custom_layout(self):
        """Create advanced warehouse layout with zones"""
        layout = """WWWWWWWWWWWWWWWWWWWWWWWW
WQQ........C.........QQW
WQQ..................QQW
W......................W
W...xxx...xx....xxx....W
W...xxx...xx....xxx....W
W......................W
W...xxx...xx....xxx....W
W...xxx...xx....xxx....W
W......................W
W...xxx...xx....xxx....W
W......................W
WQQ..................QQW
WQQ........C.........QQW
WWWWWWWWWWWWWWWWWWWWWWWW"""

        self.layout_str = layout
        self.layout = self._parse_layout(self.layout_str)

        actual_rows = self.layout.shape[0]
        actual_cols = self.layout.shape[1]
        self.grid_size = (actual_rows, actual_cols)

        self.charging_stations = []
        self.shelves = []
        self.corridors = []
        self.workstations = []

        for i, row in enumerate(self.layout):
            for j, cell in enumerate(row):
                if cell == 'C':
                    self.charging_stations.append((i, j))
                elif cell == 'Q':
                    self.workstations.append((i, j))
                elif cell == 'x':
                    self.shelves.append((i, j))
                elif cell == '.':
                    self.corridors.append((i, j))

        self.workstations = list(set(self.workstations))

    def _parse_layout(self, layout_str: str) -> np.ndarray:
        """Convert string layout to 2D numpy array"""
        lines = layout_str.strip().split('\n')
        lines = [list(line.strip()) for line in lines if line.strip()]
        if not lines:
            raise ValueError("Layout is empty")
        max_len = max(len(line) for line in lines)
        lines = [line + [' '] * (max_len - len(line)) for line in lines]
        return np.array(lines)

    def _initialize_agents(self):
        """Initialize heterogeneous agents"""
        self.agents = []

        if self.heterogeneous_agents:
            agent_types = [
                AgentType.SCOUT,
                AgentType.STANDARD,
                AgentType.HEAVY,
                AgentType.STANDARD,
            ][:self.n_agents]
        else:
            agent_types = [AgentType.STANDARD] * self.n_agents

        for i in range(self.n_agents):
            agent_type = agent_types[i]
            if agent_type == AgentType.SCOUT:
                speed = 2.0
                load_capacity = 0
                vision_range = 5
                battery_capacity = 120
            elif agent_type == AgentType.HEAVY:
                speed = 0.5
                load_capacity = 2
                vision_range = 3
                battery_capacity = 200
            else:
                speed = 1.0
                load_capacity = 1
                vision_range = 3
                battery_capacity = 150

            agent = {
                'id': i,
                'type': agent_type,
                'position': self._get_random_free_position(),
                'direction': np.random.randint(0, 4),  # 0:N,1:E,2:S,3:W
                'carrying': [],
                'speed': speed,
                'load_capacity': load_capacity,
                'vision_range': vision_range,
                'battery': battery_capacity,
                'battery_capacity': battery_capacity,
                'max_battery': battery_capacity,
                'messages_received': [],
                'message_inbox': [],
                'task_id': None,
                'tasks_completed': 0,
                'charging': False,
                'in_charging_station': False,
                'observed_agents': {},
                'low_battery_agents': set(),
                'priority_tasks_heard': [],
                'carrying_count': 0,
                'current_task': None,
                'current_path': None,   # kept for potential A* usage
                'path_index': 0,        # kept for potential A* usage
                'target_position': None,
                'path_update_frequency': 5,
                'last_dist_to_target': None
            }
            self.agents.append(agent)

    def _initialize_tasks(self):
        """Initialize task queue with priorities"""
        self.tasks = []
        self.completed_tasks = 0
        self.task_id_counter = 0
        for _ in range(self.request_queue_size):
            self._spawn_new_task()

    def _spawn_new_task(self):
        """Spawn new task from shelf to workstation"""
        if self.task_priorities:
            p = np.random.random()
            if p < 0.1:
                priority = 3; max_time = 100
            elif p < 0.8:
                priority = 2; max_time = 200
            else:
                priority = 1; max_time = 300
        else:
            priority = 2; max_time = 200

        if not self.shelves or not self.workstations:
            return

        pickup_loc = self.shelves[np.random.randint(len(self.shelves))]
        delivery_loc = self.workstations[np.random.randint(len(self.workstations))]
        task = {
            'id': self.task_id_counter,
            'pickup_loc': pickup_loc,
            'delivery_loc': delivery_loc,
            'priority': priority,
            'spawn_time': self.timestep,
            'max_time': max_time,
            'status': 'pending',
            'assigned_agent': None,
            'reserved_by': None,
            'reserved_until': 0,
        }
        self.tasks.append(task)
        self.task_id_counter += 1

    def _setup_spaces(self):
        """Define observation and action spaces"""
        self.action_space = spaces.Tuple([spaces.Discrete(len(Action)) for _ in range(self.n_agents)])

        obs_size = 210
        # Observations are mostly normalized floats; allow a safe float range.
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])

    def _get_random_free_position(self) -> Tuple[int, int]:
        """Get random free position in corridor"""
        return self.corridors[np.random.randint(len(self.corridors))]

    # ---------------- Core step/reset ----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = 0
        self.step_count = 0
        self._initialize_agents()
        self._initialize_tasks()
        observations = self._get_observations()
        infos = self._get_infos()
        return observations, infos

    def step(self, actions):
        """Execute one step with TRUE movement; A* hooks kept for future"""
        self.timestep += 1
        self.step_count += 1

        rewards = [0.0] * self.n_agents

        # Process received messages
        for agent_id in range(len(self.agents)):
            self._process_received_messages(agent_id)

        # Execute actions
        for i, action in enumerate(actions):
            reward = self._execute_action(i, int(action))
            rewards[i] = np.clip(reward, -5.0, 5.0)

        # Battery + charging
        if self.use_battery:
            self.update_batteries()
            self._update_charging_stations()

        # Charging behavior shaping
        for agent_id, agent in enumerate(self.agents):
            charger_distances = [
                abs(agent['position'][0] - c[0]) + abs(agent['position'][1] - c[1])
                for c in self.charging_stations
            ]
            min_charger_dist = min(charger_distances) if charger_distances else float('inf')
            rewards[agent_id] += self._compute_charging_reward(agent, min_charger_dist)

        # Exploration bonus near tasks
        for agent_id, agent in enumerate(self.agents):
            for task in self.tasks:
                if task['status'] == 'pending':
                    dist = abs(agent['position'][0] - task['pickup_loc'][0]) + \
                           abs(agent['position'][1] - task['pickup_loc'][1])
                    if dist <= 3:
                        rewards[agent_id] += 0.05
                    if dist <= 1:
                        rewards[agent_id] += 0.1

        # Update tasks
        self.update_tasks()

        # Observations
        observations = self._get_observations()

        # Termination
        done = self.timestep >= self.max_steps
        truncated = False
        infos = self._get_infos()
        return observations, rewards, done, truncated, infos

    # ---------------- Targeting / (optional) pathfinding helpers ----------------

    def _get_best_target(self, agent: Dict) -> Optional[Tuple[int, int]]:
        """Determine best target considering battery, deliveries, priorities"""

        battery_pct = agent['battery'] / agent['battery_capacity']

        # ✅ ONLY charge when battery < 10%
        if battery_pct < 0.10:
            return min(
                self.charging_stations,
                key=lambda c: abs(agent['position'][0] - c[0]) + abs(agent['position'][1] - c[1])
            )

        # ✅ If carrying, deliver immediately
        if agent['carrying_count'] > 0:
            return min(
                self.workstations,
                key=lambda ws: abs(agent['position'][0] - ws[0]) + abs(agent['position'][1] - ws[1])
            )

        # ✅ Go to nearest high-priority pending task
        pending_tasks = [t for t in self.tasks if t['status'] == 'pending']
        if pending_tasks:
            return max(
                pending_tasks,
                key=lambda t: (t['priority'] /
                            (1 + abs(agent['position'][0] - t['pickup_loc'][0]) +
                                    abs(agent['position'][1] - t['pickup_loc'][1])))
            )['pickup_loc']

        # ✅ No other charging, no pre-emptive charging
        return None

    def _compute_path_to_target(self, agent: Dict, target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Compute optimal path to target using A* (available if you want hybrid control)"""
        other_positions = [a['position'] for a in self.agents if a['id'] != agent['id']]
        carrying_task = agent['carrying_count'] > 0

        path = self.pathfinder.find_path(
            start=tuple(agent['position']),
            goal=target,
            carrying_task=carrying_task,
            agents_positions=other_positions
        )
        if path:
            agent['current_path'] = path
            agent['path_index'] = 0
            agent['target_position'] = target
            return path
        return None

    def _follow_path(self, agent: Dict) -> bool:
        """Follow the computed path - returns True if reached target"""
        if agent['current_path'] is None or len(agent['current_path']) == 0:
            return False
        if agent['path_index'] < len(agent['current_path']) - 1:
            next_pos = agent['current_path'][agent['path_index'] + 1]
            if self._is_valid_move(agent, next_pos):
                agent['position'] = next_pos
                agent['path_index'] += 1
                agent['battery'] -= 0.05
                return False
        return True  # reached destination

    # ---------------- Action logic (TRUE movement) ----------------

    def _execute_action(self, agent_id: int, action: int) -> float:
        """Execute action and return reward"""
        agent = self.agents[agent_id]
        reward = 0.0

        # Gradual low-battery penalty if not addressing it
        battery_pct = agent['battery'] / agent['battery_capacity']
        if battery_pct < 0.10 and action not in [Action.NOOP, Action.COMMUNICATE, Action.REQUEST_PRIORITY]:
            if agent['position'] not in self.charging_stations:
                reward -= 0.1

        if action == Action.NOOP:
            agent['battery'] -= 0.01
            battery_pct = agent['battery'] / agent['battery_capacity']
            if battery_pct > 0.3 and agent['carrying_count'] == 0:
                if any(t['status'] == 'pending' for t in self.tasks):
                    reward -= 0.01

        # Bonus while carrying
        if agent['carrying_count'] > 0:
            reward += 0.02

        elif action == Action.FORWARD:
            # TRUE GRID MOVE: move one cell in facing direction if valid
            new_pos = self._get_front_position(agent)
            if self._is_valid_move(agent, new_pos):
                agent['position'] = new_pos
                agent['battery'] -= 0.0005
                reward += 0.01  # small exploration reward

                # Shaping: if moving closer to best target
                best_target = self._get_best_target(agent)
                if best_target:
                    current_dist = abs(agent['position'][0] - best_target[0]) + abs(agent['position'][1] - best_target[1])
                    last_dist = agent.get('last_dist_to_target')
                    if last_dist is not None:
                        if current_dist < last_dist:
                            reward += 0.02
                        elif current_dist > last_dist:
                            reward -= 0.005
                    agent['last_dist_to_target'] = current_dist
            else:
                reward -= 0.02
                agent['battery'] -= 0.0005

        elif action == Action.LEFT:
            agent['direction'] = (agent['direction'] - 1) % 4
            agent['battery'] -= 0.0001
            # tiny reward if trying to escape being stuck
            if agent.get('stuck_counter', 0) > 2:
                reward += 0.01

        elif action == Action.RIGHT:
            agent['direction'] = (agent['direction'] + 1) % 4
            agent['battery'] -= 0.0001
            if agent.get('stuck_counter', 0) > 2:
                reward += 0.01

        elif action == Action.PICKUP:
            reward += self._execute_pickup(agent)
            agent['battery'] -= 0.005

        elif action == Action.DELIVERY:
            reward += self._execute_delivery(agent)
            agent['battery'] -= 0.005

        elif action == Action.COMMUNICATE:
            if self.enable_communication:
                self._broadcast_message(agent_id)
                agent['battery'] -= 0.01

        elif action == Action.REQUEST_PRIORITY:
            reward += self._execute_request_priority(agent_id)
            agent['battery'] -= 0.05

        agent['battery'] = max(0, agent['battery'])

        # Stuck detection / shaping
        if 'last_position' not in agent:
            agent['last_position'] = agent['position']
            agent['stuck_counter'] = 0
        else:
            if agent['position'] == agent['last_position']:
                agent['stuck_counter'] += 1
                reward -= 0.02
            else:
                if agent['stuck_counter'] > 5:
                    reward += 0.5  # escaped long stuck
                agent['stuck_counter'] = 0
            agent['last_position'] = agent['position']

            # Reward getting closer to best target in general step
            best_target = self._get_best_target(agent)
            if best_target:
                current_dist = abs(agent['position'][0] - best_target[0]) + abs(agent['position'][1] - best_target[1])
                if agent.get('last_dist_to_target') is not None:
                    if current_dist < agent['last_dist_to_target']:
                        reward += 0.02
                    elif current_dist > agent['last_dist_to_target']:
                        reward -= 0.01
                agent['last_dist_to_target'] = current_dist
            else:
                agent['last_dist_to_target'] = None

        return reward

    # ---------------- Pickup / Delivery / Priority ----------------

    def _execute_pickup(self, agent: Dict) -> float:
        """Execute PICKUP action"""
        reward = 0.0
        agent_pos = tuple(agent['position'])

        # Must be at shelf
        at_shelf = any(agent_pos == shelf for shelf in self.shelves)
        if not at_shelf:
            reward -= 0.1
            agent['battery'] -= 0.08
            return reward

        # Capacity
        if agent['carrying_count'] >= agent['load_capacity']:
            reward -= 0.1
            agent['battery'] -= 0.08
            return reward

        # Find pending task at this shelf
        pickup_task = None
        for task in self.tasks:
            if task['status'] == 'pending' and task['pickup_loc'] == agent_pos:
                pickup_task = task
                break

        if pickup_task:
            pickup_task['status'] = 'picked_up'
            pickup_task['agent_id'] = agent['id']
            agent['carrying_count'] += 1
            agent['current_task'] = pickup_task
            agent['task_id'] = pickup_task['id']
            agent['carrying'].append(pickup_task['id'])

            priority_multiplier = {1: 1.0, 2: 1.5, 3: 2.0}
            reward = 40.0 * priority_multiplier[pickup_task['priority']]
            agent['battery'] -= 0.08
        else:
            reward -= 0.1
            agent['battery'] -= 0.08

        return reward

    def _execute_delivery(self, agent: Dict) -> float:
        """Execute DELIVERY action"""
        reward = 0.0
        agent_pos = tuple(agent['position'])

        # Must be at workstation
        at_ws = any(agent_pos == ws for ws in self.workstations)
        if not at_ws:
            reward -= 0.1
            agent['battery'] -= 0.08
            return reward

        # Must be carrying
        if agent['carrying_count'] == 0 or agent['current_task'] is None:
            reward -= 0.1
            agent['battery'] -= 0.08
            return reward

        task = agent['current_task']
        if task['delivery_loc'] == agent_pos:
            task['status'] = 'completed'
            task['completion_time'] = self.timestep

            priority_multiplier = {1: 1.0, 2: 1.5, 3: 2.0}
            reward += 80.0 * priority_multiplier[task['priority']]

            time_taken = self.timestep - task['spawn_time']
            if time_taken < task['max_time'] * 0.5:
                reward += 20.0  # fast delivery bonus

            agent['carrying_count'] -= 1
            agent['current_task'] = None
            agent['task_id'] = None
            agent['tasks_completed'] += 1
            if task['id'] in agent['carrying']:
                agent['carrying'].remove(task['id'])
            agent['battery'] -= 0.08

            self.completed_tasks += 1
            self._spawn_new_task()
        else:
            reward -= 0.1
            agent['battery'] -= 0.08

        return reward

    def _execute_request_priority(self, agent_id: int) -> float:
        """Execute REQUEST_PRIORITY action - broadcast high priority task"""
        agent = self.agents[agent_id]
        reward = 0.0

        pending_tasks = [t for t in self.tasks if t['status'] == 'pending']
        if not pending_tasks:
            reward -= 0.2
            return reward

        best_task = self._find_highest_priority_task(agent, pending_tasks)
        if best_task and best_task['priority'] >= 2:
            priority_msg = Message(
                sender_id=agent_id,
                msg_type=MessageType.PRIORITY_HELP,
                data={
                    'task_priority': best_task['priority'],
                    'target_x': best_task['pickup_loc'][0],
                    'target_y': best_task['pickup_loc'][1]
                }
            )
            self._broadcast_message(agent_id, priority_msg)
            reward += 0.1
        else:
            reward -= 0.2
        return reward

    def _find_highest_priority_task(self, agent: Dict, pending_tasks: List) -> Optional[Dict]:
        """Find best task considering priority and distance"""
        available_tasks = [t for t in pending_tasks if agent['carrying_count'] < agent['load_capacity']]
        if not available_tasks:
            return None

        best_task = None
        best_score = -float('inf')
        agent_pos = np.array(agent['position'])

        for task in available_tasks:
            task_pos = np.array(task['pickup_loc'])
            distance = np.linalg.norm(agent_pos - task_pos)
            distance_score = 1.0 / (1.0 + distance)
            priority_score = task['priority'] / 3.0
            score = (0.6 * priority_score) + (0.4 * distance_score)
            if score > best_score:
                best_score = score
                best_task = task
        return best_task

    # ---------------- Communication ----------------

    def _broadcast_message(self, agent_id: int, message: Optional[Message] = None):
        """Broadcast encoded message to nearby agents"""
        sender = self.agents[agent_id]
        sender_pos = np.array(sender['position'])

        if message is None:
            message = Message(
                sender_id=agent_id,
                msg_type=MessageType.POSITION_SHARE,
                data={'x': sender['position'][0], 'y': sender['position'][1], 'battery': sender['battery']}
            )

        encoded_msg = message.encode()
        comm_range = 5

        for recv_id, receiver in enumerate(self.agents):
            if recv_id == agent_id:
                continue
            receiver_pos = np.array(receiver['position'])
            distance = np.linalg.norm(sender_pos - receiver_pos)
            if distance <= comm_range:
                receiver['message_inbox'].append({
                    'encoded': encoded_msg,
                    'distance': distance,
                    'timestamp': self.timestep
                })

    def _process_received_messages(self, agent_id: int):
        """Process and decode received messages"""
        agent = self.agents[agent_id]
        if not agent['message_inbox']:
            return

        messages_to_process = agent['message_inbox'][-5:]
        for msg_data in messages_to_process:
            decoded_msg = Message.decode(msg_data['encoded'])
            if decoded_msg.msg_type == MessageType.POSITION_SHARE:
                agent['observed_agents'][decoded_msg.sender_id] = decoded_msg.data
            elif decoded_msg.msg_type == MessageType.BATTERY_WARNING:
                if decoded_msg.data['battery_level'] < 30:
                    agent['low_battery_agents'].add(decoded_msg.sender_id)
            elif decoded_msg.msg_type == MessageType.PRIORITY_HELP:
                agent['priority_tasks_heard'].append({
                    'priority': decoded_msg.data['task_priority'],
                    'location': (decoded_msg.data['target_x'], decoded_msg.data['target_y'])
                })

        agent['message_inbox'] = []

    # ---------------- Rewards / Observations / Infos ----------------

    def _compute_charging_reward(self, agent, distance_to_charger):
        reward = 0.0

        # ✅ ONLY reward charging if battery < 10%
        if agent['battery'] < agent['battery_capacity'] * 0.10:
            if agent['position'] in self.charging_stations:
                reward += 0.3       # good, charging at low battery
            else:
                reward -= 0.3       # bad, not moving to charger
        else:
            # ✅ Penalize being near or inside charger unnecessarily
            if agent['position'] in self.charging_stations:
                reward -= 0.5
            if distance_to_charger <= 3:
                reward -= 0.1

        # ✅ Penalize overcharging
        if agent['battery'] > agent['battery_capacity'] * 0.95 and agent['charging']:
            reward -= 0.2

        return reward


    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents"""
        return tuple(self._get_agent_observation(agent) for agent in self.agents)

    def _get_agent_observation(self, agent: Dict) -> np.ndarray:
        """Get observation for single agent with path info (kept light)"""
        obs_parts = []

        # Self state (normalized where sensible)
        obs_parts.extend([
            agent['position'][0] / self.grid_size[0],
            agent['position'][1] / self.grid_size[1],
            agent['direction'] / 4.0,
            agent['carrying_count'] / max(1, agent['load_capacity']),
            agent['battery'] / agent['battery_capacity'],
        ])

        # Path info (kept for compatibility with earlier policies)
        path_length = len(agent['current_path']) if agent['current_path'] else 0
        obs_parts.extend([
            min(path_length / 50.0, 1.0),
            1.0 if agent['current_path'] is not None else 0.0,
        ])

        # Local grid observation placeholder (could be extended)
        grid_obs = self._get_grid_observation(agent)
        obs_parts.extend(grid_obs.flatten())

        # Nearest task
        obs_parts.extend(self._get_nearest_task_observation(agent))

        # Nearby agents placeholder
        obs_parts.extend(self._get_nearby_agents_observation(agent))

        # Charging station proximity
        charger_distances = [
            abs(agent['position'][0] - c[0]) + abs(agent['position'][1] - c[1])
            for c in self.charging_stations
        ]
        min_charger_dist = min(charger_distances) if charger_distances else float('inf')
        obs_parts.append(min(min_charger_dist / 20.0, 1.0))

        # Pad to size 210
        while len(obs_parts) < 210:
            obs_parts.append(0.0)

        return np.array(obs_parts[:210], dtype=np.float32)

    def _get_grid_observation(self, agent: Dict) -> np.ndarray:
        """Get local grid observation around agent (placeholder zeros)"""
        vision = agent['vision_range']
        grid_size = 2 * vision + 1
        obs = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
        return obs

    def _get_nearest_task_observation(self, agent: Dict) -> List[float]:
        """Get info about nearest pending task"""
        pending_tasks = [t for t in self.tasks if t['status'] == 'pending']
        if not pending_tasks:
            return [0, 0, 0, 0]

        agent_pos = np.array(agent['position'])
        nearest_task = min(pending_tasks, key=lambda t: np.linalg.norm(agent_pos - np.array(t['pickup_loc'])))

        rel_x = (nearest_task['pickup_loc'][0] - agent['position'][0]) / self.grid_size[0]
        rel_y = (nearest_task['pickup_loc'][1] - agent['position'][1]) / self.grid_size[1]
        priority = nearest_task['priority'] / 3.0
        time_remaining = max(0, nearest_task['max_time'] - (self.timestep - nearest_task['spawn_time'])) / nearest_task['max_time']
        return [rel_x, rel_y, priority, time_remaining]

    def _get_nearby_agents_observation(self, agent: Dict) -> List[float]:
        """Get info about nearby agents (placeholder zeros, length 15)"""
        return [0.0] * 15

    def _get_infos(self) -> Dict:
        """Get info dictionary"""
        return {
            'timestep': self.timestep,
            'completed_tasks': self.completed_tasks,
            'pending_tasks': len([t for t in self.tasks if t['status'] == 'pending']),
            'agents_battery': [a['battery'] for a in self.agents],
            'agent_positions': [tuple(a['position']) for a in self.agents],
        }

    # ---------------- Movement helpers ----------------

    def _get_front_position(self, agent: Dict) -> Tuple[int, int]:
        """Get position in front of agent (0:N, 1:E, 2:S, 3:W)"""
        x, y = agent['position']
        d = agent['direction']
        if d == 0:      # North
            return (x - 1, y)
        elif d == 1:    # East
            return (x, y + 1)
        elif d == 2:    # South
            return (x + 1, y)
        else:           # West
            return (x, y - 1)

    def _is_valid_move(self, agent: Dict, new_pos: Tuple[int, int]) -> bool:
        """Check if move is valid"""
        x, y = new_pos
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        cell = self.layout[x][y]
        if cell == 'W':
            return False
        if agent['carrying_count'] > 0 and cell == 'x':
            return False
        for other in self.agents:
            if other['id'] != agent['id'] and tuple(other['position']) == new_pos:
                return False
        return True

    # ---------------- Batteries / Tasks ----------------

    def update_batteries(self):
        """Update battery levels"""
        for agent in self.agents:
            battery_pct = agent['battery'] / agent['battery_capacity']
            if tuple(agent['position']) in self.charging_stations and battery_pct < 0.95:
                charge_amount = min(5.0, agent['battery_capacity'] - agent['battery'])
                agent['battery'] = min(agent['battery'] + charge_amount, agent['battery_capacity'])
            else:
                # Tiny idle drain
                agent['battery'] = max(0, agent['battery'] - 0.0001)

    def _update_charging_stations(self):
        """Update charging status"""
        for agent in self.agents:
            agent_pos = tuple(agent['position'])
            if agent_pos in self.charging_stations and agent['battery'] < 100:
                agent['in_charging_station'] = True
                agent['charging'] = True
            else:
                agent['in_charging_station'] = False
                if agent['battery'] >= agent['battery_capacity'] * 0.95:
                    agent['charging'] = False

    def update_tasks(self):
        """Update task queue"""
        self.tasks = [t for t in self.tasks if t['status'] != 'completed']
        pending_count = len([t for t in self.tasks if t['status'] == 'pending'])
        while pending_count < self.request_queue_size:
            self._spawn_new_task()
            pending_count = len([t for t in self.tasks if t['status'] == 'pending'])

    # ---------------- Rendering ----------------

    def _init_pygame(self):
        """Initialize Pygame for 2D rendering"""
        self.screen = None
        self.clock = None
        try:
            pygame.init()
            self.cell_size = 35
            self.grid_width = self.grid_size[1] * self.cell_size
            self.grid_height = self.grid_size[0] * self.cell_size

            self.legend_width = 320
            self.info_panel_height = 100

            self.screen_width = self.grid_width + self.legend_width
            self.screen_height = self.grid_height + self.info_panel_height

            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Advanced Warehouse MARL - Priority Task + Communication + TRUE Movement")
            self.clock = pygame.time.Clock()
        except Exception as e:
            print(f"⚠️  Pygame initialization failed - rendering disabled ({e})")
            self.screen = None

    def _render_pygame(self):
        """Render with Pygame and draw paths"""
        if self.screen is None:
            return

        # Basic event pump to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return

        self.screen.fill((25, 28, 32))

        # Colors
        WALL_COLOR = (52, 58, 64)
        CORRIDOR_COLOR = (248, 249, 250)
        SHELF_MAIN = (139, 87, 42)
        SHELF_HIGHLIGHT = (184, 134, 11)
        CHARGING_MAIN = (255, 179, 0)
        CHARGING_GLOW = (255, 235, 59)
        WORKSTATION_COLOR = (100, 100, 150)

        SCOUT_COLOR = (46, 204, 113)
        STANDARD_COLOR = (52, 152, 219)
        HEAVY_COLOR = (231, 76, 60)

        # Grid
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                cell = self.layout[i][j]
                x = j * self.cell_size
                y = i * self.cell_size
                cs = self.cell_size

                if cell == 'W':
                    pygame.draw.rect(self.screen, WALL_COLOR, (x+2, y+2, cs-4, cs-4))
                elif cell == 'C':
                    pygame.draw.rect(self.screen, CHARGING_MAIN, (x+5, y+5, cs-10, cs-10))
                    pygame.draw.rect(self.screen, CHARGING_GLOW, (x+5, y+5, cs-10, cs-10), 2)
                elif cell == 'Q':
                    pygame.draw.rect(self.screen, WORKSTATION_COLOR, (x+3, y+3, cs-6, cs-6))
                    pygame.draw.rect(self.screen, (200, 200, 255), (x+3, y+3, cs-6, cs-6), 2)
                elif cell == 'x':
                    pygame.draw.rect(self.screen, SHELF_MAIN, (x+5, y+5, cs-10, cs-10))
                    pygame.draw.rect(self.screen, SHELF_HIGHLIGHT, (x+8, y+2, cs-16, 3))
                else:
                    pygame.draw.rect(self.screen, CORRIDOR_COLOR, (x, y, cs, cs))

                pygame.draw.rect(self.screen, (210, 213, 218), (x, y, cs, cs), 1)

        # Tasks
        for task in self.tasks:
            if task['status'] in ['pending', 'picked_up']:
                if task['priority'] == 3:
                    task_color = (244, 67, 54)
                elif task['priority'] == 2:
                    task_color = (255, 152, 0)
                else:
                    task_color = (255, 235, 59)

                is_carried = any(bot.get('task_id') == task['id'] for bot in self.agents)
                if not is_carried:
                    pickup_pos = task['pickup_loc']
                    cx = pickup_pos[1] * self.cell_size + self.cell_size // 2
                    cy = pickup_pos[0] * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(self.screen, task_color, (cx, cy), 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 6, 2)
                    id_font = pygame.font.Font(None, 14)
                    id_text = id_font.render(str(task['id']), True, (0, 0, 0))
                    self.screen.blit(id_text, (cx - 3, cy - 3))

        # Agents
        for agent in self.agents:
            pos = agent['position']
            cx = pos[1] * self.cell_size + self.cell_size // 2
            cy = pos[0] * self.cell_size + self.cell_size // 2

            if agent['type'] == AgentType.SCOUT:
                color = SCOUT_COLOR
                pygame.draw.polygon(self.screen, color,
                                    [(cx, cy-8), (cx-8, cy+8), (cx+8, cy+8)])
            elif agent['type'] == AgentType.HEAVY:
                color = HEAVY_COLOR
                pygame.draw.circle(self.screen, color, (cx, cy), 10)
            else:
                color = STANDARD_COLOR
                pygame.draw.circle(self.screen, color, (cx, cy), 8)

            # Battery bar
            battery_pct = agent['battery'] / agent['battery_capacity']
            if battery_pct < 0.25:
                bat_color = (244, 67, 54)
            elif battery_pct < 0.6:
                bat_color = (255, 152, 0)
            else:
                bat_color = (76, 175, 80)

            bar_width, bar_height = 20, 4
            bar_x = cx - bar_width // 2
            bar_y = cy - 20
            pygame.draw.rect(self.screen, (60, 60, 65), (bar_x, bar_y, bar_width, bar_height))
            fill_width = int((bar_width - 2) * min(max(battery_pct, 0.0), 1.0))
            pygame.draw.rect(self.screen, bat_color, (bar_x + 1, bar_y + 1, fill_width, bar_height - 2))

        # Info panel
        info_y = self.grid_height
        pygame.draw.rect(self.screen, (33, 37, 41), (0, info_y, self.screen_width, self.info_panel_height))
        font_info = pygame.font.Font(None, 24)
        info_text = f"Step: {self.timestep} | Completed: {self.completed_tasks} | Pending: {len([t for t in self.tasks if t['status'] == 'pending'])} | Move: TRUE"
        text_surf = font_info.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (20, info_y + 20))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(10)

    def render(self):
        """Render the environment"""
        self._render_pygame()
        return None

    # ---------------- Close ----------------

    def close(self):
        """Clean up"""
        if self.screen is not None:
            try:
                pygame.quit()
            except Exception:
                pass


# Quick manual test (optional)
if __name__ == "__main__":
    print("=" * 80)
    print("🏭 Advanced Warehouse MARL Environment v3.0 (TRUE movement)")
    print("=" * 80)
    env = CustomWarehouseEnv(
        n_agents=4,
        heterogeneous_agents=True,
        enable_communication=True,
        task_priorities=True
    )
    print("\n🚀 Environment initialized successfully!")
    print(f"📍 Grid size: {env.grid_size}")
    print(f"🤖 Agents: {env.n_agents}")
    print(f"📦 Shelves: {len(env.shelves)}")
    print(f"🏪 Workstations: {len(env.workstations)}")
    print(f"🔌 Charging stations: {len(env.charging_stations)}")

    obs, info = env.reset()
    print("\n▶️  Running 10 random-action steps...")
    for step in range(10):
        actions = [env.action_space.spaces[i].sample() for i in range(env.n_agents)]
        obs, rewards, done, truncated, info = env.step(actions)
        print(f"Step {step+1}: Rewards={[f'{r:.2f}' for r in rewards]}, Completed={info['completed_tasks']}, Pending={info['pending_tasks']}")
        if step % 3 == 0:
            env.render()
    env.close()
    print("\n✅ Test completed successfully!")

