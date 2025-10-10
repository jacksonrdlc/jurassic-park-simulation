"""
Gymnasium Environment Wrapper for Jurassic Park Simulation
Enables PPO training with Stable-Baselines3
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from my_first_island import (IslandModel, GrassAgent, HerbivoreAgent, CarnivoreAgent,
                              Triceratops, Gallimimus, TRex, Velociraptor)


class JurassicParkHerbivoreEnv(gym.Env):
    """Gymnasium environment for training herbivore agents with PPO"""

    metadata = {'render_modes': []}

    def __init__(self, width=50, height=50, max_steps=200):
        super(JurassicParkHerbivoreEnv, self).__init__()

        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Observation space: same as Q-learning (19 features)
        # Own energy, 8 directions (grass distance), 8 directions (predator distance), temperature, rainfall
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(19,), dtype=np.float32
        )

        # Action space: 9 discrete actions (8 directions + stay)
        self.action_space = spaces.Discrete(9)

        self.model = None
        self.agent = None
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        # Create new island model
        self.model = IslandModel(
            width=self.width,
            height=self.height,
            num_herbivores=1,  # Single agent for training
            num_carnivores=5,  # Predators for challenge
            temperature=25,
            rainfall=100,
            use_learning_agents=False  # Use traditional carnivores
        )

        # Replace the herbivore with a PPO-controlled agent
        # Remove default herbivore
        herbivores = [a for a in self.model.agents if isinstance(a, HerbivoreAgent)]
        if herbivores:
            for h in herbivores:
                self.model.grid.remove_agent(h)
                h.remove()

        # Add our controlled agent
        self.agent = Triceratops(self.model)
        x = self.model.random.randrange(self.width)
        y = self.model.random.randrange(self.height)
        self.model.grid.place_agent(self.agent, (x, y))

        self.steps_taken = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        # Convert action to movement
        dx, dy = self._action_to_movement(action)

        # Store previous state for reward calculation
        prev_energy = self.agent.energy
        prev_pos = self.agent.pos

        # Execute movement
        for _ in range(self.agent.speed):
            new_x = (self.agent.pos[0] + dx) % self.model.grid.width
            new_y = (self.agent.pos[1] + dy) % self.model.grid.height
            self.model.grid.move_agent(self.agent, (new_x, new_y))

        # Update direction
        if (new_x, new_y) != prev_pos:
            self.agent.direction = (dx, dy)

        # Eat grass
        cell_contents = self.model.grid.get_cell_list_contents([self.agent.pos])
        grass = [obj for obj in cell_contents if isinstance(obj, GrassAgent)]
        ate_grass = False
        if grass and grass[0].energy > 0:
            energy_gained = grass[0].energy
            self.agent.energy += energy_gained
            grass[0].energy = 0
            grass[0].countdown = grass[0].regrowth_time
            ate_grass = True

        # Lose energy
        energy_loss = 1 * self.model.metabolism_multiplier * self.agent.size_metabolism
        self.agent.energy -= energy_loss

        # Calculate reward
        reward = 0.0
        if ate_grass:
            reward += 10.0
        reward += 0.1  # Survival bonus
        reward -= 0.05  # Energy cost penalty

        # Check if agent died or was hunted
        terminated = False
        if self.agent.energy <= 0:
            reward -= 50.0
            terminated = True
        elif self.agent.pos is None:  # Agent was hunted
            reward -= 50.0
            terminated = True

        # Step the rest of the simulation (carnivores, grass)
        if not terminated:
            carnivores = [a for a in self.model.agents if isinstance(a, CarnivoreAgent)]
            for carnivore in carnivores:
                carnivore.step()

            # Check if agent was hunted during carnivore step
            if self.agent.pos is None or self.agent.energy <= 0:
                reward -= 50.0
                terminated = True

            # Grass regrowth
            grass_agents = [a for a in self.model.agents if isinstance(a, GrassAgent)]
            for g in grass_agents:
                g.step()

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps

        # Get next observation
        if terminated:
            observation = np.zeros(19, dtype=np.float32)
        else:
            observation = self._get_observation()

        info = {
            'steps_alive': self.steps_taken,
            'energy': self.agent.energy if not terminated else 0,
            'ate_grass': ate_grass
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation vector for the agent"""
        obs = []

        # Own energy (normalized)
        obs.append(self.agent.energy / 100.0)

        # Scan 8 directions
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Look for grass
        for dx, dy in directions:
            grass_distance = 1.0
            for dist in range(1, 6):
                check_x = (self.agent.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.agent.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                grass = [obj for obj in cell_contents if isinstance(obj, GrassAgent) and obj.energy > 0]
                if grass:
                    grass_distance = dist / 5.0
                    break
            obs.append(grass_distance)

        # Look for predators
        for dx, dy in directions:
            predator_distance = 1.0
            for dist in range(1, 6):
                check_x = (self.agent.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.agent.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                predators = [obj for obj in cell_contents if isinstance(obj, CarnivoreAgent)]
                if predators:
                    predator_distance = dist / 5.0
                    break
            obs.append(predator_distance)

        # Environmental factors
        obs.append(self.model.temperature / 50.0)
        obs.append(self.model.rainfall / 200.0)

        return np.array(obs, dtype=np.float32)

    def _action_to_movement(self, action):
        """Convert action index to movement vector"""
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        if action == 8:
            return (0, 0)  # Stay
        return movements[action]


class JurassicParkCarnivoreEnv(gym.Env):
    """Gymnasium environment for training carnivore agents with PPO"""

    metadata = {'render_modes': []}

    def __init__(self, width=50, height=50, max_steps=200):
        super(JurassicParkCarnivoreEnv, self).__init__()

        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Observation space: 19 features (energy, prey, allies, environment)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(19,), dtype=np.float32
        )

        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)

        self.model = None
        self.agent = None
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        # Create new island
        self.model = IslandModel(
            width=self.width,
            height=self.height,
            num_herbivores=20,  # Plenty of prey
            num_carnivores=1,   # Single predator for training
            temperature=25,
            rainfall=100,
            use_learning_agents=False
        )

        # Replace carnivore with PPO-controlled agent
        carnivores = [a for a in self.model.agents if isinstance(a, CarnivoreAgent)]
        if carnivores:
            for c in carnivores:
                self.model.grid.remove_agent(c)
                c.remove()

        # Add controlled agent
        self.agent = Velociraptor(self.model)
        x = self.model.random.randrange(self.width)
        y = self.model.random.randrange(self.height)
        self.model.grid.place_agent(self.agent, (x, y))

        self.steps_taken = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Execute one step"""
        # Convert action
        dx, dy = self._action_to_movement(action)

        prev_energy = self.agent.energy
        prev_pos = self.agent.pos

        # Calculate distance to nearest prey BEFORE moving
        min_distance_before = float('inf')
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        for dx_scan, dy_scan in directions:
            for dist in range(1, 8):
                check_x = (self.agent.pos[0] + dx_scan * dist) % self.model.grid.width
                check_y = (self.agent.pos[1] + dy_scan * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                if any(isinstance(obj, HerbivoreAgent) for obj in cell_contents):
                    min_distance_before = min(min_distance_before, dist)
                    break

        # Move
        for _ in range(self.agent.speed):
            new_x = (self.agent.pos[0] + dx) % self.model.grid.width
            new_y = (self.agent.pos[1] + dy) % self.model.grid.height
            self.model.grid.move_agent(self.agent, (new_x, new_y))

        # Update direction
        if (new_x, new_y) != prev_pos:
            self.agent.direction = (dx, dy)

        # Calculate distance AFTER moving
        min_distance_after = float('inf')
        for dx_scan, dy_scan in directions:
            for dist in range(1, 8):
                check_x = (self.agent.pos[0] + dx_scan * dist) % self.model.grid.width
                check_y = (self.agent.pos[1] + dy_scan * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                if any(isinstance(obj, HerbivoreAgent) for obj in cell_contents):
                    min_distance_after = min(min_distance_after, dist)
                    break

        # Hunt
        cell_contents = self.model.grid.get_cell_list_contents([self.agent.pos])
        herbivores = [obj for obj in cell_contents if isinstance(obj, HerbivoreAgent)]

        hunted = False
        hunting_attempt = False
        if herbivores:
            prey = self.model.random.choice(herbivores)
            hunting_attempt = True

            if self.model.random.random() > prey.defense:
                self.agent.energy += prey.energy * 0.8
                hunted = True
                self.model.grid.remove_agent(prey)
                prey.remove()

        # Lose energy
        energy_loss = 1.5 * self.model.metabolism_multiplier * self.agent.size_metabolism
        self.agent.energy -= energy_loss

        # Calculate reward
        reward = 0.0
        if hunted:
            reward += 30.0
        if hunting_attempt and not hunted:
            reward += 2.0

        # Proximity reward
        if min_distance_before != float('inf') and min_distance_after != float('inf'):
            if min_distance_after < min_distance_before:
                reward += 1.0
            elif min_distance_after > min_distance_before:
                reward -= 0.5

        reward += 0.2  # Survival

        # Check death
        terminated = False
        if self.agent.energy <= 0:
            reward -= 50.0
            terminated = True

        # Step herbivores and grass
        if not terminated:
            herbivores_list = [a for a in self.model.agents if isinstance(a, HerbivoreAgent)]
            for h in herbivores_list:
                h.step()

            grass_agents = [a for a in self.model.agents if isinstance(a, GrassAgent)]
            for g in grass_agents:
                g.step()

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps

        # Get observation
        if terminated:
            observation = np.zeros(19, dtype=np.float32)
        else:
            observation = self._get_observation()

        info = {
            'steps_alive': self.steps_taken,
            'energy': self.agent.energy if not terminated else 0,
            'hunted': hunted
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation vector"""
        obs = []

        # Own energy
        obs.append(self.agent.energy / 150.0)

        # Scan directions
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Look for prey
        for dx, dy in directions:
            prey_distance = 1.0
            for dist in range(1, 6):
                check_x = (self.agent.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.agent.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                prey = [obj for obj in cell_contents if isinstance(obj, HerbivoreAgent)]
                if prey:
                    prey_distance = dist / 5.0
                    break
            obs.append(prey_distance)

        # Look for allies
        for dx, dy in directions:
            ally_distance = 1.0
            for dist in range(1, 6):
                check_x = (self.agent.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.agent.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                allies = [obj for obj in cell_contents if isinstance(obj, type(self.agent)) and obj != self.agent]
                if allies:
                    ally_distance = dist / 5.0
                    break
            obs.append(ally_distance)

        # Environment
        obs.append(self.model.temperature / 50.0)
        obs.append(self.model.rainfall / 200.0)

        return np.array(obs, dtype=np.float32)

    def _action_to_movement(self, action):
        """Convert action to movement"""
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        if action == 8:
            return (0, 0)
        return movements[action]
