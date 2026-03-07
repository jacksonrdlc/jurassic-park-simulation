"""
Gymnasium Environment Wrapper for Jurassic Park Simulation
Enables PPO training using the ACTUAL IslandModel from my_first_island.py.
Bridge pattern: Wraps the Mesa IslandModel to provide a Gymnasium interface.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from my_first_island import (IslandModel, GrassAgent, HerbivoreAgent, CarnivoreAgent,
                              Triceratops, Gallimimus, TRex, Velociraptor)

class HerbivoreEnv(gym.Env):
    """
    Gymnasium environment for training herbivores using the real IslandModel simulation.
    Tracks a single herbivore and returns normalized observations.
    """
    metadata = {'render_modes': []}

    def __init__(self, width=50, height=50, max_steps=500):
        super(HerbivoreEnv, self).__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        
        # Observation space: Box(19,)
        # [energy_norm, grass_dist_8_dirs, predator_dist_8_dirs, temp_norm, rain_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(19,), dtype=np.float32
        )
        
        # Action space: Discrete(9) - N, NE, E, SE, S, SW, W, NW, Stay
        self.action_space = spaces.Discrete(9)
        
        self.island_model = None
        self.tracked_agent = None
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Create fresh IslandModel
        self.island_model = IslandModel(
            width=self.width,
            height=self.height,
            num_herbivores=1, # We'll track this one
            num_carnivores=5, # Predators for challenge
            use_learning_agents=False
        )
        
        # 2. Pick the herbivore to track
        herbivores = [a for a in self.island_model.agents if isinstance(a, HerbivoreAgent)]
        self.tracked_agent = herbivores[0]
        
        self.steps_taken = 0
        return self._get_observation(), {}

    def step(self, action):
        # 1. Execute action on tracked herbivore
        # Map action 0-8 to (dx, dy)
        dx, dy = self._action_to_direction(action)
        
        # Calculate new position (Mesa handles torus/wrapping if configured, but we'll be explicit)
        if dx != 0 or dy != 0:
            new_x = (self.tracked_agent.pos[0] + dx) % self.island_model.grid.width
            new_y = (self.tracked_agent.pos[1] + dy) % self.island_model.grid.height
            
            # Use Mesa API to move
            self.island_model.grid.move_agent(self.tracked_agent, (new_x, new_y))
            self.tracked_agent.direction = (dx, dy)

        # 2. Step the actual IslandModel (updates all other agents, environment, etc.)
        self.island_model.step()
        self.steps_taken += 1

        # 3. Calculate reward & check status
        reward = 0.1 # Survival bonus
        terminated = False
        
        # Check if agent still exists in model.agents
        if self.tracked_agent not in self.island_model.agents or self.tracked_agent.energy <= 0:
            reward = -50.0
            terminated = True
        else:
            # Check for specific events (e.g. eating)
            # We check if the agent just ate (energy increased more than just metabolism loss)
            # Or we can look at the log, but simpler is checking energy delta if we had stored it.
            # However, the task specifies rewards for: eat grass (+10), reproduce (+100)
            
            # Check if agent reproduced (Mesa 3.x: child added to model.agents)
            # In my_first_island, reproduce() halves energy and adds a baby.
            # Since IslandModel.step() runs shuffle_do("step"), it's hard to track exactly.
            # For simplicity, we'll monitor reproduction cooldown changes or count species.
            # But the prompt says: "+10 eat grass, +0.1 survive step, -50 death, +100 reproduce"
            
            # Simple heuristic for this bridge:
            if hasattr(self.tracked_agent, 'last_energy'):
                if self.tracked_agent.energy > self.tracked_agent.last_energy:
                    reward += 10.0 # Ate grass
                elif self.tracked_agent.energy < self.tracked_agent.last_energy / 2.1: # Threshold for reproduction energy split
                     reward += 100.0 # Reproduced
            
            self.tracked_agent.last_energy = self.tracked_agent.energy

        truncated = self.steps_taken >= self.max_steps
        
        obs = self._get_observation() if not terminated else np.zeros(19, dtype=np.float32)
        info = {
            "energy": self.tracked_agent.energy if self.tracked_agent in self.island_model.agents else 0
        }
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        if self.tracked_agent not in self.island_model.agents:
            return np.zeros(19, dtype=np.float32)
            
        obs = []
        # 1. energy_norm
        obs.append(min(self.tracked_agent.energy / 100.0, 1.0))
        
        # Directions for scanning
        directions = [(-1,1), (0,1), (1,1), (-1,0), (1,0), (-1,-1), (0,-1), (1,-1)]
        
        # 2. grass_dist_8_dirs (radius 5)
        for dx, dy in directions:
            found = False
            for r in range(1, 6):
                px = (self.tracked_agent.pos[0] + dx * r) % self.island_model.grid.width
                py = (self.tracked_agent.pos[1] + dy * r) % self.island_model.grid.height
                cell = self.island_model.grid.get_cell_list_contents([(px, py)])
                if any(isinstance(a, GrassAgent) and a.energy > 0 for a in cell):
                    obs.append(r / 5.0)
                    found = True
                    break
            if not found: obs.append(1.0)

        # 3. predator_dist_8_dirs (radius 5)
        for dx, dy in directions:
            found = False
            for r in range(1, 6):
                px = (self.tracked_agent.pos[0] + dx * r) % self.island_model.grid.width
                py = (self.tracked_agent.pos[1] + dy * r) % self.island_model.grid.height
                cell = self.island_model.grid.get_cell_list_contents([(px, py)])
                if any(isinstance(a, CarnivoreAgent) for a in cell):
                    obs.append(r / 5.0)
                    found = True
                    break
            if not found: obs.append(1.0)
            
        # 4. temp_norm, rain_norm
        obs.append(self.island_model.temperature / 50.0)
        obs.append(self.island_model.rainfall / 200.0)
        
        return np.array(obs, dtype=np.float32)

    def _action_to_direction(self, action):
        # 0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW, 8:Stay
        # Note: Mesa y-up or y-down depends on rendering, usually (0,1) is North
        mapping = {
            0: (0, 1),   # N
            1: (1, 1),   # NE
            2: (1, 0),   # E
            3: (1, -1),  # SE
            4: (0, -1),  # S
            5: (-1, -1), # SW
            6: (-1, 0),  # W
            7: (-1, 1),  # NW
            8: (0, 0)    # Stay
        }
        return mapping.get(action, (0, 0))

class CarnivoreEnv(gym.Env):
    """
    Gymnasium environment for training carnivores using the real IslandModel simulation.
    """
    metadata = {'render_modes': []}

    def __init__(self, width=50, height=50, max_steps=500):
        super(CarnivoreEnv, self).__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.island_model = None
        self.tracked_agent = None
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.island_model = IslandModel(
            width=self.width, height=self.height,
            num_herbivores=20, num_carnivores=1,
            use_learning_agents=False
        )
        self.tracked_agent = [a for a in self.island_model.agents if isinstance(a, CarnivoreAgent)][0]
        self.steps_taken = 0
        return self._get_observation(), {}

    def step(self, action):
        # Store state for reward calculation
        if self.tracked_agent in self.island_model.agents:
            pre_pos = self.tracked_agent.pos
            pre_energy = self.tracked_agent.energy
            
            # Find nearest prey before move
            prey_loc = self._find_nearest_prey_pos()
            dist_before = self._dist(pre_pos, prey_loc) if prey_loc else None
        
        # Execute action
        dx, dy = self._action_to_direction(action)
        if (dx != 0 or dy != 0) and self.tracked_agent in self.island_model.agents:
            new_x = (self.tracked_agent.pos[0] + dx) % self.island_model.grid.width
            new_y = (self.tracked_agent.pos[1] + dy) % self.island_model.grid.height
            self.island_model.grid.move_agent(self.tracked_agent, (new_x, new_y))
            self.tracked_agent.direction = (dx, dy)

        self.island_model.step()
        self.steps_taken += 1

        reward = 0.0
        terminated = False
        
        if self.tracked_agent not in self.island_model.agents or self.tracked_agent.energy <= 0:
            reward = -50.0
            terminated = True
        else:
            # Reward: +1.0 step closer, +2.0 hunt attempt, +30 kill, +150 reproduce
            
            # Step closer
            prey_loc = self._find_nearest_prey_pos()
            dist_after = self._dist(self.tracked_agent.pos, prey_loc) if prey_loc else None
            if dist_before and dist_after and dist_after < dist_before:
                reward += 1.0
            
            # Kill/Hunt (Check energy gain)
            if self.tracked_agent.energy > pre_energy:
                reward += 30.0 # Successful kill
            
            # Reproduce (Energy split)
            if self.tracked_agent.energy < pre_energy / 2.1:
                reward += 150.0
                
            # Hunt attempt (Mesa logs)
            # For simplicity, we'll assume being on same cell as prey is an attempt
            cell = self.island_model.grid.get_cell_list_contents([self.tracked_agent.pos])
            if any(isinstance(a, HerbivoreAgent) for a in cell):
                reward += 2.0

        truncated = self.steps_taken >= self.max_steps
        obs = self._get_observation() if not terminated else np.zeros(19, dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        if self.tracked_agent not in self.island_model.agents:
            return np.zeros(19, dtype=np.float32)
        obs = [min(self.tracked_agent.energy / 150.0, 1.0)]
        directions = [(-1,1), (0,1), (1,1), (-1,0), (1,0), (-1,-1), (0,-1), (1,-1)]
        # Prey dist
        for dx, dy in directions:
            found = False
            for r in range(1, 6):
                px = (self.tracked_agent.pos[0] + dx * r) % self.island_model.grid.width
                py = (self.tracked_agent.pos[1] + dy * r) % self.island_model.grid.height
                if any(isinstance(a, HerbivoreAgent) for a in self.island_model.grid.get_cell_list_contents([(px, py)])):
                    obs.append(r / 5.0); found = True; break
            if not found: obs.append(1.0)
        # Ally dist
        for dx, dy in directions:
            found = False
            for r in range(1, 6):
                px = (self.tracked_agent.pos[0] + dx * r) % self.island_model.grid.width
                py = (self.tracked_agent.pos[1] + dy * r) % self.island_model.grid.height
                if any(isinstance(a, CarnivoreAgent) and a != self.tracked_agent for a in self.island_model.grid.get_cell_list_contents([(px, py)])):
                    obs.append(r / 5.0); found = True; break
            if not found: obs.append(1.0)
        obs.append(self.island_model.temperature / 50.0)
        obs.append(self.island_model.rainfall / 200.0)
        return np.array(obs, dtype=np.float32)

    def _find_nearest_prey_pos(self):
        prey = [a.pos for a in self.island_model.agents if isinstance(a, HerbivoreAgent)]
        if not prey: return None
        return min(prey, key=lambda p: self._dist(self.tracked_agent.pos, p))

    def _dist(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def _action_to_direction(self, action):
        mapping = {0:(0,1), 1:(1,1), 2:(1,0), 3:(1,-1), 4:(0,-1), 5:(-1,-1), 6:(-1,0), 7:(-1,1), 8:(0,0)}
        return mapping.get(action, (0,0))
