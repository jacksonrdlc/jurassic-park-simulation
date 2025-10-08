"""
PyTorch Learning Agents for Jurassic Park Simulation
Uses neural networks and reinforcement learning for intelligent agent behavior
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from my_first_island import HerbivoreAgent, CarnivoreAgent, GrassAgent, Triceratops, Gallimimus, TRex, Velociraptor


class ExperienceReplay:
    """Experience replay buffer for storing and sampling agent experiences"""

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def size(self):
        return len(self.buffer)


class NeuralNetwork(nn.Module):
    """Neural network for decision making"""

    def __init__(self, input_size, output_size, hidden_layers=[64, 32]):
        super(NeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)


class LearningHerbivoreAgent(HerbivoreAgent):
    """Herbivore that learns optimal movement using neural networks"""

    # Class-level shared components
    neural_net = None
    optimizer = None
    replay_buffer = None
    epsilon = 0.2  # Exploration rate
    training_mode = True

    def __init__(self, model_instance):
        super().__init__(model_instance)

        # Initialize shared neural network if not exists
        if LearningHerbivoreAgent.neural_net is None:
            # Input: own energy, 8 directions (grass, predators), temperature, rainfall
            input_size = 1 + 8 + 8 + 2  # = 19 features
            output_size = 9  # 8 directions + stay
            LearningHerbivoreAgent.neural_net = NeuralNetwork(input_size, output_size)
            LearningHerbivoreAgent.optimizer = optim.Adam(LearningHerbivoreAgent.neural_net.parameters(), lr=0.001)
            LearningHerbivoreAgent.replay_buffer = ExperienceReplay()

        self.previous_state = None
        self.previous_action = None
        self.steps_alive = 0
        self.total_reward = 0

    def get_observation(self):
        """Create state observation vector"""
        obs = []

        # Own energy (normalized)
        obs.append(self.energy / 100.0)

        # Scan 8 directions for grass (distance to nearest grass)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            # Look up to 5 cells away in each direction
            grass_distance = 1.0  # Normalized (1.0 = no grass found)
            for dist in range(1, 6):
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height

                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                grass = [obj for obj in cell_contents if isinstance(obj, GrassAgent) and obj.energy > 0]

                if grass:
                    grass_distance = dist / 5.0  # Normalize to 0-1
                    break

            obs.append(grass_distance)

        # Scan 8 directions for predators (distance to nearest carnivore)
        for dx, dy in directions:
            predator_distance = 1.0  # 1.0 = no predator
            for dist in range(1, 6):
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height

                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                predators = [obj for obj in cell_contents if isinstance(obj, CarnivoreAgent)]

                if predators:
                    predator_distance = dist / 5.0
                    break

            obs.append(predator_distance)

        # Environmental factors
        obs.append(self.model.temperature / 50.0)  # Normalized
        obs.append(self.model.rainfall / 200.0)  # Normalized

        return np.array(obs, dtype=np.float32)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if self.training_mode and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, 8)
        else:
            # Exploit: use neural network
            self.neural_net.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                predictions = self.neural_net(state_tensor)[0]
                return torch.argmax(predictions).item()

    def action_to_movement(self, action):
        """Convert action index to grid movement"""
        # 0-7: 8 directions, 8: stay
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        if action == 8:
            return (0, 0)  # Stay
        return movements[action]

    def step(self):
        """Execute one step with learning"""
        # Safety check: agent already removed from grid
        if self.pos is None:
            return

        # Get current state
        state = self.get_observation()

        # Choose action
        action = self.choose_action(state)

        # Record previous state and action for reward calculation
        prev_energy = self.energy
        prev_pos = self.pos

        # Execute action (move)
        dx, dy = self.action_to_movement(action)
        for _ in range(self.speed):
            new_x = (self.pos[0] + dx) % self.model.grid.width
            new_y = (self.pos[1] + dy) % self.model.grid.height
            self.model.grid.move_agent(self, (new_x, new_y))

        # Eat grass
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        grass = [obj for obj in cell_contents if isinstance(obj, GrassAgent)]
        ate_grass = False
        if grass and grass[0].energy > 0:
            energy_gained = grass[0].energy
            self.energy += energy_gained
            grass[0].energy = 0
            grass[0].countdown = grass[0].regrowth_time
            ate_grass = True
            if self.model.random.random() < 0.15:
                self.model.log_event("EAT", f"🌿 {self.species_name} (AI) ate grass at {self.pos}")

        # Lose energy (affected by size!)
        energy_loss = 1 * self.model.metabolism_multiplier * self.size_metabolism
        self.energy -= energy_loss

        # Calculate reward
        reward = 0.0

        # Reward for eating
        if ate_grass:
            reward += 10.0

        # Reward for surviving
        reward += 0.1

        # Penalty for energy loss
        reward -= 0.05

        # Check if died
        done = False
        if self.energy <= 0:
            reward -= 50.0  # Death penalty
            done = True
            self.model.log_event("DEATH", f"💀 {self.species_name} (AI) starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()

        # Get next state
        next_state = self.get_observation() if not done else np.zeros_like(state)

        # Store experience
        if self.training_mode:
            self.replay_buffer.add(state, action, reward, next_state, done)

        self.total_reward += reward
        self.steps_alive += 1

        # Train periodically
        if self.training_mode and self.replay_buffer.size() > 100 and self.model.steps % 50 == 0:
            self.train_network()

        # Reproduce
        if not done and self.energy > 100 and self.random.random() < 0.05:
            self.reproduce()

    def train_network(self):
        """Train the neural network using experience replay"""
        if self.replay_buffer.size() < 32:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(32)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        # Compute Q-values
        self.neural_net.train()
        current_q = self.neural_net(states_tensor)

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.neural_net(next_states_tensor)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q = current_q.clone()

            for i in range(len(states)):
                if dones[i]:
                    target = rewards_tensor[i]
                else:
                    target = rewards_tensor[i] + 0.95 * max_next_q[i]
                target_q[i][actions_tensor[i]] = target

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class LearningCarnivoreAgent(CarnivoreAgent):
    """Carnivore that learns optimal hunting using neural networks"""

    # Class-level shared components
    neural_net = None
    optimizer = None
    replay_buffer = None
    epsilon = 0.2
    training_mode = True

    def __init__(self, model_instance):
        super().__init__(model_instance)

        # Initialize shared neural network if not exists
        if LearningCarnivoreAgent.neural_net is None:
            input_size = 1 + 8 + 8 + 2  # = 19 features (energy, prey, allies, env)
            output_size = 9  # 8 directions + stay
            LearningCarnivoreAgent.neural_net = NeuralNetwork(input_size, output_size)
            LearningCarnivoreAgent.optimizer = optim.Adam(LearningCarnivoreAgent.neural_net.parameters(), lr=0.001)
            LearningCarnivoreAgent.replay_buffer = ExperienceReplay()

        self.steps_alive = 0
        self.total_reward = 0

    def get_observation(self):
        """Create state observation vector"""
        obs = []

        # Own energy
        obs.append(self.energy / 150.0)

        # Scan 8 directions for prey
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            prey_distance = 1.0
            for dist in range(1, 6):
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height

                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                prey = [obj for obj in cell_contents if isinstance(obj, HerbivoreAgent)]

                if prey:
                    prey_distance = dist / 5.0
                    break

            obs.append(prey_distance)

        # Scan 8 directions for allies (for pack hunting)
        for dx, dy in directions:
            ally_distance = 1.0
            for dist in range(1, 6):
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height

                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                allies = [obj for obj in cell_contents if isinstance(obj, self.__class__) and obj != self]

                if allies:
                    ally_distance = dist / 5.0
                    break

            obs.append(ally_distance)

        # Environmental factors
        obs.append(self.model.temperature / 50.0)
        obs.append(self.model.rainfall / 200.0)

        return np.array(obs, dtype=np.float32)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if self.training_mode and random.random() < self.epsilon:
            return random.randint(0, 8)
        else:
            self.neural_net.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                predictions = self.neural_net(state_tensor)[0]
                return torch.argmax(predictions).item()

    def action_to_movement(self, action):
        """Convert action to movement"""
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        if action == 8:
            return (0, 0)
        return movements[action]

    def step(self):
        """Execute one step with learning"""
        # Safety check: agent already removed from grid
        if self.pos is None:
            return

        state = self.get_observation()
        action = self.choose_action(state)

        prev_energy = self.energy
        prev_pos = self.pos

        # Calculate distance to nearest prey BEFORE moving
        min_distance_before = float('inf')
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        for dx, dy in directions:
            for dist in range(1, 8):
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                if any(isinstance(obj, HerbivoreAgent) for obj in cell_contents):
                    min_distance_before = min(min_distance_before, dist)
                    break

        # Move
        dx, dy = self.action_to_movement(action)
        for _ in range(self.speed):
            new_x = (self.pos[0] + dx) % self.model.grid.width
            new_y = (self.pos[1] + dy) % self.model.grid.height
            self.model.grid.move_agent(self, (new_x, new_y))

        # Calculate distance to nearest prey AFTER moving
        min_distance_after = float('inf')
        for dx, dy in directions:
            for dist in range(1, 8):
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                if any(isinstance(obj, HerbivoreAgent) for obj in cell_contents):
                    min_distance_after = min(min_distance_after, dist)
                    break

        # Hunt
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        herbivores = [obj for obj in cell_contents if isinstance(obj, HerbivoreAgent)]

        hunted = False
        hunting_attempt = False
        if herbivores:
            prey = self.random.choice(herbivores)
            hunting_attempt = True

            # Pack bonus
            pack_bonus = 1.0
            if hasattr(self, 'pack_bonus'):
                nearby_carnivores = [obj for obj in cell_contents
                                   if isinstance(obj, self.__class__) and obj != self]
                if nearby_carnivores:
                    pack_bonus = self.pack_bonus

            # Attack
            if self.random.random() > prey.defense:
                self.energy += prey.energy * 0.8  # Kept at 0.8 for learning agents (already buffed)
                hunted = True
                pack_msg = " (pack)" if pack_bonus > 1.0 else ""
                self.model.log_event("HUNT", f"🦖 {self.species_name} (AI) hunted {prey.species_name}{pack_msg} at {self.pos}")
                self.model.grid.remove_agent(prey)
                prey.remove()
            else:
                self.model.log_event("HUNT", f"🦖 {prey.species_name} defended against {self.species_name} (AI) at {self.pos}")

        # Lose energy (affected by size!)
        energy_loss = 1.5 * self.model.metabolism_multiplier * self.size_metabolism
        self.energy -= energy_loss

        # Calculate reward
        reward = 0.0

        # Big reward for successful hunt
        if hunted:
            reward += 30.0  # Increased from 20.0

        # Small reward for hunting attempt (even if failed)
        if hunting_attempt and not hunted:
            reward += 2.0  # Encourages trying to hunt

        # Proximity reward - reward for moving closer to prey
        if min_distance_before != float('inf') and min_distance_after != float('inf'):
            if min_distance_after < min_distance_before:
                # Moved closer to prey - good!
                reward += 1.0
            elif min_distance_after > min_distance_before:
                # Moved away from prey - bad
                reward -= 0.5

        # Small survival reward
        reward += 0.2  # Increased from 0.1

        # Check death
        done = False
        if self.energy <= 0:
            reward -= 50.0
            done = True
            self.model.log_event("DEATH", f"💀 {self.species_name} (AI) starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()

        next_state = self.get_observation() if not done else np.zeros_like(state)

        if self.training_mode:
            self.replay_buffer.add(state, action, reward, next_state, done)

        self.total_reward += reward
        self.steps_alive += 1

        # Train
        if self.training_mode and self.replay_buffer.size() > 100 and self.model.steps % 50 == 0:
            self.train_network()

        # Reproduce - REDUCED threshold from 200 to 150 for easier breeding
        if not done and self.energy > 150 and self.random.random() < 0.03:
            self.reproduce()

    def train_network(self):
        """Train using experience replay"""
        if self.replay_buffer.size() < 32:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(32)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        self.neural_net.train()
        current_q = self.neural_net(states_tensor)

        with torch.no_grad():
            next_q = self.neural_net(next_states_tensor)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q = current_q.clone()

            for i in range(len(states)):
                if dones[i]:
                    target = rewards_tensor[i]
                else:
                    target = rewards_tensor[i] + 0.95 * max_next_q[i]
                target_q[i][actions_tensor[i]] = target

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Learning variants of specialized species
class LearningTriceratops(LearningHerbivoreAgent, Triceratops):
    def __init__(self, model):
        LearningHerbivoreAgent.__init__(self, model)
        self.energy = 80
        self.defense = 0.5  # NERFED from 0.7
        self.speed = 1
        self.size_metabolism = 1.5  # LARGE dinosaur
        self.species_name = "Triceratops"


class LearningGallimimus(LearningHerbivoreAgent, Gallimimus):
    def __init__(self, model):
        LearningHerbivoreAgent.__init__(self, model)
        self.energy = 40
        self.defense = 0.3  # BUFFED from 0.2
        self.speed = 3
        self.size_metabolism = 0.7  # SMALL dinosaur
        self.species_name = "Gallimimus"


class LearningTRex(LearningCarnivoreAgent, TRex):
    def __init__(self, model):
        LearningCarnivoreAgent.__init__(self, model)
        self.energy = 250  # Increased from 150 to give more time to learn
        self.attack_power = 1.0
        self.speed = 1
        self.size_metabolism = 1.8  # MASSIVE dinosaur
        self.species_name = "TRex"


class LearningVelociraptor(LearningCarnivoreAgent, Velociraptor):
    def __init__(self, model):
        LearningCarnivoreAgent.__init__(self, model)
        self.energy = 120  # Increased from 60 to give more time to learn
        self.attack_power = 0.5
        self.speed = 2
        self.pack_bonus = 1.5
        self.size_metabolism = 0.8  # SMALL-MEDIUM dinosaur
        self.species_name = "Velociraptor"
