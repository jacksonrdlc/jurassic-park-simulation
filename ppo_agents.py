"""
PPO-Powered Agents for Jurassic Park Visualization
These agents use trained PPO models from stable-baselines3
"""

import numpy as np
from stable_baselines3 import PPO
from my_first_island import (HerbivoreAgent, CarnivoreAgent, GrassAgent,
                              Triceratops, Gallimimus, TRex, Velociraptor)


class PPOHerbivoreAgent(HerbivoreAgent):
    """Herbivore controlled by trained PPO model"""

    # Class-level shared model
    ppo_model = None

    def __init__(self, model_instance):
        super().__init__(model_instance)
        self.steps_alive = 0
        self.total_reward = 0

    @classmethod
    def load_model(cls, model_path):
        """Load trained PPO model"""
        cls.ppo_model = PPO.load(model_path)
        print(f"✅ Loaded PPO herbivore model from {model_path}")

    def get_observation(self):
        """Create observation vector (same as training)"""
        obs = []

        # Own energy
        obs.append(self.energy / 100.0)

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
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height
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
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                predators = [obj for obj in cell_contents if isinstance(obj, CarnivoreAgent)]
                if predators:
                    predator_distance = dist / 5.0
                    break
            obs.append(predator_distance)

        # Environment
        obs.append(self.model.temperature / 50.0)
        obs.append(self.model.rainfall / 200.0)

        return np.array(obs, dtype=np.float32)

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
        """Execute step using PPO policy"""
        if self.pos is None:
            return

        # Get observation
        obs = self.get_observation()

        # Get action from PPO model
        action, _states = self.ppo_model.predict(obs, deterministic=True)

        # Execute action
        prev_pos = self.pos
        dx, dy = self.action_to_movement(int(action))

        for _ in range(self.speed):
            new_x = (self.pos[0] + dx) % self.model.grid.width
            new_y = (self.pos[1] + dy) % self.model.grid.height
            self.model.grid.move_agent(self, (new_x, new_y))

        # Update direction for visualization
        if self.pos != prev_pos:
            self.direction = (dx, dy)

        # Update movement history for trails
        self.movement_history.append(prev_pos)
        if len(self.movement_history) > 3:
            self.movement_history.pop(0)

        # Eat grass
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        grass = [obj for obj in cell_contents if isinstance(obj, GrassAgent)]
        if grass and grass[0].energy > 0:
            energy_gained = grass[0].energy
            self.energy += energy_gained
            grass[0].energy = 0
            grass[0].countdown = grass[0].regrowth_time
            if self.model.random.random() < 0.15:
                self.model.log_event("EAT", f"🌿 {self.species_name} (PPO) ate grass at {self.pos}")

        # Lose energy
        energy_loss = 1 * self.model.metabolism_multiplier * self.size_metabolism
        self.energy -= energy_loss

        # Die if out of energy
        if self.energy <= 0:
            self.model.log_event("DEATH", f"💀 {self.species_name} (PPO) starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()
            return

        # Reproduce
        if self.energy > 100 and self.random.random() < 0.05:
            self.reproduce()

        self.steps_alive += 1


class PPOCarnivoreAgent(CarnivoreAgent):
    """Carnivore controlled by trained PPO model"""

    # Class-level shared model
    ppo_model = None

    def __init__(self, model_instance):
        super().__init__(model_instance)
        self.steps_alive = 0
        self.total_reward = 0

    @classmethod
    def load_model(cls, model_path):
        """Load trained PPO model"""
        cls.ppo_model = PPO.load(model_path)
        print(f"✅ Loaded PPO carnivore model from {model_path}")

    def get_observation(self):
        """Create observation vector"""
        obs = []

        # Own energy
        obs.append(self.energy / 150.0)

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
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height
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
                check_x = (self.pos[0] + dx * dist) % self.model.grid.width
                check_y = (self.pos[1] + dy * dist) % self.model.grid.height
                cell_contents = self.model.grid.get_cell_list_contents([(check_x, check_y)])
                allies = [obj for obj in cell_contents if isinstance(obj, self.__class__) and obj != self]
                if allies:
                    ally_distance = dist / 5.0
                    break
            obs.append(ally_distance)

        # Environment
        obs.append(self.model.temperature / 50.0)
        obs.append(self.model.rainfall / 200.0)

        return np.array(obs, dtype=np.float32)

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
        """Execute step using PPO policy"""
        if self.pos is None:
            return

        # Get observation
        obs = self.get_observation()

        # Get action from PPO model
        action, _states = self.ppo_model.predict(obs, deterministic=True)

        # Execute action
        prev_pos = self.pos
        dx, dy = self.action_to_movement(int(action))

        for _ in range(self.speed):
            new_x = (self.pos[0] + dx) % self.model.grid.width
            new_y = (self.pos[1] + dy) % self.model.grid.height
            self.model.grid.move_agent(self, (new_x, new_y))

        # Update direction
        if self.pos != prev_pos:
            self.direction = (dx, dy)

        # Update movement history
        self.movement_history.append(prev_pos)
        if len(self.movement_history) > 3:
            self.movement_history.pop(0)

        # Hunt
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        herbivores = [obj for obj in cell_contents if isinstance(obj, HerbivoreAgent)]

        if herbivores:
            prey = self.random.choice(herbivores)

            # Pack bonus
            pack_bonus = 1.0
            if hasattr(self, 'pack_bonus'):
                nearby_carnivores = [obj for obj in cell_contents
                                   if isinstance(obj, self.__class__) and obj != self]
                if nearby_carnivores:
                    pack_bonus = self.pack_bonus

            # Attack
            if self.random.random() > prey.defense:
                self.energy += prey.energy * 0.8
                pack_msg = " (pack)" if pack_bonus > 1.0 else ""
                self.model.log_event("HUNT", f"🦖 {self.species_name} (PPO) hunted {prey.species_name}{pack_msg} at {self.pos}")
                self.model.grid.remove_agent(prey)
                prey.remove()
            else:
                self.model.log_event("HUNT", f"🦖 {prey.species_name} defended against {self.species_name} (PPO) at {self.pos}")

        # Lose energy
        energy_loss = 1.5 * self.model.metabolism_multiplier * self.size_metabolism
        self.energy -= energy_loss

        # Die if out of energy
        if self.energy <= 0:
            self.model.log_event("DEATH", f"💀 {self.species_name} (PPO) starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()
            return

        # Reproduce
        if self.energy > 150 and self.random.random() < 0.03:
            self.reproduce()

        self.steps_alive += 1


# PPO variants of specialized species
class PPOTriceratops(PPOHerbivoreAgent, Triceratops):
    def __init__(self, model):
        PPOHerbivoreAgent.__init__(self, model)
        self.energy = 80
        self.defense = 0.5
        self.speed = 1
        self.size_metabolism = 1.5
        self.species_name = "Triceratops"


class PPOGallimimus(PPOHerbivoreAgent, Gallimimus):
    def __init__(self, model):
        PPOHerbivoreAgent.__init__(self, model)
        self.energy = 40
        self.defense = 0.3
        self.speed = 3
        self.size_metabolism = 0.7
        self.species_name = "Gallimimus"


class PPOTRex(PPOCarnivoreAgent, TRex):
    def __init__(self, model):
        PPOCarnivoreAgent.__init__(self, model)
        self.energy = 250
        self.attack_power = 1.0
        self.speed = 1
        self.size_metabolism = 1.8
        self.species_name = "TRex"


class PPOVelociraptor(PPOCarnivoreAgent, Velociraptor):
    def __init__(self, model):
        PPOCarnivoreAgent.__init__(self, model)
        self.energy = 120
        self.attack_power = 0.5
        self.speed = 2
        self.pack_bonus = 1.5
        self.size_metabolism = 0.8
        self.species_name = "Velociraptor"
