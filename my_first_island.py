# island_model.py - Complete Phase 3 with proper import structure

# ============= ALL IMPORTS AT THE TOP =============
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# ============= AGENT CLASSES =============

class GrassAgent(Agent):
    """A grass patch that can be eaten and regrows"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 10
        self.regrowth_time = 5
        self.countdown = 0
    
    def step(self):
        if self.energy == 0:
            # Growth affected by rainfall
            growth_chance = self.model.grass_growth_rate
            if self.random.random() < growth_chance:
                self.countdown -= 1
                if self.countdown <= 0:
                    self.energy = 10


class HerbivoreAgent(Agent):
    """A plant-eating dinosaur"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 50
        self.defense = 0.3  # Base defense chance
        self.speed = 1  # Base movement speed (cells per step)
        self.species_name = "Herbivore"
        self.size_metabolism = 1.0  # Size-based energy consumption multiplier
        self.direction = (1, 0)  # Direction facing (dx, dy)
        self.movement_history = []  # Track recent positions for trails
        self.is_moving = False  # Track if agent moved this step
        self.reproduction_cooldown = 0  # Steps until can reproduce again

    def is_walkable(self, pos):
        """Check if a position is walkable based on terrain"""
        if self.model.terrain_map is None:
            return True  # No terrain restrictions

        from terrain_generator import TerrainType, TERRAIN_WALKABLE
        x, y = pos
        terrain_type = TerrainType(self.model.terrain_map[y, x])
        return TERRAIN_WALKABLE[terrain_type]

    def find_nearest_grass(self):
        """Find nearest grass within vision range"""
        search_radius = 10  # Can see 10 cells away
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=search_radius
        )

        # Find grass patches with energy
        grass_locations = []
        for pos in neighborhood:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            for agent in cell_contents:
                if isinstance(agent, GrassAgent) and agent.energy > 0:
                    grass_locations.append(pos)
                    break

        if grass_locations:
            # Find closest grass
            closest_grass = min(grass_locations,
                              key=lambda pos: abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1]))
            return closest_grass
        return None

    def step(self):
        # Move (speed determines how many cells to move)
        old_pos = self.pos
        current_pos = self.pos

        # Look for nearby grass to move towards
        target_grass = self.find_nearest_grass()

        for _ in range(self.speed):
            possible_steps = self.model.grid.get_neighborhood(
                current_pos, moore=True, include_center=False
            )
            # Filter out non-walkable terrain
            walkable_steps = [pos for pos in possible_steps if self.is_walkable(pos)]

            if walkable_steps:
                # If we see grass, move towards it
                if target_grass:
                    # Calculate which step gets us closer to grass
                    best_step = min(walkable_steps,
                                   key=lambda pos: abs(pos[0] - target_grass[0]) + abs(pos[1] - target_grass[1]))
                    current_pos = best_step
                else:
                    # Random walk if no grass visible
                    new_position = self.random.choice(walkable_steps)
                    current_pos = new_position
            else:
                # No walkable positions, stay in place
                break

        # Track movement history for trails
        self.movement_history.append(old_pos)
        if len(self.movement_history) > 3:  # Keep last 3 positions
            self.movement_history.pop(0)

        # Calculate direction based on movement
        self.is_moving = (current_pos != old_pos)
        if self.is_moving:
            self.direction = (current_pos[0] - old_pos[0], current_pos[1] - old_pos[1])

        self.model.grid.move_agent(self, current_pos)

        # Eat grass
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        grass = [obj for obj in cell_contents if isinstance(obj, GrassAgent)]
        if grass and grass[0].energy > 0:
            energy_gained = grass[0].energy
            self.energy += energy_gained
            grass[0].energy = 0
            grass[0].countdown = grass[0].regrowth_time
            # Log eating occasionally to avoid spam
            if self.model.random.random() < 0.15:  # 15% chance to log
                self.model.log_event("EAT", f"🌿 {self.species_name} ate grass at {self.pos}")
        
        # Lose energy (affected by temperature AND body size!)
        energy_loss = 1 * self.model.metabolism_multiplier * self.size_metabolism
        self.energy -= energy_loss

        # Update reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        # Die if out of energy
        if self.energy <= 0:
            self.model.log_event("DEATH", f"💀 {self.species_name} starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()  # Mesa 3.x: removes from model.agents

        # Reproduce if enough energy and not on cooldown
        # BALANCED: Higher threshold (150) and lower chance (1%) for realistic 3-hour growth to 8k dinos
        if self.energy > 150 and self.reproduction_cooldown == 0 and self.random.random() < 0.01:
            self.reproduce()
    
    def reproduce(self):
        self.energy /= 2
        self.reproduction_cooldown = 100  # Must wait 100 steps before reproducing again
        baby = self.__class__(self.model)  # Create same species as parent
        self.model.grid.place_agent(baby, self.pos)
        self.model.log_event("BIRTH", f"🦕 {self.species_name} born at {self.pos}")


class CarnivoreAgent(Agent):
    """A meat-eating dinosaur"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 100
        self.attack_power = 0.8  # Base attack power
        self.speed = 1  # Base movement speed
        self.species_name = "Carnivore"
        self.size_metabolism = 1.0  # Size-based energy consumption multiplier
        self.direction = (1, 0)  # Direction facing (dx, dy)
        self.movement_history = []  # Track recent positions for trails
        self.is_moving = False  # Track if agent moved this step
        self.reproduction_cooldown = 0  # Steps until can reproduce again

    def is_walkable(self, pos):
        """Check if a position is walkable based on terrain"""
        if self.model.terrain_map is None:
            return True  # No terrain restrictions

        from terrain_generator import TerrainType, TERRAIN_WALKABLE
        x, y = pos
        terrain_type = TerrainType(self.model.terrain_map[y, x])
        return TERRAIN_WALKABLE[terrain_type]

    def find_nearest_prey(self):
        """Find nearest herbivore within vision range"""
        search_radius = 15  # Carnivores can see 15 cells away
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=search_radius
        )

        # Find herbivores
        prey_locations = []
        for pos in neighborhood:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            for agent in cell_contents:
                if isinstance(agent, HerbivoreAgent):
                    prey_locations.append(pos)
                    break

        if prey_locations:
            # Find closest prey
            closest_prey = min(prey_locations,
                             key=lambda pos: abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1]))
            return closest_prey
        return None

    def step(self):
        # Move (speed determines how many cells to move)
        old_pos = self.pos
        current_pos = self.pos

        # Look for nearby prey to hunt
        target_prey = self.find_nearest_prey()

        for _ in range(self.speed):
            possible_steps = self.model.grid.get_neighborhood(
                current_pos, moore=True, include_center=False
            )
            # Filter out non-walkable terrain
            walkable_steps = [pos for pos in possible_steps if self.is_walkable(pos)]

            if walkable_steps:
                # If we see prey, move towards it (hunt!)
                if target_prey:
                    # Calculate which step gets us closer to prey
                    best_step = min(walkable_steps,
                                   key=lambda pos: abs(pos[0] - target_prey[0]) + abs(pos[1] - target_prey[1]))
                    current_pos = best_step
                else:
                    # Random patrol if no prey visible
                    new_position = self.random.choice(walkable_steps)
                    current_pos = new_position
            else:
                # No walkable positions, stay in place
                break

        # Track movement history for trails
        self.movement_history.append(old_pos)
        if len(self.movement_history) > 3:  # Keep last 3 positions
            self.movement_history.pop(0)

        # Calculate direction based on movement
        self.is_moving = (current_pos != old_pos)
        if self.is_moving:
            self.direction = (current_pos[0] - old_pos[0], current_pos[1] - old_pos[1])

        self.model.grid.move_agent(self, current_pos)

        # Hunt
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        herbivores = [obj for obj in cell_contents if isinstance(obj, HerbivoreAgent)]
        if herbivores:
            prey = self.random.choice(herbivores)

            # Check for pack bonus (count nearby carnivores of same type)
            pack_bonus = 1.0
            if hasattr(self, 'pack_bonus'):
                nearby_carnivores = [obj for obj in cell_contents
                                   if isinstance(obj, self.__class__) and obj != self]
                if nearby_carnivores:
                    pack_bonus = self.pack_bonus

            # Calculate effective attack power
            effective_attack = self.attack_power * pack_bonus

            # Check if prey defends successfully
            if self.random.random() > prey.defense:
                # Successful hunt - BUFFED energy gain from 0.5 to 0.75
                self.energy += prey.energy * 0.75
                pack_msg = " (pack hunt)" if pack_bonus > 1.0 else ""
                self.model.log_event("HUNT", f"🦖 {self.species_name} hunted {prey.species_name}{pack_msg} at {self.pos}")
                self.model.grid.remove_agent(prey)
                prey.remove()  # Mesa 3.x: removes from model.agents
            else:
                # Prey defended successfully
                self.model.log_event("HUNT", f"🦖 {prey.species_name} defended against {self.species_name} at {self.pos}")
        
        # Lose energy (affected by temperature AND body size!)
        energy_loss = 2 * self.model.metabolism_multiplier * self.size_metabolism
        self.energy -= energy_loss

        # Update reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        # Die
        if self.energy <= 0:
            self.model.log_event("DEATH", f"💀 {self.species_name} starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()  # Mesa 3.x: removes from model.agents

        # Reproduce if enough energy and not on cooldown
        # BALANCED: Higher threshold (200) and lower chance (0.8%) for realistic 3-hour growth to 8k dinos
        if self.energy > 200 and self.reproduction_cooldown == 0 and self.random.random() < 0.008:
            self.reproduce()
    
    def reproduce(self):
        self.energy /= 2
        self.reproduction_cooldown = 150  # Must wait 150 steps before reproducing again (longer than herbivores)
        baby = self.__class__(self.model)  # Create same species as parent
        self.model.grid.place_agent(baby, self.pos)
        self.model.log_event("BIRTH", f"🦖 {self.species_name} born at {self.pos}")


# ============= SPECIALIZED SPECIES =============

class Triceratops(HerbivoreAgent):
    """Heavily armored herbivore - slow but tough (LARGE = high metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 80
        self.defense = 0.5  # 50% chance to survive attack (NERFED from 0.7)
        self.speed = 1
        self.size_metabolism = 1.5  # LARGE dinosaur = 50% more energy consumption
        self.species_name = "Triceratops"


class Gallimimus(HerbivoreAgent):
    """Fast herbivore - quick but fragile (SMALL = low metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 40
        self.defense = 0.3  # 30% chance to survive (BUFFED from 0.2)
        self.speed = 3  # Can move 3 cells per step
        self.size_metabolism = 0.7  # SMALL dinosaur = 30% less energy consumption
        self.species_name = "Gallimimus"


class TRex(CarnivoreAgent):
    """Apex predator - slow but deadly (MASSIVE = very high metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 150
        self.attack_power = 1.0  # Can kill anything
        self.speed = 1
        self.size_metabolism = 1.8  # MASSIVE dinosaur = 80% more energy consumption
        self.species_name = "TRex"


class Velociraptor(CarnivoreAgent):
    """Pack hunter - fast and coordinated (SMALL = low metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 60
        self.attack_power = 0.5
        self.speed = 2
        self.pack_bonus = 1.5  # Bonus when hunting with others
        self.size_metabolism = 0.8  # SMALL-MEDIUM dinosaur = 20% less energy consumption
        self.species_name = "Velociraptor"


class Stegosaurus(HerbivoreAgent):
    """Plated herbivore - slow but deadly tail spikes (LARGE = high metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 90
        self.defense = 0.70  # 70% chance to survive (deadly tail spikes!)
        self.speed = 1  # Very slow like real Stegosaurus
        self.size_metabolism = 1.6  # LARGE dinosaur = 60% more energy consumption
        self.species_name = "Stegosaurus"
        self.can_counterattack = True  # Special ability!

    def step(self):
        # Check for nearby carnivores to potentially counterattack
        if hasattr(self, 'can_counterattack') and self.can_counterattack:
            cell_contents = self.model.grid.get_cell_list_contents([self.pos])
            carnivores = [obj for obj in cell_contents if isinstance(obj, CarnivoreAgent)]
            if carnivores and self.random.random() < 0.30:  # 30% chance to strike with tail
                attacker = self.random.choice(carnivores)
                attacker.energy -= 30  # Tail spike damage!
                self.model.log_event("HUNT", f"⚔️ Stegosaurus counterattacked {attacker.species_name} at {self.pos}!")

        # Normal herbivore behavior
        super().step()


class Pachycephalosaurus(HerbivoreAgent):
    """Dome-headed herbivore - fast, territorial, can head-butt (MEDIUM metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 70
        self.defense = 0.45  # 45% defense (head-butting, intimidation)
        self.speed = 2  # Bipedal, fairly fast
        self.size_metabolism = 0.9  # MEDIUM dinosaur = 10% less energy consumption
        self.species_name = "Pachycephalosaurus"
        self.territorial = True

    def step(self):
        # Territorial behavior - head-butt other Pachycephalosaurus
        if self.territorial and self.random.random() < 0.05:  # 5% chance per step
            cell_contents = self.model.grid.get_cell_list_contents([self.pos])
            rivals = [obj for obj in cell_contents
                     if isinstance(obj, Pachycephalosaurus) and obj != self]
            if rivals:
                rival = self.random.choice(rivals)
                # Head-butt contest - both lose some energy
                energy_loss = 10
                self.energy -= energy_loss
                rival.energy -= energy_loss
                if self.random.random() < 0.3:  # 30% chance to log
                    self.model.log_event("INFO", f"💥 Pachycephalosaurus head-butt contest at {self.pos}!")

        # Normal herbivore behavior
        super().step()


class Brachiosaurus(HerbivoreAgent):
    """Massive sauropod - slow, high browser, huge (VERY LARGE = very high metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 150  # Massive starting energy
        self.defense = 0.85  # 85% defense (size intimidates predators)
        self.speed = 1  # Very slow, cannot run
        self.size_metabolism = 2.2  # MASSIVE dinosaur = 120% more energy consumption
        self.species_name = "Brachiosaurus"
        self.reproduce_threshold = 180  # Needs more energy to reproduce

    def step(self):
        # Massive size discourages weak predators
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        carnivores = [obj for obj in cell_contents if isinstance(obj, CarnivoreAgent)]
        for carnivore in carnivores:
            # Weak carnivores won't even try to attack
            if carnivore.energy < 50 and self.random.random() < 0.7:  # 70% chance to flee
                # Carnivore backs off
                if self.random.random() < 0.2:  # 20% chance to log
                    self.model.log_event("INFO", f"🦖 {carnivore.species_name} intimidated by Brachiosaurus size at {self.pos}")

        # Normal herbivore behavior
        super().step()

    def reproduce(self):
        # Brachiosaurus needs more energy to reproduce
        if self.energy > self.reproduce_threshold:
            self.energy /= 2
            self.reproduction_cooldown = 200  # Long cooldown for massive dinosaur
            baby = self.__class__(self.model)
            self.model.grid.place_agent(baby, self.pos)
            self.model.log_event("BIRTH", f"🦕 {self.species_name} born at {self.pos}")


class Archeopteryx(HerbivoreAgent):
    """Proto-bird - very fast, can glide, insectivore (VERY SMALL = low metabolism)"""
    def __init__(self, model):
        super().__init__(model)
        self.energy = 30
        self.defense = 0.15  # 15% defense (small, relies on escape)
        self.speed = 3  # Very fast (flying/gliding)
        self.size_metabolism = 0.5  # VERY SMALL = 50% less energy consumption
        self.species_name = "Archeopteryx"
        self.can_glide = True
        self.reproduce_threshold = 40  # Low reproduction threshold

    def step(self):
        # Gliding escape mechanism when threatened
        if self.can_glide and self.random.random() < 0.10:  # 10% chance to check
            cell_contents = self.model.grid.get_cell_list_contents([self.pos])
            carnivores = [obj for obj in cell_contents if isinstance(obj, CarnivoreAgent)]
            if carnivores and self.random.random() < 0.40:  # 40% chance to glide away
                # Glide 5 cells away in a random direction
                glide_distance = 5
                possible_positions = self.model.grid.get_neighborhood(
                    self.pos, moore=True, include_center=False, radius=glide_distance
                )
                walkable = [pos for pos in possible_positions if self.is_walkable(pos)]
                if walkable:
                    glide_target = self.random.choice(walkable)
                    self.model.grid.move_agent(self, glide_target)
                    if self.random.random() < 0.3:  # 30% chance to log
                        self.model.log_event("INFO", f"🦅 Archeopteryx glided to safety at {glide_target}!")
                    return  # Skip normal movement

        # Normal herbivore behavior
        super().step()

    def reproduce(self):
        # Archeopteryx reproduces easily (but still has cooldown to prevent spam)
        if self.energy > self.reproduce_threshold:
            self.energy /= 2
            self.reproduction_cooldown = 50  # Short cooldown for small bird-like dinosaur
            baby = self.__class__(self.model)
            self.model.grid.place_agent(baby, self.pos)
            self.model.log_event("BIRTH", f"🦅 {self.species_name} born at {self.pos}")


# ============= MODEL CLASS =============

class IslandModel(Model):
    """The island ecosystem with environmental variables"""
    def __init__(self, width=50, height=50,
                 num_herbivores=20, num_carnivores=10,  # INCREASED from 5 to 10
                 temperature=25, rainfall=100, use_learning_agents=False, use_ppo_agents=False,
                 terrain_map=None):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        # Note: Mesa 3.x doesn't need a separate scheduler
        # Agents are automatically tracked in self.agents

        # Terrain system
        self.terrain_map = terrain_map

        # Environmental variables
        self.temperature = temperature  # Celsius
        self.rainfall = rainfall  # mm per month

        # Temperature affects metabolism (higher temp = more energy used)
        self.metabolism_multiplier = 1 + (self.temperature - 25) / 50

        # Rainfall affects grass growth (more rain = faster growth)
        self.grass_growth_rate = rainfall / 100

        # Learning agent settings
        self.use_learning_agents = use_learning_agents
        self.use_ppo_agents = use_ppo_agents

        # Event log for visualization
        self.event_log = []
        self.max_log_size = 50  # Keep last 50 events

        # Add data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Herbivores": lambda m: len([a for a in m.agents
                                            if isinstance(a, HerbivoreAgent)]),
                "Carnivores": lambda m: len([a for a in m.agents
                                            if isinstance(a, CarnivoreAgent)]),
                "Triceratops": lambda m: len([a for a in m.agents
                                             if isinstance(a, Triceratops)]),
                "Gallimimus": lambda m: len([a for a in m.agents
                                            if isinstance(a, Gallimimus)]),
                "Stegosaurus": lambda m: len([a for a in m.agents
                                             if isinstance(a, Stegosaurus)]),
                "Pachycephalosaurus": lambda m: len([a for a in m.agents
                                                    if isinstance(a, Pachycephalosaurus)]),
                "Brachiosaurus": lambda m: len([a for a in m.agents
                                               if isinstance(a, Brachiosaurus)]),
                "Archeopteryx": lambda m: len([a for a in m.agents
                                              if isinstance(a, Archeopteryx)]),
                "TRex": lambda m: len([a for a in m.agents
                                     if isinstance(a, TRex)]),
                "Velociraptor": lambda m: len([a for a in m.agents
                                             if isinstance(a, Velociraptor)]),
                "Grass": lambda m: len([a for a in m.agents
                                       if isinstance(a, GrassAgent) and a.energy > 0]),
                "Temperature": lambda m: m.temperature,
                "Rainfall": lambda m: m.rainfall,
                "Events": lambda m: len(m.event_log),  # Track event count
            }
        )

        # Add grass everywhere (only if no terrain map provided)
        if terrain_map is None:
            for x in range(width):
                for y in range(height):
                    grass = GrassAgent(self)
                    self.grid.place_agent(grass, (x, y))
        else:
            # With terrain, add grass only on land tiles
            from terrain_generator import TerrainType, TERRAIN_WALKABLE
            for x in range(width):
                for y in range(height):
                    terrain_type = TerrainType(terrain_map[y, x])
                    # Add grass on walkable land (not ocean/river)
                    if TERRAIN_WALKABLE[terrain_type] and terrain_type not in [TerrainType.BEACH, TerrainType.SAND]:
                        grass = GrassAgent(self)
                        self.grid.place_agent(grass, (x, y))

        # Get valid spawn locations for dinosaurs
        if terrain_map is not None:
            from terrain_generator import TerrainType
            # Find valid spawn locations (grassland, forest areas)
            valid_spawns = []
            for x in range(width):
                for y in range(height):
                    terrain_type = TerrainType(terrain_map[y, x])
                    if terrain_type in [TerrainType.GRASSLAND, TerrainType.FOREST]:
                        valid_spawns.append((x, y))
        else:
            # No terrain - any location is valid
            valid_spawns = [(x, y) for x in range(width) for y in range(height)]

        # Add herbivores (mix of all species)
        if self.use_ppo_agents:
            # Import PPO agents (requires stable_baselines3)
            try:
                from ppo_agents import PPOTriceratops, PPOGallimimus
                herbivore_species = [PPOTriceratops, PPOGallimimus]
            except ImportError as e:
                print(f"⚠️  Cannot load PPO agents: {e}")
                print("   Install with: pip install stable-baselines3")
                print("   Falling back to traditional agents...")
                self.use_ppo_agents = False
                herbivore_species = [
                    Triceratops, Gallimimus, Stegosaurus,
                    Pachycephalosaurus, Brachiosaurus, Archeopteryx
                ]
        elif self.use_learning_agents:
            # Import Q-learning agents
            from learning_agents import LearningTriceratops, LearningGallimimus
            herbivore_species = [LearningTriceratops, LearningGallimimus]
        else:
            # All herbivore species available in traditional mode
            herbivore_species = [
                Triceratops, Gallimimus, Stegosaurus,
                Pachycephalosaurus, Brachiosaurus, Archeopteryx
            ]

        for i in range(num_herbivores):
            species_class = self.random.choice(herbivore_species)
            herbivore = species_class(self)
            spawn_loc = self.random.choice(valid_spawns)
            self.grid.place_agent(herbivore, spawn_loc)

        # Add carnivores (mix of TRex and Velociraptor)
        if self.use_ppo_agents:
            # Import PPO agents (requires stable_baselines3)
            try:
                from ppo_agents import PPOTRex, PPOVelociraptor
                carnivore_species = [PPOTRex, PPOVelociraptor]
            except ImportError as e:
                print(f"⚠️  Cannot load PPO carnivore agents: {e}")
                print("   Falling back to traditional carnivores...")
                carnivore_species = [TRex, Velociraptor]
        elif self.use_learning_agents:
            from learning_agents import LearningTRex, LearningVelociraptor
            carnivore_species = [LearningTRex, LearningVelociraptor]
        else:
            carnivore_species = [TRex, Velociraptor]

        for i in range(num_carnivores):
            species_class = self.random.choice(carnivore_species)
            carnivore = species_class(self)
            spawn_loc = self.random.choice(valid_spawns)
            self.grid.place_agent(carnivore, spawn_loc)
    
    def log_event(self, event_type, message):
        """Add an event to the log with step number"""
        self.event_log.append(f"Step {self.steps}: {event_type} - {message}")
        # Keep only the most recent events
        if len(self.event_log) > self.max_log_size:
            self.event_log.pop(0)

    def step(self):
        # Log major events each step
        if self.steps % 5 == 0 and self.steps > 0:  # Every 5 steps instead of 10
            h_count = len([a for a in self.agents if isinstance(a, HerbivoreAgent)])
            c_count = len([a for a in self.agents if isinstance(a, CarnivoreAgent)])
            g_count = len([a for a in self.agents if isinstance(a, GrassAgent) and a.energy > 0])
            self.log_event("INFO", f"📊 Pop: {h_count}H {c_count}C {g_count}G")

        self.datacollector.collect(self)
        # In Mesa 3.x, use shuffle_do to activate agents in random order
        self.agents.shuffle_do("step")


# ============= TESTING / EXPERIMENTS =============

if __name__ == "__main__":
    print("🦕 Testing Environmental Changes...")
    print("=" * 60)
    
    # Experiment 1: Normal conditions
    print("\n📊 Experiment 1: Normal Conditions (25°C, 100mm rain)")
    model = IslandModel(temperature=25, rainfall=100)
    
    for i in range(100):
        model.step()
    
    print(f"After 100 steps:")
    print(f"  Herbivores: {len([a for a in model.agents if isinstance(a, HerbivoreAgent)])}")
    print(f"  Carnivores: {len([a for a in model.agents if isinstance(a, CarnivoreAgent)])}")
    
    # NOW SIMULATE A HEAT WAVE!
    print("\n🔥 HEAT WAVE STRIKES! Temperature rising to 35°C...")
    model.temperature = 35
    model.metabolism_multiplier = 1 + (35 - 25) / 50  # Recalculate
    
    for i in range(100):
        model.step()
    
    print(f"\nAfter heat wave (200 steps total):")
    print(f"  Herbivores: {len([a for a in model.agents if isinstance(a, HerbivoreAgent)])}")
    print(f"  Carnivores: {len([a for a in model.agents if isinstance(a, CarnivoreAgent)])}")
    
    # Get all the data
    data = model.datacollector.get_model_vars_dataframe()
    
    # Plot the results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Population plot
    axes[0].plot(data.index, data['Herbivores'], label='Herbivores', color='blue', linewidth=2)
    axes[0].plot(data.index, data['Carnivores'], label='Carnivores', color='red', linewidth=2)
    axes[0].axvline(x=100, color='orange', linestyle='--', label='Heat Wave Starts')
    axes[0].set_ylabel('Population')
    axes[0].set_title('Population During Heat Wave Event')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Temperature plot
    axes[1].plot(data.index, data['Temperature'], color='orange', linewidth=2)
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Temperature Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_wave_experiment.png')
    print("\n📊 Graph saved as 'heat_wave_experiment.png'")
    plt.close()
    
    # ============= MORE EXPERIMENTS =============
    
    print("\n" + "=" * 60)
    print("🌧️ Experiment 2: Drought Conditions")
    
    drought_model = IslandModel(temperature=30, rainfall=30)
    
    for i in range(200):
        drought_model.step()
    
    drought_data = drought_model.datacollector.get_model_vars_dataframe()
    
    print(f"Final populations in drought:")
    print(f"  Herbivores: {drought_data['Herbivores'].iloc[-1]}")
    print(f"  Carnivores: {drought_data['Carnivores'].iloc[-1]}")
    print(f"  Grass: {drought_data['Grass'].iloc[-1]}")
    
    # Compare scenarios
    print("\n" + "=" * 60)
    print("📊 Comparing Different Environmental Scenarios...")
    
    scenarios = {
        "Ideal": {"temperature": 25, "rainfall": 100},
        "Hot": {"temperature": 35, "rainfall": 100},
        "Cold": {"temperature": 15, "rainfall": 100},
        "Drought": {"temperature": 25, "rainfall": 30},
        "Flood": {"temperature": 25, "rainfall": 200}
    }
    
    results = {}
    
    for name, params in scenarios.items():
        print(f"  Running {name} scenario...")
        model = IslandModel(**params)
        for i in range(200):
            model.step()
        data = model.datacollector.get_model_vars_dataframe()
        results[name] = data
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        axes[idx].plot(data['Herbivores'], label='Herbivores', color='blue')
        axes[idx].plot(data['Carnivores'], label='Carnivores', color='red')
        axes[idx].set_title(name)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('scenario_comparison.png')
    print("\n📊 Comparison saved as 'scenario_comparison.png'")
    plt.close()
    
    print("\n" + "=" * 60)
    print("✅ All experiments complete!")