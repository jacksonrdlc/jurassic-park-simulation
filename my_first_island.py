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

    def step(self):
        # Move (speed determines how many cells to move)
        old_pos = self.pos
        current_pos = self.pos
        for _ in range(self.speed):
            possible_steps = self.model.grid.get_neighborhood(
                current_pos, moore=True, include_center=False
            )
            new_position = self.random.choice(possible_steps)
            current_pos = new_position

        # Track movement history for trails
        self.movement_history.append(old_pos)
        if len(self.movement_history) > 3:  # Keep last 3 positions
            self.movement_history.pop(0)

        # Calculate direction based on movement
        if current_pos != old_pos:
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
        
        # Die if out of energy
        if self.energy <= 0:
            self.model.log_event("DEATH", f"💀 {self.species_name} starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()  # Mesa 3.x: removes from model.agents
        
        # Reproduce if enough energy
        if self.energy > 100 and self.random.random() < 0.05:
            self.reproduce()
    
    def reproduce(self):
        self.energy /= 2
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

    def step(self):
        # Move (speed determines how many cells to move)
        old_pos = self.pos
        current_pos = self.pos
        for _ in range(self.speed):
            possible_steps = self.model.grid.get_neighborhood(
                current_pos, moore=True, include_center=False
            )
            new_position = self.random.choice(possible_steps)
            current_pos = new_position

        # Track movement history for trails
        self.movement_history.append(old_pos)
        if len(self.movement_history) > 3:  # Keep last 3 positions
            self.movement_history.pop(0)

        # Calculate direction based on movement
        if current_pos != old_pos:
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
        
        # Die
        if self.energy <= 0:
            self.model.log_event("DEATH", f"💀 {self.species_name} starved at {self.pos}")
            self.model.grid.remove_agent(self)
            self.remove()  # Mesa 3.x: removes from model.agents
        
        # Reproduce - REDUCED threshold from 150 to 120 for easier breeding
        if self.energy > 120 and self.random.random() < 0.03:
            self.reproduce()
    
    def reproduce(self):
        self.energy /= 2
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


# ============= MODEL CLASS =============

class IslandModel(Model):
    """The island ecosystem with environmental variables"""
    def __init__(self, width=50, height=50,
                 num_herbivores=20, num_carnivores=10,  # INCREASED from 5 to 10
                 temperature=25, rainfall=100, use_learning_agents=False):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        # Note: Mesa 3.x doesn't need a separate scheduler
        # Agents are automatically tracked in self.agents

        # Environmental variables
        self.temperature = temperature  # Celsius
        self.rainfall = rainfall  # mm per month

        # Temperature affects metabolism (higher temp = more energy used)
        self.metabolism_multiplier = 1 + (self.temperature - 25) / 50

        # Rainfall affects grass growth (more rain = faster growth)
        self.grass_growth_rate = rainfall / 100

        # Learning agent settings
        self.use_learning_agents = use_learning_agents

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
        
        # Add grass everywhere
        for x in range(width):
            for y in range(height):
                grass = GrassAgent(self)
                self.grid.place_agent(grass, (x, y))

        # Add herbivores (mix of Triceratops and Gallimimus)
        if self.use_learning_agents:
            # Import learning agents only when needed
            from learning_agents import LearningTriceratops, LearningGallimimus
            herbivore_species = [LearningTriceratops, LearningGallimimus]
        else:
            herbivore_species = [Triceratops, Gallimimus]

        for i in range(num_herbivores):
            species_class = self.random.choice(herbivore_species)
            herbivore = species_class(self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(herbivore, (x, y))

        # Add carnivores (mix of TRex and Velociraptor)
        if self.use_learning_agents:
            from learning_agents import LearningTRex, LearningVelociraptor
            carnivore_species = [LearningTRex, LearningVelociraptor]
        else:
            carnivore_species = [TRex, Velociraptor]

        for i in range(num_carnivores):
            species_class = self.random.choice(carnivore_species)
            carnivore = species_class(self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(carnivore, (x, y))
    
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