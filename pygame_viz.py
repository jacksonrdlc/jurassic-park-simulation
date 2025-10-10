#!/usr/bin/env python3
"""
Pygame Visualization for Jurassic Park Simulation
Runs at 60 FPS with ZERO flickering - smooth as butter!

Controls:
  SPACE - Pause/Play
  R - Reset simulation
  UP/DOWN - Adjust speed
  ESC - Quit
"""

import pygame
import sys
from my_first_island import (IslandModel, GrassAgent, HerbivoreAgent, CarnivoreAgent,
                              Triceratops, Gallimimus, TRex, Velociraptor)

# Jurassic Park Color Scheme
COLORS = {
    # Environment
    'background': (15, 25, 15),        # Deep jungle green
    'grass_full': (45, 85, 35),        # Rich grass
    'grass_eaten': (75, 60, 40),       # Dry earth
    'grid_lines': (80, 100, 50),       # Subtle grid overlay

    # Jurassic Park Branding
    'jp_yellow': (255, 210, 0),        # Iconic JP yellow
    'jp_red': (220, 20, 20),           # Danger red
    'jp_black': (20, 20, 20),          # Deep black
    'warning_stripe': (255, 200, 0),   # Warning yellow

    # Dinosaurs - Species-specific colors
    'triceratops': (85, 160, 70),      # Forest green herbivore
    'gallimimus': (200, 180, 90),      # Tan/beige fast herbivore
    'trex': (140, 40, 30),             # Blood red apex predator
    'velociraptor': (100, 60, 120),    # Deep purple pack hunter
    'herbivore': (80, 140, 80),        # Generic herbivore
    'carnivore': (180, 40, 40),        # Generic carnivore

    # UI Elements
    'text': (240, 240, 240),           # Off-white text
    'text_highlight': (255, 210, 0),   # JP yellow for highlights
    'panel': (25, 30, 25),             # Dark panel background
    'panel_border': (255, 210, 0),     # Yellow border
    'containment_active': (50, 200, 80),   # Green = safe
    'containment_warning': (255, 180, 0),  # Yellow = caution
    'containment_breach': (255, 50, 50),   # Red = danger

    # Events
    'event_birth': (80, 220, 100),
    'event_death': (255, 80, 80),
    'event_hunt': (255, 140, 0),
    'event_eat': (120, 200, 120),
    'event_info': (150, 180, 255),

    # Energy bars
    'energy_high': (50, 200, 80),      # Green
    'energy_medium': (255, 180, 0),    # Yellow
    'energy_low': (255, 80, 60),       # Red
    'energy_bg': (40, 40, 40),         # Dark background
}

class JurassicParkViz:
    def __init__(self, width=50, height=50, cell_size=20, use_learning_agents=False, use_ppo_agents=False):  # INCREASED from 16 to 20 for better visibility
        pygame.init()

        # Model parameters
        self.model_width = width
        self.model_height = height
        self.cell_size = cell_size
        self.use_learning_agents = use_learning_agents
        self.use_ppo_agents = use_ppo_agents

        # Window dimensions
        self.grid_width = width * cell_size
        self.grid_height = height * cell_size
        self.panel_width = 450
        self.window_width = self.grid_width + self.panel_width
        self.window_height = max(self.grid_height, 800)  # Ensure minimum height

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Jurassic Park Ecosystem")

        # Fonts - Use bold for Jurassic Park aesthetic
        self.font_title = pygame.font.Font(None, 42)      # Large title
        self.font_large = pygame.font.Font(None, 32)      # Headers
        self.font_medium = pygame.font.Font(None, 22)     # Stats
        self.font_small = pygame.font.Font(None, 16)      # Details

        # Simulation state
        self.model = None
        self.running = True
        self.paused = False
        self.step_count = 0
        self.speed = 1  # Steps per second
        self.clock = pygame.time.Clock()

        # Terrain noise for procedural grass variation
        import random
        self.terrain_noise = [[random.randint(0, 20) for _ in range(width)] for _ in range(height)]

        # Reset model
        self.reset_model()

    def reset_model(self):
        """Create a new model"""
        self.model = IslandModel(
            width=self.model_width,
            height=self.model_height,
            num_herbivores=20,
            num_carnivores=5,
            temperature=25,
            rainfall=100,
            use_learning_agents=self.use_learning_agents,
            use_ppo_agents=self.use_ppo_agents
        )
        self.step_count = 0

        # Load trained PPO models if using PPO agents
        if self.use_ppo_agents:
            import os
            if os.path.exists('models_ppo/herbivore_ppo_final.zip') and os.path.exists('models_ppo/carnivore_ppo_final.zip'):
                from ppo_agents import PPOHerbivoreAgent, PPOCarnivoreAgent
                # Load PPO models
                PPOHerbivoreAgent.load_model('models_ppo/herbivore_ppo_final.zip')
                PPOCarnivoreAgent.load_model('models_ppo/carnivore_ppo_final.zip')
            else:
                print("⚠️  PPO models not found! Please train them first with:")
                print("   python train_ppo.py")
        # Load trained models if using Q-learning agents
        elif self.use_learning_agents:
            import os
            if os.path.exists('models/herbivore_final.pth'):
                import torch
                from learning_agents import LearningHerbivoreAgent, LearningCarnivoreAgent, NeuralNetwork
                # Initialize models if not already done
                if LearningHerbivoreAgent.neural_net is None:
                    LearningHerbivoreAgent.neural_net = NeuralNetwork(19, 9)
                if LearningCarnivoreAgent.neural_net is None:
                    LearningCarnivoreAgent.neural_net = NeuralNetwork(19, 9)
                # Load saved weights
                LearningHerbivoreAgent.neural_net.load_state_dict(torch.load('models/herbivore_final.pth'))
                LearningCarnivoreAgent.neural_net.load_state_dict(torch.load('models/carnivore_final.pth'))
                LearningHerbivoreAgent.training_mode = False
                LearningCarnivoreAgent.training_mode = False
                LearningHerbivoreAgent.epsilon = 0.0
                LearningCarnivoreAgent.epsilon = 0.0
                print("✅ Loaded trained PyTorch Q-learning models for visualization")

    def handle_events(self):
        """Handle keyboard/mouse events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_model()
                elif event.key == pygame.K_UP:
                    self.speed = min(60, self.speed + 1)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 1)

    def update(self):
        """Update simulation"""
        if not self.paused:
            self.model.step()
            self.step_count += 1

    def draw_energy_bar(self, rect, energy, max_energy):
        """Draw energy bar above an agent"""
        bar_width = self.cell_size
        bar_height = 4
        bar_x = rect.centerx - bar_width // 2
        bar_y = rect.top - 7

        # Background
        pygame.draw.rect(self.screen, COLORS['energy_bg'],
                        (bar_x, bar_y, bar_width, bar_height))

        # Energy fill
        energy_pct = max(0, min(1, energy / max_energy))
        fill_width = int(bar_width * energy_pct)

        if energy_pct > 0.6:
            energy_color = COLORS['energy_high']
        elif energy_pct > 0.3:
            energy_color = COLORS['energy_medium']
        else:
            energy_color = COLORS['energy_low']

        if fill_width > 0:
            pygame.draw.rect(self.screen, energy_color,
                            (bar_x, bar_y, fill_width, bar_height))

    def draw_directional_sprite(self, rect, color, direction, size_ratio, is_carnivore=False):
        """Draw directional sprite (triangle/polygon pointing in movement direction)"""
        import math

        # Calculate angle from direction
        if direction == (0, 0):
            angle = 0
        else:
            angle = math.atan2(direction[1], direction[0])

        # Create triangle pointing in direction
        radius = int(self.cell_size * size_ratio)
        center_x, center_y = rect.center

        # Triangle points (pointing right initially, then rotated)
        points = [
            (center_x + radius, center_y),  # Front tip
            (center_x - radius//2, center_y - radius//2),  # Back top
            (center_x - radius//2, center_y + radius//2),  # Back bottom
        ]

        # Rotate points around center
        rotated_points = []
        for px, py in points:
            # Translate to origin
            tx = px - center_x
            ty = py - center_y
            # Rotate
            rx = tx * math.cos(angle) - ty * math.sin(angle)
            ry = tx * math.sin(angle) + ty * math.cos(angle)
            # Translate back
            rotated_points.append((center_x + rx, center_y + ry))

        # Draw filled triangle
        pygame.draw.polygon(self.screen, color, rotated_points)

        # Draw outline
        outline_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.polygon(self.screen, outline_color, rotated_points, 2)

    def draw_ai_indicator(self, rect, species_name):
        """Draw species-specific symbol for AI agents in top-left corner"""
        # Map species to colored symbols/letters
        symbol_map = {
            'TRex': ('T', (255, 100, 100)),        # Red T for T-Rex
            'Velociraptor': ('V', (200, 150, 255)),  # Purple V for Velociraptor
            'Triceratops': ('3', (150, 255, 150)),   # Green 3 for Triceratops
            'Gallimimus': ('G', (255, 255, 150)),    # Yellow G for Gallimimus
            'CarnivoreAgent': ('C', (255, 100, 100)),
            'HerbivoreAgent': ('H', (150, 255, 150))
        }

        symbol, color = symbol_map.get(species_name, ('A', (255, 255, 255)))

        # Draw a small circle background
        badge_center = (rect.left + int(self.cell_size * 0.2), rect.top + int(self.cell_size * 0.2))
        badge_radius = int(self.cell_size * 0.18)

        # Draw circle background (slightly transparent dark circle)
        circle_surface = pygame.Surface((badge_radius * 2 + 2, badge_radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (0, 0, 0, 180), (badge_radius + 1, badge_radius + 1), badge_radius)
        pygame.draw.circle(circle_surface, color, (badge_radius + 1, badge_radius + 1), badge_radius, 2)  # Colored outline
        self.screen.blit(circle_surface, (badge_center[0] - badge_radius - 1, badge_center[1] - badge_radius - 1))

        # Draw the symbol/letter
        symbol_font = pygame.font.Font(None, int(self.cell_size * 0.8))
        symbol_surface = symbol_font.render(symbol, True, color)
        symbol_rect = symbol_surface.get_rect(center=badge_center)
        self.screen.blit(symbol_surface, symbol_rect)

    def draw_movement_trail(self, agent):
        """Draw fading trail showing recent movement"""
        if not hasattr(agent, 'movement_history') or not agent.movement_history:
            return

        # Draw trail with fading alpha
        for i, old_pos in enumerate(agent.movement_history):
            alpha = int(100 * (i + 1) / len(agent.movement_history))  # Fade from 33 to 100
            trail_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)

            # Get agent color
            if isinstance(agent, Triceratops):
                color = COLORS['triceratops']
            elif isinstance(agent, Gallimimus):
                color = COLORS['gallimimus']
            elif isinstance(agent, TRex):
                color = COLORS['trex']
            elif isinstance(agent, Velociraptor):
                color = COLORS['velociraptor']
            else:
                color = (128, 128, 128)

            trail_color = (*color, alpha)
            pygame.draw.circle(trail_surface, trail_color,
                             (self.cell_size // 2, self.cell_size // 2),
                             self.cell_size // 4)

            self.screen.blit(trail_surface,
                           (old_pos[0] * self.cell_size, old_pos[1] * self.cell_size))

    def draw_grid(self):
        """Draw the ecosystem grid"""
        # Draw subtle grid overlay
        for x in range(0, self.grid_width, self.cell_size * 5):
            pygame.draw.line(self.screen, COLORS['grid_lines'],
                           (x, 0), (x, self.grid_height), 1)
        for y in range(0, self.grid_height, self.cell_size * 5):
            pygame.draw.line(self.screen, COLORS['grid_lines'],
                           (0, y), (self.grid_width, y), 1)

        # Draw grid coordinates (A-Z, 1-N)
        coord_font = pygame.font.Font(None, 14)
        # X-axis labels (A, B, C...)
        for i in range(0, self.model.grid.width, 5):
            label = chr(65 + (i // 5) % 26)  # A-Z
            text = coord_font.render(label, True, COLORS['text_highlight'])
            self.screen.blit(text, (i * self.cell_size + 2, 2))

        # Y-axis labels (1, 2, 3...)
        for i in range(0, self.model.grid.height, 5):
            label = str((i // 5) + 1)
            text = coord_font.render(label, True, COLORS['text_highlight'])
            self.screen.blit(text, (2, i * self.cell_size + 2))

        # Draw cells
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                # Get cell contents
                cell_contents = self.model.grid.get_cell_list_contents([(x, y)])

                # Determine what to draw (priority: carnivore > herbivore > grass)
                color = COLORS['background']
                is_learning_agent = False
                drawn_agent = None

                for agent in cell_contents:
                    # Check if it's a learning agent
                    if hasattr(agent, '__class__'):
                        is_learning_agent = 'Learning' in agent.__class__.__name__

                    if isinstance(agent, TRex):
                        color = COLORS['trex']
                        # Draw movement trail first (behind sprite)
                        self.draw_movement_trail(agent)
                        # Draw directional sprite
                        direction = getattr(agent, 'direction', (1, 0))
                        self.draw_directional_sprite(rect, color, direction, 0.45, is_carnivore=True)
                        # Draw AI indicator (emoji)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'TRex')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Velociraptor):
                        color = COLORS['velociraptor']
                        self.draw_movement_trail(agent)
                        direction = getattr(agent, 'direction', (1, 0))
                        self.draw_directional_sprite(rect, color, direction, 0.4, is_carnivore=True)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Velociraptor')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Triceratops):
                        color = COLORS['triceratops']
                        self.draw_movement_trail(agent)
                        direction = getattr(agent, 'direction', (1, 0))
                        self.draw_directional_sprite(rect, color, direction, 0.35, is_carnivore=False)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Triceratops')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Gallimimus):
                        color = COLORS['gallimimus']
                        self.draw_movement_trail(agent)
                        direction = getattr(agent, 'direction', (1, 0))
                        self.draw_directional_sprite(rect, color, direction, 0.3, is_carnivore=False)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Gallimimus')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, CarnivoreAgent):
                        color = COLORS['carnivore']
                        self.draw_movement_trail(agent)
                        direction = getattr(agent, 'direction', (1, 0))
                        self.draw_directional_sprite(rect, color, direction, 0.4, is_carnivore=True)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'CarnivoreAgent')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, HerbivoreAgent):
                        color = COLORS['herbivore']
                        self.draw_movement_trail(agent)
                        direction = getattr(agent, 'direction', (1, 0))
                        self.draw_directional_sprite(rect, color, direction, 0.3, is_carnivore=False)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'HerbivoreAgent')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, GrassAgent):
                        # Add procedural variation to grass color
                        base_color = COLORS['grass_full'] if agent.energy > 0 else COLORS['grass_eaten']
                        noise = self.terrain_noise[y][x]
                        varied_color = tuple(max(0, min(255, c + noise - 10)) for c in base_color)
                        pygame.draw.rect(self.screen, varied_color, rect)

                # Draw energy bar for dinosaurs
                if drawn_agent and hasattr(drawn_agent, 'energy'):
                    if isinstance(drawn_agent, (HerbivoreAgent, CarnivoreAgent)):
                        max_energy = 150 if isinstance(drawn_agent, CarnivoreAgent) else 100
                        self.draw_energy_bar(rect, drawn_agent.energy, max_energy)

    def draw_panel(self):
        """Draw info panel on the right"""
        panel_x = self.grid_width

        # Background
        pygame.draw.rect(
            self.screen,
            COLORS['panel'],
            (panel_x, 0, self.panel_width, self.window_height)
        )

        # Yellow border (Jurassic Park aesthetic)
        pygame.draw.rect(
            self.screen,
            COLORS['panel_border'],
            (panel_x, 0, self.panel_width, self.window_height),
            3  # Border width
        )

        y_offset = 15

        # === JURASSIC PARK BRANDING ===
        # Main title
        title = self.font_title.render("JURASSIC PARK", True, COLORS['text_highlight'])
        title_rect = title.get_rect(center=(panel_x + self.panel_width // 2, y_offset + 20))
        self.screen.blit(title, title_rect)
        y_offset += 50

        # Subtitle
        subtitle = self.font_small.render("ISLA NUBLAR - SECTOR 7G", True, COLORS['text'])
        subtitle_rect = subtitle.get_rect(center=(panel_x + self.panel_width // 2, y_offset))
        self.screen.blit(subtitle, subtitle_rect)
        y_offset += 25

        # Warning stripe separator
        for i in range(0, self.panel_width, 40):
            stripe_color = COLORS['jp_yellow'] if (i // 40) % 2 == 0 else COLORS['jp_black']
            pygame.draw.rect(self.screen, stripe_color, (panel_x + i, y_offset, 40, 8))
        y_offset += 18

        # === CONTAINMENT STATUS ===
        # Calculate containment status based on population
        tri_count = len([a for a in self.model.agents if isinstance(a, Triceratops)])
        gal_count = len([a for a in self.model.agents if isinstance(a, Gallimimus)])
        trex_count = len([a for a in self.model.agents if isinstance(a, TRex)])
        vel_count = len([a for a in self.model.agents if isinstance(a, Velociraptor)])
        total_dinos = tri_count + gal_count + trex_count + vel_count

        if total_dinos > 40:
            status_text = "⚠ OVERPOPULATION"
            status_color = COLORS['containment_warning']
        elif total_dinos < 10:
            status_text = "⚠ CRITICAL - LOW POP"
            status_color = COLORS['containment_breach']
        else:
            status_text = "✓ CONTAINMENT ACTIVE"
            status_color = COLORS['containment_active']

        status = self.font_medium.render(status_text, True, status_color)
        status_rect = status.get_rect(center=(panel_x + self.panel_width // 2, y_offset))
        self.screen.blit(status, status_rect)
        y_offset += 35

        # Separator
        pygame.draw.line(self.screen, COLORS['jp_yellow'],
                        (panel_x + 20, y_offset),
                        (panel_x + self.panel_width - 20, y_offset), 2)
        y_offset += 20

        # === SYSTEM INFO ===
        system_header = self.font_medium.render("SYSTEM STATUS", True, COLORS['text_highlight'])
        self.screen.blit(system_header, (panel_x + 20, y_offset))
        y_offset += 30

        g_count = len([a for a in self.model.agents if isinstance(a, GrassAgent) and a.energy > 0])
        agent_type = "AI (LEARNING)" if self.use_learning_agents else "TRADITIONAL"

        system_stats = [
            f"Step: {self.step_count:,}",
            f"Speed: {self.speed}x",
            f"Mode: {agent_type}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
        ]

        for stat in system_stats:
            text = self.font_small.render(stat, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 25, y_offset))
            y_offset += 20

        y_offset += 10

        # === POPULATION MONITORING ===
        pop_header = self.font_medium.render("POPULATION", True, COLORS['text_highlight'])
        self.screen.blit(pop_header, (panel_x + 20, y_offset))
        y_offset += 30

        # Herbivores section
        herb_label = self.font_small.render("HERBIVORES:", True, COLORS['text'])
        self.screen.blit(herb_label, (panel_x + 25, y_offset))
        y_offset += 22

        # Draw colored indicators + counts
        pygame.draw.circle(self.screen, COLORS['triceratops'], (panel_x + 35, y_offset + 6), 5)
        tri_text = self.font_small.render(f"Triceratops: {tri_count}", True, COLORS['text'])
        self.screen.blit(tri_text, (panel_x + 50, y_offset))
        y_offset += 20

        pygame.draw.circle(self.screen, COLORS['gallimimus'], (panel_x + 35, y_offset + 6), 5)
        gal_text = self.font_small.render(f"Gallimimus: {gal_count}", True, COLORS['text'])
        self.screen.blit(gal_text, (panel_x + 50, y_offset))
        y_offset += 25

        # Carnivores section
        carn_label = self.font_small.render("CARNIVORES:", True, COLORS['text'])
        self.screen.blit(carn_label, (panel_x + 25, y_offset))
        y_offset += 22

        pygame.draw.circle(self.screen, COLORS['trex'], (panel_x + 35, y_offset + 6), 5)
        trex_text = self.font_small.render(f"T-Rex: {trex_count}", True, COLORS['text'])
        self.screen.blit(trex_text, (panel_x + 50, y_offset))
        y_offset += 20

        pygame.draw.circle(self.screen, COLORS['velociraptor'], (panel_x + 35, y_offset + 6), 5)
        vel_text = self.font_small.render(f"Velociraptor: {vel_count}", True, COLORS['text'])
        self.screen.blit(vel_text, (panel_x + 50, y_offset))
        y_offset += 25

        # Environment
        env_label = self.font_small.render("ENVIRONMENT:", True, COLORS['text'])
        self.screen.blit(env_label, (panel_x + 25, y_offset))
        y_offset += 22

        pygame.draw.rect(self.screen, COLORS['grass_full'], (panel_x + 30, y_offset + 3, 10, 10))
        grass_text = self.font_small.render(f"Vegetation: {g_count}", True, COLORS['text'])
        self.screen.blit(grass_text, (panel_x + 50, y_offset))
        y_offset += 20

        temp_text = self.font_small.render(f"Temp: {self.model.temperature}°C", True, COLORS['text'])
        self.screen.blit(temp_text, (panel_x + 50, y_offset))
        y_offset += 18

        rain_text = self.font_small.render(f"Rainfall: {self.model.rainfall}mm", True, COLORS['text'])
        self.screen.blit(rain_text, (panel_x + 50, y_offset))
        y_offset += 20

        # Separator
        y_offset += 10
        pygame.draw.line(self.screen, COLORS['text'],
                        (panel_x + 20, y_offset),
                        (panel_x + self.panel_width - 20, y_offset), 1)
        y_offset += 20

        # Event log title
        log_title = self.font_medium.render("EVENT LOG", True, COLORS['text'])
        self.screen.blit(log_title, (panel_x + 20, y_offset))
        y_offset += 30

        # Calculate max space for events (leave room for controls)
        max_log_y = self.window_height - 180

        # Show events
        if hasattr(self.model, 'event_log'):
            events = list(reversed(self.model.event_log[-25:]))

            for event in events:
                if y_offset > max_log_y:
                    break

                # Determine color and prefix
                if "BIRTH" in event:
                    color = COLORS['event_birth']
                    prefix = "[+]"
                elif "DEATH" in event:
                    color = COLORS['event_death']
                    prefix = "[-]"
                elif "HUNT" in event:
                    color = COLORS['event_hunt']
                    prefix = "[!]"
                elif "EAT" in event:
                    color = COLORS['event_eat']
                    prefix = "[*]"
                elif "INFO" in event:
                    color = COLORS['event_info']
                    prefix = "[i]"
                else:
                    color = COLORS['text']
                    prefix = "   "

                # Remove emoji characters
                event = event.replace('🦕', '').replace('🦖', '').replace('🌿', '')
                event = event.replace('💀', '').replace('📊', '').replace('🟢', '')

                # Truncate long events
                if len(event) > 38:
                    event = event[:35] + "..."

                text = self.font_small.render(f"{prefix} {event}", True, color)
                self.screen.blit(text, (panel_x + 15, y_offset))
                y_offset += 17

        # Separator before controls
        y_offset = self.window_height - 160
        pygame.draw.line(self.screen, COLORS['text'],
                        (panel_x + 20, y_offset),
                        (panel_x + self.panel_width - 20, y_offset), 1)
        y_offset += 20

        # Controls section (fixed at bottom)
        controls_title = self.font_medium.render("CONTROLS", True, COLORS['text'])
        self.screen.blit(controls_title, (panel_x + 20, y_offset))
        y_offset += 28

        controls = [
            "SPACE - Pause/Play",
            "R     - Reset",
            "UP/DN - Speed +/-",
            "ESC   - Quit",
        ]

        for control in controls:
            text = self.font_small.render(control, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 20, y_offset))
            y_offset += 20

    def draw_legend(self):
        """Draw legend in top-left corner"""
        legend_x = 10
        legend_y = 10

        # Semi-transparent background (larger for more species)
        legend_height = 190 if self.use_learning_agents else 170
        legend_surface = pygame.Surface((200, legend_height))
        legend_surface.set_alpha(220)
        legend_surface.fill(COLORS['panel'])
        self.screen.blit(legend_surface, (legend_x, legend_y))

        # Title
        title_text = self.font_small.render("LEGEND", True, COLORS['text'])
        self.screen.blit(title_text, (legend_x + 10, legend_y + 8))

        legend_items = [
            ("Grass (full)", COLORS['grass_full'], 'rect'),
            ("Grass (eaten)", COLORS['grass_eaten'], 'rect'),
            ("Triceratops", COLORS['triceratops'], 'circle'),
            ("Gallimimus", COLORS['gallimimus'], 'circle'),
            ("T-Rex", COLORS['trex'], 'circle'),
            ("Velociraptor", COLORS['velociraptor'], 'circle'),
        ]

        if self.use_learning_agents:
            legend_items.append(("AI Agent (ring)", (255, 255, 255), 'ring'))

        y = legend_y + 30
        for label, color, shape in legend_items:
            if shape == 'circle':
                pygame.draw.circle(self.screen, color, (legend_x + 15, y + 8), 6)
            elif shape == 'ring':
                pygame.draw.circle(self.screen, (100, 100, 100), (legend_x + 15, y + 8), 6)  # Inner circle
                pygame.draw.circle(self.screen, color, (legend_x + 15, y + 8), 6, 2)  # White ring
            else:
                pygame.draw.rect(self.screen, color, (legend_x + 10, y + 3, 12, 12))

            text = self.font_small.render(label, True, COLORS['text'])
            self.screen.blit(text, (legend_x + 35, y))
            y += 20

    def draw_environmental_effects(self):
        """Draw weather effects (rain, heat)"""
        if not self.model:
            return

        # Rain effect when rainfall is high
        if self.model.rainfall > 120:
            import random
            rain_surface = pygame.Surface((self.grid_width, self.grid_height), pygame.SRCALPHA)
            # Draw rain drops
            for _ in range(int(self.model.rainfall / 3)):
                x = random.randint(0, self.grid_width)
                y = random.randint(0, self.grid_height)
                pygame.draw.line(rain_surface, (100, 150, 200, 100), (x, y), (x + 2, y + 5), 1)
            self.screen.blit(rain_surface, (0, 0))

        # Heat shimmer when temperature is high
        if self.model.temperature > 32:
            heat_surface = pygame.Surface((self.grid_width, self.grid_height), pygame.SRCALPHA)
            heat_alpha = min(80, int((self.model.temperature - 32) * 5))
            heat_surface.fill((255, 100, 50, heat_alpha))
            self.screen.blit(heat_surface, (0, 0))

    def run(self):
        """Main game loop - runs at 60 FPS!"""
        frame_count = 0

        while self.running:
            self.handle_events()

            # Update simulation at specified speed
            if frame_count % max(1, 60 // self.speed) == 0:
                self.update()

            # Draw everything
            self.screen.fill(COLORS['background'])
            self.draw_grid()
            self.draw_environmental_effects()  # Add weather overlays
            self.draw_panel()
            self.draw_legend()

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS - smooth as butter!
            frame_count += 1

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    import sys

    # Check for agent type flags
    use_ppo = '--ppo' in sys.argv
    use_learning = ('--learning-agents' in sys.argv or '--ai' in sys.argv) and not use_ppo

    print("Starting Jurassic Park Simulation...")
    if use_ppo:
        print("Mode: PPO Agents (Proximal Policy Optimization) 🚀")
        print("  State-of-the-art reinforcement learning!")
    elif use_learning:
        print("Mode: Q-Learning Agents 🧠")
        print("  Neural network Q-learning")
    else:
        print("Mode: Traditional Rule-Based Agents")
        print("  (use --ppo for PPO agents, --ai for Q-learning agents)")
    print("Controls: SPACE=pause, R=reset, UP/DOWN=speed, ESC=quit")
    print()

    viz = JurassicParkViz(width=50, height=50, cell_size=24,
                          use_learning_agents=use_learning,
                          use_ppo_agents=use_ppo)
    viz.run()
