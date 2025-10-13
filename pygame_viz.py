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
                              Triceratops, Gallimimus, TRex, Velociraptor,
                              Stegosaurus, Pachycephalosaurus, Brachiosaurus, Archeopteryx)
from sprite_sheet import DinosaurSprite

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
    'stegosaurus': (120, 140, 60),     # Olive green plated herbivore
    'pachycephalosaurus': (180, 120, 90),  # Brown dome-headed herbivore
    'brachiosaurus': (100, 180, 140),  # Teal/cyan massive sauropod
    'archeopteryx': (140, 180, 200),   # Light blue-gray proto-bird
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
    def __init__(self, width=625, height=375, cell_size=16, use_learning_agents=False, use_ppo_agents=False):  # World: 625x375 @ 16px cells
        pygame.init()

        # Model parameters - Costa Rican Island
        self.model_width = width  # 625 cells wide
        self.model_height = height  # 375 cells tall
        self.cell_size = cell_size  # 16 pixels per cell (zoomed in terrain)

        self.use_learning_agents = use_learning_agents
        self.use_ppo_agents = use_ppo_agents

        # Window dimensions - Widescreen viewport for modern monitors
        self.viewport_width = 1600   # ~100 cells visible width at 16px
        self.viewport_height = 900   # ~56 cells visible height at 16px
        self.panel_width = 450
        self.window_width = self.viewport_width + self.panel_width
        self.window_height = self.viewport_height

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Jurassic Park - Costa Rican Island")

        # Camera system for scrolling
        from camera import Camera
        self.camera = Camera(
            self.viewport_width,
            self.viewport_height,
            self.model_width,
            self.model_height,
            self.cell_size
        )

        # Fonts
        self.font_title = pygame.font.Font(None, 42)
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 16)

        # Load dinosaur sprite sheets with proper clipping (SCALED DOWN for better proportion at 16px cells)
        self.dino_sprites = {}
        try:
            # 16x16 sprites (scaled to fit nicely on 16px cells - ~1.5-2 cells wide)
            self.dino_sprites['trex'] = DinosaurSprite('dino_sprites/TyrannosaurusRex_16x16.png', scale=1.8)  # ~29px (1.8 cells)
            self.dino_sprites['triceratops'] = DinosaurSprite('dino_sprites/Triceratops_16x16.png', scale=1.5)  # ~24px (1.5 cells)
            self.dino_sprites['velociraptor'] = DinosaurSprite('dino_sprites/Spinosaurus_16x16.png', scale=1.5)  # ~24px (1.5 cells)
            self.dino_sprites['gallimimus'] = DinosaurSprite('dino_sprites/Parasaurolophus_16x16.png', scale=1.2)  # ~19px (1.2 cells)
            self.dino_sprites['pachycephalosaurus'] = DinosaurSprite('dino_sprites/Pachycephalosaurus_16x16.png', scale=1.3)  # ~21px
            self.dino_sprites['archeopteryx'] = DinosaurSprite('dino_sprites/Archeopteryx_16x16.png', scale=1.0)  # ~16px (1 cell)

            # 32x32 sprites (need different frame size)
            self.dino_sprites['stegosaurus'] = DinosaurSprite('dino_sprites/Stegosaurus_32x32.png', frame_size=32, scale=1.2)  # ~38px (2.4 cells)
            self.dino_sprites['brachiosaurus'] = DinosaurSprite('dino_sprites/Brachiosaurus_32x32.png', frame_size=32, scale=1.5)  # ~48px (3 cells)

            print("✅ Loaded dinosaur sprite sheets with directional frames!")
        except Exception as e:
            print(f"⚠️  Warning: Could not load sprite sheets: {e}")
            self.dino_sprites = None

        # Color tints for each species (earthy colors, not green!)
        self.dino_tints = {
            'trex': (140, 40, 30),             # Blood red/brown apex predator
            'triceratops': (120, 90, 60),      # Earthy tan/brown (not green!)
            'velociraptor': (100, 70, 50),     # Dark brown pack hunter
            'gallimimus': (180, 150, 100),     # Tan/beige fast herbivore
            'stegosaurus': (130, 100, 70),     # Brown plated herbivore
            'pachycephalosaurus': (150, 110, 80),  # Light brown dome-headed
            'brachiosaurus': (160, 140, 100),  # Warm tan massive sauropod
            'archeopteryx': (110, 100, 90),    # Gray-brown proto-bird
        }

        # Simulation state
        self.model = None
        self.running = True
        self.paused = False
        self.step_count = 0
        self.speed = 1
        self.clock = pygame.time.Clock()

        # Time of day for day/night cycle
        self.time_of_day = 0.5  # 0=midnight, 0.5=noon, 1.0=midnight
        self.time_speed = 0.001  # How fast time passes

        # Water animation
        self.water_offset = 0

        # Volcano particles
        self.volcano_particles = []

        # Terrain
        self.terrain_map = None
        self.terrain_generator = None
        self.terrain_noise = None  # Procedural noise for terrain variation

        # Minimap
        self.show_minimap = True
        self.minimap_width = 300  # Proportional minimap for widescreen
        self.minimap_height = 180

        # Reset model
        self.reset_model()

    def reset_model(self):
        """Create a new model with terrain generation"""
        # Generate terrain for Costa Rican island
        print("\n🏝️  Generating Costa Rican island...")
        from terrain_generator import generate_island
        self.terrain_map, self.terrain_generator = generate_island(
            self.model_width,
            self.model_height,
            seed=None  # Random seed each time
        )

        # Generate procedural noise for terrain color variation
        import random
        self.terrain_noise = [[random.randint(-10, 10) for _ in range(self.model_width)]
                             for _ in range(self.model_height)]

        # Create island model with terrain
        # Balanced populations for island size
        self.model = IslandModel(
            width=self.model_width,
            height=self.model_height,
            num_herbivores=125,  # Balanced for 625x375 world
            num_carnivores=50,
            temperature=25,
            rainfall=100,
            use_learning_agents=self.use_learning_agents,
            use_ppo_agents=self.use_ppo_agents,
            terrain_map=self.terrain_map  # Pass terrain
        )
        self.step_count = 0

        # Reset time and effects
        self.time_of_day = 0.5  # Start at noon
        self.water_offset = 0
        self.volcano_particles = []

        # Center camera on island center
        self.camera.center_on(self.model_width / 2, self.model_height / 2)

        # Load trained PPO models if using PPO agents
        if self.use_ppo_agents:
            import os
            if os.path.exists('models_ppo/herbivore_ppo_final.zip') and os.path.exists('models_ppo/carnivore_ppo_final.zip'):
                try:
                    from ppo_agents import PPOHerbivoreAgent, PPOCarnivoreAgent
                    # Load PPO models
                    PPOHerbivoreAgent.load_model('models_ppo/herbivore_ppo_final.zip')
                    PPOCarnivoreAgent.load_model('models_ppo/carnivore_ppo_final.zip')
                    print("✅ Loaded trained PPO models for visualization")
                except ImportError as e:
                    print(f"⚠️  Cannot load PPO model loader: {e}")
                    print("   PPO agents were created but models cannot be loaded")
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
                elif event.key == pygame.K_m:
                    # Toggle minimap
                    self.show_minimap = not self.show_minimap
                # Speed controls removed - arrow keys now pan camera

            # Mouse drag controls for panning
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Only drag if clicking in viewport area (not panel)
                    if event.pos[0] < self.viewport_width:
                        self.camera.start_drag(event.pos)

                elif event.button == 3:  # Right click
                    # Center camera on clicked location
                    world_x, world_y = self.camera.screen_to_world(event.pos[0], event.pos[1])
                    self.camera.center_on(int(world_x), int(world_y))

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.camera.end_drag()

            elif event.type == pygame.MOUSEMOTION:
                if self.camera.dragging:
                    self.camera.update_drag(event.pos)

        # Arrow key panning (continuous)
        keys = pygame.key.get_pressed()
        self.camera.update_keyboard(keys)

    def update(self):
        """Update simulation"""
        if not self.paused:
            self.model.step()
            self.step_count += 1

        # Always update visual effects (even when paused)
        # Update time of day (day/night cycle)
        self.time_of_day = (self.time_of_day + self.time_speed) % 1.0

        # Update water animation
        self.water_offset = (self.water_offset + 0.5) % 20

        # Update volcano particles
        import math
        self.update_volcano_particles()

    def update_volcano_particles(self):
        """Update volcano smoke particles"""
        from terrain_generator import TerrainType

        # Spawn new particles from volcano tiles (with low probability)
        if self.terrain_map is not None and len(self.volcano_particles) < 100:
            import random
            # Find volcano positions
            for world_y in range(self.model_height):
                for world_x in range(self.model_width):
                    terrain_type = TerrainType(self.terrain_map[world_y, world_x])
                    if terrain_type == TerrainType.VOLCANO and random.random() < 0.01:
                        # Check if visible
                        if self.camera.is_visible(world_x, world_y):
                            # Spawn particle
                            self.volcano_particles.append({
                                'x': world_x,
                                'y': world_y,
                                'offset_x': random.uniform(-0.5, 0.5),
                                'offset_y': 0,
                                'vy': -0.05 - random.uniform(0, 0.05),
                                'life': 1.0,
                                'size': random.randint(2, 5)
                            })

        # Update existing particles
        for particle in self.volcano_particles[:]:
            particle['offset_y'] += particle['vy']
            particle['life'] -= 0.01
            if particle['life'] <= 0:
                self.volcano_particles.remove(particle)

    def draw_tree(self, rect, terrain_type):
        """Draw simple tree sprite on forest tiles"""
        from terrain_generator import TerrainType

        # Tree colors based on terrain
        if terrain_type == TerrainType.RAINFOREST:
            trunk_color = (101, 67, 33)  # Dark brown
            foliage_color = (27, 94, 32)  # Dark green
        else:  # FOREST
            trunk_color = (121, 85, 72)  # Medium brown
            foliage_color = (56, 142, 60)  # Medium green

        # Calculate tree dimensions (proportional to cell size)
        trunk_width = max(2, self.cell_size // 8)
        trunk_height = self.cell_size // 2
        foliage_radius = self.cell_size // 3

        # Tree position (centered in cell)
        center_x = rect.centerx
        base_y = rect.bottom - 2

        # Draw trunk
        trunk_rect = pygame.Rect(
            center_x - trunk_width // 2,
            base_y - trunk_height,
            trunk_width,
            trunk_height
        )
        pygame.draw.rect(self.screen, trunk_color, trunk_rect)

        # Draw foliage (simple circle)
        foliage_center_y = base_y - trunk_height - foliage_radius // 2
        pygame.draw.circle(self.screen, foliage_color,
                          (center_x, foliage_center_y),
                          foliage_radius)

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

    def draw_sprite(self, rect, sprite_key, direction, animate=True):
        """Draw a directional animated sprite from the sprite sheet with earthy color tinting"""
        if not self.dino_sprites or sprite_key not in self.dino_sprites:
            # Fallback to colored circle if sprite not available
            color = COLORS.get(sprite_key, (128, 128, 128))
            pygame.draw.circle(self.screen, color, rect.center, rect.width // 3)
            return

        # Get the dinosaur sprite handler
        dino_sprite = self.dino_sprites[sprite_key]

        # Get directional sprite based on movement (dx, dy)
        sprite = dino_sprite.get_sprite_from_movement(direction[0], direction[1], animate)

        # Apply earthy color tint (replaces green with brown/tan colors)
        if sprite_key in self.dino_tints:
            tint_color = self.dino_tints[sprite_key]

            # Create a copy to apply tint
            tinted_sprite = sprite.copy()

            # Create color overlay surface
            color_overlay = pygame.Surface(tinted_sprite.get_size(), pygame.SRCALPHA)
            color_overlay.fill((*tint_color, 180))  # Semi-transparent tint

            # Apply tint using multiply blend mode
            tinted_sprite.blit(color_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            sprite = tinted_sprite

        # Center the sprite on the rect
        sprite_rect = sprite.get_rect(center=rect.center)

        # Draw the sprite (allow it to extend beyond cell bounds - no clipping!)
        self.screen.blit(sprite, sprite_rect)

    def draw_directional_sprite(self, rect, color, direction, size_ratio, is_carnivore=False):
        """Draw dinosaur-shaped sprite pointing in movement direction (multi-tile sprites)"""
        import math

        # Calculate angle from direction
        if direction == (0, 0):
            angle = 0
        else:
            angle = math.atan2(direction[1], direction[0])

        # Scale dinosaurs to span multiple tiles for visibility
        # Size ratio represents relative size (0.45 = ~3 tiles for T-Rex, 0.3 = ~2 tiles for Gallimimus)
        radius = int(self.cell_size * size_ratio * 3.5)  # Make them span multiple tiles for visibility
        center_x, center_y = rect.center

        # Create dinosaur-shaped sprite (body, head, tail, legs)
        # Points are relative to center, pointing right initially
        body_length = radius
        body_width = radius * 0.6
        head_size = radius * 0.4
        tail_length = radius * 0.7

        # Body (ellipse-ish shape with multiple points)
        body_points = [
            (body_length * 0.3, 0),  # Neck connection
            (body_length * 0.5, -body_width * 0.3),  # Top of body
            (body_length * 0.3, -body_width * 0.5),  # Upper back
            (-body_length * 0.2, -body_width * 0.4),  # Back top
            (-body_length * 0.5, -body_width * 0.2),  # Tail start top
            (-body_length * 0.8, 0),  # Tail end
            (-body_length * 0.5, body_width * 0.2),  # Tail start bottom
            (-body_length * 0.2, body_width * 0.4),  # Back bottom
            (body_length * 0.3, body_width * 0.5),  # Lower back
            (body_length * 0.5, body_width * 0.3),  # Bottom of body
        ]

        # Head (pointing forward)
        head_points = [
            (body_length * 0.3, 0),  # Neck
            (body_length * 0.7, -head_size * 0.3),  # Top of head
            (body_length, 0),  # Snout/mouth
            (body_length * 0.7, head_size * 0.3),  # Bottom of head
        ]

        # Rotate all points
        def rotate_point(px, py):
            rx = px * math.cos(angle) - py * math.sin(angle)
            ry = px * math.sin(angle) + py * math.cos(angle)
            return (center_x + rx, center_y + ry)

        body_rotated = [rotate_point(px, py) for px, py in body_points]
        head_rotated = [rotate_point(px, py) for px, py in head_points]

        # Draw body (main dinosaur shape)
        pygame.draw.polygon(self.screen, color, body_rotated)

        # Draw head
        head_color = tuple(min(255, c + 20) for c in color)  # Slightly lighter
        pygame.draw.polygon(self.screen, head_color, head_rotated)

        # Draw outlines for definition
        outline_color = tuple(max(0, c - 50) for c in color)
        pygame.draw.polygon(self.screen, outline_color, body_rotated, 2)
        pygame.draw.polygon(self.screen, outline_color, head_rotated, 2)

        # Add eye dot
        eye_x = body_length * 0.6
        eye_y = -head_size * 0.15
        eye_rotated = rotate_point(eye_x, eye_y)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(eye_rotated[0]), int(eye_rotated[1])), max(2, int(radius * 0.15)))

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

    def draw_movement_trail(self, agent, current_x, current_y):
        """Draw fading trail showing recent movement (camera-aware)"""
        if not hasattr(agent, 'movement_history') or not agent.movement_history:
            return

        # Draw trail with fading alpha
        for i, old_pos in enumerate(agent.movement_history):
            # Check if trail position is visible
            if not self.camera.is_visible(old_pos[0], old_pos[1]):
                continue

            alpha = int(100 * (i + 1) / len(agent.movement_history))  # Fade from 33 to 100
            trail_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)

            # Get agent color
            if isinstance(agent, Triceratops):
                color = COLORS['triceratops']
            elif isinstance(agent, Gallimimus):
                color = COLORS['gallimimus']
            elif isinstance(agent, Stegosaurus):
                color = COLORS['stegosaurus']
            elif isinstance(agent, Pachycephalosaurus):
                color = COLORS['pachycephalosaurus']
            elif isinstance(agent, Brachiosaurus):
                color = COLORS['brachiosaurus']
            elif isinstance(agent, Archeopteryx):
                color = COLORS['archeopteryx']
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

            # Convert world coordinates to screen coordinates
            screen_x, screen_y = self.camera.world_to_screen(old_pos[0], old_pos[1])
            self.screen.blit(trail_surface, (int(screen_x), int(screen_y)))

    def draw_grid(self):
        """Draw terrain and agents in two separate passes to prevent clipping"""
        from terrain_generator import TERRAIN_COLORS, TerrainType

        # Get visible cell bounds from camera
        min_x, min_y, max_x, max_y = self.camera.get_visible_bounds()

        # ===== PASS 1: Draw terrain only =====
        for world_y in range(min_y, max_y):
            for world_x in range(min_x, max_x):
                # Convert world coordinates to screen coordinates
                screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)

                # Create rect in screen space
                rect = pygame.Rect(
                    int(screen_x),
                    int(screen_y),
                    self.cell_size,
                    self.cell_size
                )

                # Only draw if rect is actually visible on screen
                if rect.right < 0 or rect.left > self.viewport_width:
                    continue
                if rect.bottom < 0 or rect.top > self.viewport_height:
                    continue

                # Draw terrain background
                if self.terrain_map is not None:
                    terrain_type = TerrainType(self.terrain_map[world_y, world_x])
                    terrain_color = TERRAIN_COLORS[terrain_type]

                    # Animated water effect for ocean and rivers
                    if terrain_type in [TerrainType.OCEAN, TerrainType.RIVER]:
                        import math
                        # Create wave effect
                        wave = math.sin((world_x + self.water_offset) * 0.5) * math.cos((world_y + self.water_offset) * 0.3)
                        brightness_change = int(wave * 15)
                        terrain_color = tuple(max(0, min(255, c + brightness_change)) for c in terrain_color)

                        # Add shimmer effect
                        pygame.draw.rect(self.screen, terrain_color, rect)
                        if wave > 0.5:
                            # Draw highlight
                            shimmer_color = tuple(min(255, c + 30) for c in terrain_color)
                            shimmer_rect = pygame.Rect(rect.x, rect.y, rect.width, 2)
                            pygame.draw.rect(self.screen, shimmer_color, shimmer_rect)
                    else:
                        # Add slight procedural variation to land tiles
                        noise = self.terrain_noise[world_y][world_x]
                        terrain_color = tuple(max(0, min(255, c + noise - 10)) for c in terrain_color)
                        pygame.draw.rect(self.screen, terrain_color, rect)

                        # Draw tree sprites on forest and rainforest tiles
                        if terrain_type in [TerrainType.FOREST, TerrainType.RAINFOREST]:
                            # Only draw trees occasionally for performance (every 2nd cell with randomness)
                            if (world_x + world_y) % 2 == 0:
                                self.draw_tree(rect, terrain_type)

        # ===== PASS 2: Draw agents on top (NO viewport clipping to prevent sprite cutoff) =====
        for world_y in range(min_y, max_y):
            for world_x in range(min_x, max_x):
                # Convert world coordinates to screen coordinates
                screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)

                # Create rect in screen space (for centering sprites)
                rect = pygame.Rect(
                    int(screen_x),
                    int(screen_y),
                    self.cell_size,
                    self.cell_size
                )

                # NO viewport clipping check here - let large sprites extend beyond cell!
                # This prevents sprite clipping at viewport edges

                # Get cell contents from model
                cell_contents = self.model.grid.get_cell_list_contents([(world_x, world_y)])

                # Determine what to draw (priority: carnivore > herbivore > grass)
                is_learning_agent = False
                drawn_agent = None

                for agent in cell_contents:
                    # Check if it's a learning agent
                    if hasattr(agent, '__class__'):
                        is_learning_agent = 'Learning' in agent.__class__.__name__ or 'PPO' in agent.__class__.__name__

                    if isinstance(agent, TRex):
                        color = COLORS['trex']
                        # Draw movement trail first (behind sprite)
                        self.draw_movement_trail(agent, world_x, world_y)
                        # Draw sprite with animation ONLY if moving
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'trex', direction, animate=is_moving)
                        # Draw AI indicator
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'TRex')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Velociraptor):
                        color = COLORS['velociraptor']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'velociraptor', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Velociraptor')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Triceratops):
                        color = COLORS['triceratops']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'triceratops', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Triceratops')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Gallimimus):
                        color = COLORS['gallimimus']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'gallimimus', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Gallimimus')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Stegosaurus):
                        color = COLORS['stegosaurus']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'stegosaurus', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Stegosaurus')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Pachycephalosaurus):
                        color = COLORS['pachycephalosaurus']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'pachycephalosaurus', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Pachycephalosaurus')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Brachiosaurus):
                        color = COLORS['brachiosaurus']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'brachiosaurus', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Brachiosaurus')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Archeopteryx):
                        color = COLORS['archeopteryx']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'archeopteryx', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'Archeopteryx')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, CarnivoreAgent):
                        color = COLORS['carnivore']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'trex', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'CarnivoreAgent')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, HerbivoreAgent):
                        color = COLORS['herbivore']
                        self.draw_movement_trail(agent, world_x, world_y)
                        direction = getattr(agent, 'direction', (1, 0))
                        is_moving = getattr(agent, 'is_moving', False)
                        self.draw_sprite(rect, 'triceratops', direction, animate=is_moving)
                        if is_learning_agent:
                            self.draw_ai_indicator(rect, 'HerbivoreAgent')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, GrassAgent):
                        # Grass is now drawn as terrain, skip
                        pass

                # Draw energy bar for dinosaurs
                if drawn_agent and hasattr(drawn_agent, 'energy'):
                    if isinstance(drawn_agent, (HerbivoreAgent, CarnivoreAgent)):
                        max_energy = 150 if isinstance(drawn_agent, CarnivoreAgent) else 100
                        self.draw_energy_bar(rect, drawn_agent.energy, max_energy)

        # Draw grid overlay (on visible area only) - adjusted for cell size
        grid_spacing = max(4, int(64 / self.cell_size))  # At 16px: every 4 cells, at 8px: every 8 cells
        for x in range(0, self.viewport_width, self.cell_size * grid_spacing):
            pygame.draw.line(self.screen, COLORS['grid_lines'],
                           (x, 0), (x, self.viewport_height), 1)
        for y in range(0, self.viewport_height, self.cell_size * grid_spacing):
            pygame.draw.line(self.screen, COLORS['grid_lines'],
                           (0, y), (self.viewport_width, y), 1)

        # Draw grid coordinates (based on camera position) - adjusted for cell size
        coord_font = pygame.font.Font(None, 14)
        # X-axis labels
        for i in range(min_x, max_x, grid_spacing):
            label = chr(65 + (i // grid_spacing) % 26)  # A-Z
            screen_x, _ = self.camera.world_to_screen(i, 0)
            if 0 <= screen_x < self.viewport_width:
                text = coord_font.render(label, True, COLORS['text_highlight'])
                self.screen.blit(text, (int(screen_x) + 2, 2))

        # Y-axis labels
        for i in range(min_y, max_y, grid_spacing):
            label = str((i // grid_spacing) + 1)
            _, screen_y = self.camera.world_to_screen(0, i)
            if 0 <= screen_y < self.viewport_height:
                text = coord_font.render(label, True, COLORS['text_highlight'])
                self.screen.blit(text, (2, int(screen_y) + 2))

    def draw_panel(self):
        """Draw info panel on the right"""
        panel_x = self.viewport_width

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
        steg_count = len([a for a in self.model.agents if isinstance(a, Stegosaurus)])
        pachy_count = len([a for a in self.model.agents if isinstance(a, Pachycephalosaurus)])
        brach_count = len([a for a in self.model.agents if isinstance(a, Brachiosaurus)])
        arch_count = len([a for a in self.model.agents if isinstance(a, Archeopteryx)])
        trex_count = len([a for a in self.model.agents if isinstance(a, TRex)])
        vel_count = len([a for a in self.model.agents if isinstance(a, Velociraptor)])
        total_dinos = tri_count + gal_count + steg_count + pachy_count + brach_count + arch_count + trex_count + vel_count

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
        y_offset += 20

        pygame.draw.circle(self.screen, COLORS['stegosaurus'], (panel_x + 35, y_offset + 6), 5)
        steg_text = self.font_small.render(f"Stegosaurus: {steg_count}", True, COLORS['text'])
        self.screen.blit(steg_text, (panel_x + 50, y_offset))
        y_offset += 20

        pygame.draw.circle(self.screen, COLORS['pachycephalosaurus'], (panel_x + 35, y_offset + 6), 5)
        pachy_text = self.font_small.render(f"Pachyceph.: {pachy_count}", True, COLORS['text'])
        self.screen.blit(pachy_text, (panel_x + 50, y_offset))
        y_offset += 20

        pygame.draw.circle(self.screen, COLORS['brachiosaurus'], (panel_x + 35, y_offset + 6), 5)
        brach_text = self.font_small.render(f"Brachiosaur: {brach_count}", True, COLORS['text'])
        self.screen.blit(brach_text, (panel_x + 50, y_offset))
        y_offset += 20

        pygame.draw.circle(self.screen, COLORS['archeopteryx'], (panel_x + 35, y_offset + 6), 5)
        arch_text = self.font_small.render(f"Archeopteryx: {arch_count}", True, COLORS['text'])
        self.screen.blit(arch_text, (panel_x + 50, y_offset))
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
            "Arrows- Pan camera",
            "Drag  - Move view",
            "M     - Toggle map",
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
        legend_height = 230 if self.use_learning_agents else 210
        legend_surface = pygame.Surface((200, legend_height))
        legend_surface.set_alpha(220)
        legend_surface.fill(COLORS['panel'])
        self.screen.blit(legend_surface, (legend_x, legend_y))

        # Title
        title_text = self.font_small.render("LEGEND", True, COLORS['text'])
        self.screen.blit(title_text, (legend_x + 10, legend_y + 8))

        legend_items = [
            ("Grass", COLORS['grass_full'], 'rect'),
            ("Triceratops", COLORS['triceratops'], 'circle'),
            ("Gallimimus", COLORS['gallimimus'], 'circle'),
            ("Stegosaurus", COLORS['stegosaurus'], 'circle'),
            ("Pachyceph.", COLORS['pachycephalosaurus'], 'circle'),
            ("Brachiosaurus", COLORS['brachiosaurus'], 'circle'),
            ("Archeopteryx", COLORS['archeopteryx'], 'circle'),
            ("T-Rex", COLORS['trex'], 'circle'),
            ("Velociraptor", COLORS['velociraptor'], 'circle'),
        ]

        if self.use_learning_agents:
            legend_items.append(("AI Agent", (255, 255, 255), 'ring'))

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

    def draw_minimap(self):
        """Draw minimap showing full island and camera position"""
        if not self.show_minimap or self.terrain_map is None:
            return

        from terrain_generator import TERRAIN_COLORS, TerrainType

        # Minimap position (bottom-left corner of main viewport)
        minimap_x = 10
        minimap_y = self.viewport_height - self.minimap_height - 10

        # Draw semi-transparent background
        minimap_bg = pygame.Surface((self.minimap_width, self.minimap_height))
        minimap_bg.set_alpha(220)
        minimap_bg.fill(COLORS['panel'])
        self.screen.blit(minimap_bg, (minimap_x, minimap_y))

        # Draw border
        pygame.draw.rect(self.screen, COLORS['panel_border'],
                        (minimap_x, minimap_y, self.minimap_width, self.minimap_height), 2)

        # Scale factors
        scale_x = (self.minimap_width - 4) / self.model_width
        scale_y = (self.minimap_height - 4) / self.model_height

        # Draw simplified terrain (every Nth cell)
        step = max(1, int(1 / min(scale_x, scale_y)))
        for y in range(0, self.model_height, step):
            for x in range(0, self.model_width, step):
                terrain_type = TerrainType(self.terrain_map[y, x])
                terrain_color = TERRAIN_COLORS[terrain_type]

                # Calculate minimap pixel position
                mini_x = minimap_x + 2 + int(x * scale_x)
                mini_y = minimap_y + 2 + int(y * scale_y)
                mini_w = max(1, int(step * scale_x))
                mini_h = max(1, int(step * scale_y))

                pygame.draw.rect(self.screen, terrain_color,
                               (mini_x, mini_y, mini_w, mini_h))

        # Draw viewport indicator (yellow rectangle)
        viewport_rect = self.camera.minimap_coords(self.minimap_width - 4, self.minimap_height - 4)
        viewport_rect.x += minimap_x + 2
        viewport_rect.y += minimap_y + 2
        pygame.draw.rect(self.screen, COLORS['panel_border'], viewport_rect, 2)

        # Draw minimap title
        title = self.font_small.render("MAP", True, COLORS['text_highlight'])
        self.screen.blit(title, (minimap_x + 5, minimap_y + 3))

    def draw_environmental_effects(self):
        """Draw weather effects (rain, heat), volcano particles, and day/night cycle"""
        if not self.model:
            return

        # Rain effect when rainfall is high
        if self.model.rainfall > 120:
            import random
            rain_surface = pygame.Surface((self.viewport_width, self.viewport_height), pygame.SRCALPHA)
            # Draw rain drops
            for _ in range(int(self.model.rainfall / 3)):
                x = random.randint(0, self.viewport_width)
                y = random.randint(0, self.viewport_height)
                pygame.draw.line(rain_surface, (100, 150, 200, 100), (x, y), (x + 2, y + 5), 1)
            self.screen.blit(rain_surface, (0, 0))

        # Heat shimmer when temperature is high
        if self.model.temperature > 32:
            heat_surface = pygame.Surface((self.viewport_width, self.viewport_height), pygame.SRCALPHA)
            heat_alpha = min(80, int((self.model.temperature - 32) * 5))
            heat_surface.fill((255, 100, 50, heat_alpha))
            self.screen.blit(heat_surface, (0, 0))

        # Draw volcano smoke particles
        self.draw_volcano_particles()

        # Day/night cycle lighting overlay
        self.draw_day_night_cycle()

    def draw_volcano_particles(self):
        """Draw smoke/steam particles from volcanoes"""
        for particle in self.volcano_particles:
            # Convert world coordinates to screen coordinates
            screen_x, screen_y = self.camera.world_to_screen(
                particle['x'] + particle['offset_x'],
                particle['y'] + particle['offset_y']
            )

            # Only draw if visible
            if 0 <= screen_x < self.viewport_width and 0 <= screen_y < self.viewport_height:
                # Gray smoke with alpha based on life
                alpha = int(particle['life'] * 180)
                smoke_surface = pygame.Surface((particle['size']*2, particle['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(
                    smoke_surface,
                    (80, 80, 80, alpha),
                    (particle['size'], particle['size']),
                    particle['size']
                )
                self.screen.blit(smoke_surface, (int(screen_x) - particle['size'], int(screen_y) - particle['size']))

    def draw_day_night_cycle(self):
        """Draw lighting overlay for day/night cycle"""
        import math

        # Calculate darkness based on time of day
        # time_of_day: 0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset, 1.0=midnight
        if self.time_of_day < 0.25:
            # Late night to sunrise
            darkness = 1.0 - (self.time_of_day / 0.25) * 0.7
        elif self.time_of_day < 0.5:
            # Sunrise to noon (getting brighter)
            darkness = 0.3 - ((self.time_of_day - 0.25) / 0.25) * 0.3
        elif self.time_of_day < 0.75:
            # Noon to sunset (getting darker)
            darkness = ((self.time_of_day - 0.5) / 0.25) * 0.6
        else:
            # Sunset to midnight
            darkness = 0.6 + ((self.time_of_day - 0.75) / 0.25) * 0.4

        # Apply darkness overlay
        if darkness > 0.05:  # Only draw if there's noticeable darkness
            night_surface = pygame.Surface((self.viewport_width, self.viewport_height), pygame.SRCALPHA)
            night_alpha = int(darkness * 180)  # Max 180 alpha (not completely black)

            # Darker blue tint at night
            night_color = (10, 20, 40, night_alpha)
            night_surface.fill(night_color)
            self.screen.blit(night_surface, (0, 0))

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
            self.draw_minimap()  # Draw minimap on top

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

    # Use default world (625x375) with 16px cells on widescreen viewport
    viz = JurassicParkViz(width=625, height=375, cell_size=16,
                          use_learning_agents=use_learning,
                          use_ppo_agents=use_ppo)
    viz.run()
