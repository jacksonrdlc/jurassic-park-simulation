"""
Terrain Rendering
Handles drawing of terrain tiles, grass, water effects, sprite sheets
"""

import math
import random
import pygame
from terrain_generator import TERRAIN_COLORS, TerrainType
from my_first_island import (GrassAgent, HerbivoreAgent, CarnivoreAgent,
                              Triceratops, Gallimimus, TRex, Velociraptor)
from src.config.colors import COLORS
from src.config.settings import (CELL_SIZE, VIEWPORT_WIDTH, VIEWPORT_HEIGHT,
                                 SPRITE_SCALES, ENERGY_BAR_HEIGHT, ENERGY_BAR_OFFSET,
                                 MOVEMENT_TRAIL_LENGTH, MOVEMENT_TRAIL_ALPHA_MIN,
                                 MOVEMENT_TRAIL_ALPHA_MAX)
from src.utils.sprite_loader import SpriteLoader


class TerrainRenderer:
    """Renders terrain, agents, and grid overlay"""

    def __init__(self, screen, camera):
        """
        Initialize terrain renderer

        Args:
            screen: Pygame screen surface
            camera: Camera object
        """
        self.screen = screen
        self.camera = camera
        self.water_offset = 0
        self.coord_font = pygame.font.Font(None, 14)

        # Load sprites
        SpriteLoader.load_sprites()

        # Cache tree positions for performance (only render trees on checkerboard pattern)
        self.tree_positions = set()

    def update_water_animation(self, offset):
        """Update water animation offset"""
        self.water_offset = offset

    def draw_terrain(self, terrain_map, terrain_noise):
        """
        Draw terrain tiles with tree sprites

        Args:
            terrain_map: 2D array of terrain types
            terrain_noise: 2D array of color noise values
        """
        if terrain_map is None:
            return

        min_x, min_y, max_x, max_y = self.camera.get_visible_bounds()

        for world_y in range(min_y, max_y):
            for world_x in range(min_x, max_x):
                screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)

                rect = pygame.Rect(
                    int(screen_x),
                    int(screen_y),
                    CELL_SIZE,
                    CELL_SIZE
                )

                # Skip if not visible
                if rect.right < 0 or rect.left > VIEWPORT_WIDTH:
                    continue
                if rect.bottom < 0 or rect.top > VIEWPORT_HEIGHT:
                    continue

                # Get terrain type and color
                terrain_type = TerrainType(terrain_map[world_y, world_x])
                terrain_color = TERRAIN_COLORS[terrain_type]

                # Animated water effect for ocean and rivers
                if terrain_type in [TerrainType.OCEAN, TerrainType.RIVER]:
                    terrain_color = self._animate_water(terrain_color, world_x, world_y)
                    pygame.draw.rect(self.screen, terrain_color, rect)
                else:
                    # Add procedural variation to land tiles
                    noise = terrain_noise[world_y][world_x]
                    terrain_color = tuple(max(0, min(255, c + noise - 10)) for c in terrain_color)
                    pygame.draw.rect(self.screen, terrain_color, rect)

                    # Draw trees on forests/rainforests (checkerboard pattern for performance)
                    if terrain_type in [TerrainType.FOREST, TerrainType.RAINFOREST]:
                        if (world_x + world_y) % 2 == 0:
                            self._draw_tree(rect, terrain_type)

    def _animate_water(self, base_color, world_x, world_y):
        """
        Apply wave animation to water tiles

        Args:
            base_color: Base water color
            world_x: World x coordinate
            world_y: World y coordinate

        Returns:
            Animated color
        """
        wave = math.sin((world_x + self.water_offset) * 0.5) * math.cos((world_y + self.water_offset) * 0.3)
        brightness_change = int(wave * 15)
        animated_color = tuple(max(0, min(255, c + brightness_change)) for c in base_color)

        # Draw shimmer effect on wave peaks
        if wave > 0.5:
            # Shimmer will be drawn as highlight in draw_water_shimmer
            pass

        return animated_color

    def _draw_tree(self, rect, terrain_type):
        """
        Draw procedural tree sprite

        Args:
            rect: Screen rect for the cell
            terrain_type: Type of terrain (forest or rainforest)
        """
        # Seed random with position for consistent tree appearance
        seed = rect.x * 1000 + rect.y
        tree_rng = random.Random(seed)

        # Tree colors
        if terrain_type == TerrainType.RAINFOREST:
            trunk_color = (70, 45, 30)  # Dark brown
            leaf_color = (30, 80, 30)   # Dark green
        else:  # Regular forest
            trunk_color = (101, 67, 33)  # Medium brown
            leaf_color = (46, 125, 50)  # Medium green

        # Tree dimensions (scaled for 16px cells)
        trunk_width = max(2, int(CELL_SIZE * 0.15))
        trunk_height = max(4, int(CELL_SIZE * 0.4))
        canopy_radius = max(3, int(CELL_SIZE * 0.3))

        # Tree position (slightly randomized within cell)
        tree_x = rect.centerx + tree_rng.randint(-2, 2)
        tree_y = rect.centery + tree_rng.randint(-2, 2)

        # Draw trunk
        trunk_rect = pygame.Rect(
            tree_x - trunk_width // 2,
            tree_y,
            trunk_width,
            trunk_height
        )
        pygame.draw.rect(self.screen, trunk_color, trunk_rect)

        # Draw canopy (circle)
        canopy_center = (tree_x, tree_y - trunk_height // 2)
        pygame.draw.circle(self.screen, leaf_color, canopy_center, canopy_radius)

        # Add highlight for depth
        highlight_color = tuple(min(255, c + 20) for c in leaf_color)
        pygame.draw.circle(self.screen, highlight_color, canopy_center, canopy_radius // 2)

    def draw_water_shimmer(self, terrain_map):
        """Draw shimmer highlights on water"""
        if terrain_map is None:
            return

        min_x, min_y, max_x, max_y = self.camera.get_visible_bounds()

        for world_y in range(min_y, max_y):
            for world_x in range(min_x, max_x):
                terrain_type = TerrainType(terrain_map[world_y, world_x])

                if terrain_type in [TerrainType.OCEAN, TerrainType.RIVER]:
                    wave = math.sin((world_x + self.water_offset) * 0.5) * math.cos((world_y + self.water_offset) * 0.3)

                    if wave > 0.5:
                        screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)
                        rect = pygame.Rect(int(screen_x), int(screen_y), CELL_SIZE, 2)

                        terrain_color = TERRAIN_COLORS[terrain_type]
                        shimmer_color = tuple(min(255, c + 30) for c in terrain_color)
                        pygame.draw.rect(self.screen, shimmer_color, rect)

    def draw_agents(self, model):
        """
        Draw all agents (dinosaurs, grass, etc.) using sprite sheets
        NO viewport clipping - allows sprites to extend beyond cell boundaries

        Args:
            model: Simulation model
        """
        min_x, min_y, max_x, max_y = self.camera.get_visible_bounds()

        # Iterate through visible cells
        for world_y in range(min_y, max_y):
            for world_x in range(min_x, max_x):
                # Get cell contents
                cell_contents = model.grid.get_cell_list_contents([(world_x, world_y)])

                # Draw agents (priority: carnivore > herbivore > grass)
                drawn_agent = None
                is_learning_agent = False
                species_name = None

                for agent in cell_contents:
                    # Check if it's a learning agent
                    if hasattr(agent, '__class__'):
                        is_learning_agent = 'Learning' in agent.__class__.__name__ or 'PPO' in agent.__class__.__name__

                    # Identify species and draw sprite
                    if isinstance(agent, TRex):
                        species_name = 'trex'
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Velociraptor):
                        species_name = 'velociraptor'
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Triceratops):
                        species_name = 'triceratops'
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Gallimimus):
                        species_name = 'gallimimus'
                        drawn_agent = agent
                        break
                    elif isinstance(agent, CarnivoreAgent):
                        species_name = 'velociraptor'  # Default carnivore sprite
                        drawn_agent = agent
                        break
                    elif isinstance(agent, HerbivoreAgent):
                        species_name = 'triceratops'  # Default herbivore sprite
                        drawn_agent = agent
                        break

                # Draw the agent if found
                if drawn_agent and species_name:
                    # Get screen position (NO viewport clipping - sprites can extend freely!)
                    screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)

                    # Draw movement trail
                    self._draw_movement_trail(drawn_agent, screen_x, screen_y, species_name)

                    # Get direction and is_moving state
                    direction = getattr(drawn_agent, 'direction', (1, 0))
                    is_moving = getattr(drawn_agent, 'is_moving', False)

                    # Get animated sprite (only animates if is_moving=True)
                    sprite = SpriteLoader.get_sprite(species_name, direction, animate=is_moving)

                    if sprite:
                        # Center sprite on cell
                        sprite_rect = sprite.get_rect()
                        sprite_rect.center = (screen_x + CELL_SIZE // 2, screen_y + CELL_SIZE // 2)

                        # Draw sprite (NO clipping - let it extend beyond cell!)
                        self.screen.blit(sprite, sprite_rect)

                        # Draw AI indicator if learning agent
                        if is_learning_agent:
                            self._draw_ai_indicator_centered(sprite_rect, species_name.title())

                        # Draw energy bar
                        if hasattr(drawn_agent, 'energy'):
                            max_energy = 150 if isinstance(drawn_agent, CarnivoreAgent) else 100
                            self._draw_energy_bar_centered(sprite_rect, drawn_agent.energy, max_energy)

    def _draw_movement_trail(self, agent, screen_x, screen_y, species_name):
        """
        Draw fading ghost trail behind moving dinosaur

        Args:
            agent: Agent object
            screen_x: Screen x coordinate
            screen_y: Screen y coordinate
            species_name: Name of species for sprite lookup
        """
        if not hasattr(agent, 'position_history') or len(agent.position_history) < 2:
            return

        # Draw trail positions (oldest to newest, fading in)
        for i, (old_x, old_y) in enumerate(agent.position_history[-MOVEMENT_TRAIL_LENGTH:]):
            # Calculate alpha based on age (older = more transparent)
            progress = (i + 1) / MOVEMENT_TRAIL_LENGTH
            alpha = int(MOVEMENT_TRAIL_ALPHA_MIN + (MOVEMENT_TRAIL_ALPHA_MAX - MOVEMENT_TRAIL_ALPHA_MIN) * progress)

            # Get trail sprite
            direction = getattr(agent, 'direction', (1, 0))
            trail_sprite = SpriteLoader.get_sprite(species_name, direction, animate=False)

            if trail_sprite:
                # Make sprite semi-transparent
                trail_sprite = trail_sprite.copy()
                trail_sprite.set_alpha(alpha)

                # Calculate screen position for old position
                old_screen_x, old_screen_y = self.camera.world_to_screen(old_x, old_y)

                # Center on cell
                sprite_rect = trail_sprite.get_rect()
                sprite_rect.center = (old_screen_x + CELL_SIZE // 2, old_screen_y + CELL_SIZE // 2)

                # Draw trail sprite
                self.screen.blit(trail_sprite, sprite_rect)

    def _draw_energy_bar_centered(self, sprite_rect, energy, max_energy):
        """Draw energy bar above sprite"""
        bar_width = sprite_rect.width
        bar_height = ENERGY_BAR_HEIGHT
        bar_x = sprite_rect.centerx - bar_width // 2
        bar_y = sprite_rect.top - ENERGY_BAR_OFFSET

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

    def _draw_ai_indicator_centered(self, sprite_rect, species_name):
        """Draw AI badge for learning agents"""
        from src.config.colors import AI_BADGE_COLORS
        from src.config.settings import AI_BADGE_RADIUS_RATIO, AI_BADGE_OFFSET_RATIO, AI_BADGE_FONT_SIZE_RATIO

        symbol, color = AI_BADGE_COLORS.get(species_name, ('A', (255, 255, 255)))

        # Calculate badge position relative to sprite
        badge_size = max(sprite_rect.width, sprite_rect.height)
        badge_center = (
            sprite_rect.left + int(badge_size * AI_BADGE_OFFSET_RATIO),
            sprite_rect.top + int(badge_size * AI_BADGE_OFFSET_RATIO)
        )
        badge_radius = int(badge_size * AI_BADGE_RADIUS_RATIO)

        # Draw circle background
        circle_surface = pygame.Surface((badge_radius * 2 + 2, badge_radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (0, 0, 0, 180), (badge_radius + 1, badge_radius + 1), badge_radius)
        pygame.draw.circle(circle_surface, color, (badge_radius + 1, badge_radius + 1), badge_radius, 2)
        self.screen.blit(circle_surface, (badge_center[0] - badge_radius - 1, badge_center[1] - badge_radius - 1))

        # Draw symbol
        symbol_font = pygame.font.Font(None, int(badge_size * AI_BADGE_FONT_SIZE_RATIO))
        symbol_surface = symbol_font.render(symbol, True, color)
        symbol_rect = symbol_surface.get_rect(center=badge_center)
        self.screen.blit(symbol_surface, symbol_rect)

    def draw_grid_overlay(self):
        """Draw grid lines and coordinates"""
        # Grid lines
        for x in range(0, VIEWPORT_WIDTH, CELL_SIZE * 5):
            pygame.draw.line(self.screen, COLORS['grid_lines'],
                           (x, 0), (x, VIEWPORT_HEIGHT), 1)
        for y in range(0, VIEWPORT_HEIGHT, CELL_SIZE * 5):
            pygame.draw.line(self.screen, COLORS['grid_lines'],
                           (0, y), (VIEWPORT_WIDTH, y), 1)

        # Grid coordinates
        min_x, min_y, max_x, max_y = self.camera.get_visible_bounds()

        # X-axis labels
        for i in range(min_x, max_x, 5):
            label = chr(65 + (i // 5) % 26)  # A-Z
            screen_x, _ = self.camera.world_to_screen(i, 0)
            if 0 <= screen_x < VIEWPORT_WIDTH:
                text = self.coord_font.render(label, True, COLORS['text_highlight'])
                self.screen.blit(text, (int(screen_x) + 2, 2))

        # Y-axis labels
        for i in range(min_y, max_y, 5):
            label = str((i // 5) + 1)
            _, screen_y = self.camera.world_to_screen(0, i)
            if 0 <= screen_y < VIEWPORT_HEIGHT:
                text = self.coord_font.render(label, True, COLORS['text_highlight'])
                self.screen.blit(text, (2, int(screen_y) + 2))
