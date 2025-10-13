"""
Terrain Rendering
Handles drawing of terrain tiles, grass, water effects
"""

import math
import pygame
from terrain_generator import TERRAIN_COLORS, TerrainType
from my_first_island import (GrassAgent, HerbivoreAgent, CarnivoreAgent,
                              Triceratops, Gallimimus, TRex, Velociraptor)
from src.config.colors import COLORS
from src.config.settings import CELL_SIZE, VIEWPORT_WIDTH, VIEWPORT_HEIGHT
from src.entities.dinosaur_sprite import draw_dinosaur, draw_movement_trail


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

    def update_water_animation(self, offset):
        """Update water animation offset"""
        self.water_offset = offset

    def draw_terrain(self, terrain_map, terrain_noise):
        """
        Draw terrain tiles

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
        Draw all agents (dinosaurs, grass, etc.)

        Args:
            model: Simulation model
        """
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

                # Get cell contents
                cell_contents = model.grid.get_cell_list_contents([(world_x, world_y)])

                # Draw agents (priority: carnivore > herbivore > grass)
                drawn_agent = None
                is_learning_agent = False

                for agent in cell_contents:
                    # Check if it's a learning agent
                    if hasattr(agent, '__class__'):
                        is_learning_agent = 'Learning' in agent.__class__.__name__ or 'PPO' in agent.__class__.__name__

                    # Draw appropriate sprite
                    if isinstance(agent, TRex):
                        color = COLORS['trex']
                        draw_movement_trail(self.screen, CELL_SIZE, agent, self.camera, color)
                        direction = getattr(agent, 'direction', (1, 0))
                        draw_dinosaur(self.screen, rect, color, direction, 0.45)
                        if is_learning_agent:
                            self._draw_ai_indicator(rect, 'TRex')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Velociraptor):
                        color = COLORS['velociraptor']
                        draw_movement_trail(self.screen, CELL_SIZE, agent, self.camera, color)
                        direction = getattr(agent, 'direction', (1, 0))
                        draw_dinosaur(self.screen, rect, color, direction, 0.4)
                        if is_learning_agent:
                            self._draw_ai_indicator(rect, 'Velociraptor')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Triceratops):
                        color = COLORS['triceratops']
                        draw_movement_trail(self.screen, CELL_SIZE, agent, self.camera, color)
                        direction = getattr(agent, 'direction', (1, 0))
                        draw_dinosaur(self.screen, rect, color, direction, 0.35)
                        if is_learning_agent:
                            self._draw_ai_indicator(rect, 'Triceratops')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, Gallimimus):
                        color = COLORS['gallimimus']
                        draw_movement_trail(self.screen, CELL_SIZE, agent, self.camera, color)
                        direction = getattr(agent, 'direction', (1, 0))
                        draw_dinosaur(self.screen, rect, color, direction, 0.3)
                        if is_learning_agent:
                            self._draw_ai_indicator(rect, 'Gallimimus')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, CarnivoreAgent):
                        color = COLORS['carnivore']
                        draw_movement_trail(self.screen, CELL_SIZE, agent, self.camera, color)
                        direction = getattr(agent, 'direction', (1, 0))
                        draw_dinosaur(self.screen, rect, color, direction, 0.4)
                        if is_learning_agent:
                            self._draw_ai_indicator(rect, 'CarnivoreAgent')
                        drawn_agent = agent
                        break
                    elif isinstance(agent, HerbivoreAgent):
                        color = COLORS['herbivore']
                        draw_movement_trail(self.screen, CELL_SIZE, agent, self.camera, color)
                        direction = getattr(agent, 'direction', (1, 0))
                        draw_dinosaur(self.screen, rect, color, direction, 0.3)
                        if is_learning_agent:
                            self._draw_ai_indicator(rect, 'HerbivoreAgent')
                        drawn_agent = agent
                        break

                # Draw energy bar
                if drawn_agent and hasattr(drawn_agent, 'energy'):
                    if isinstance(drawn_agent, (HerbivoreAgent, CarnivoreAgent)):
                        max_energy = 150 if isinstance(drawn_agent, CarnivoreAgent) else 100
                        self._draw_energy_bar(rect, drawn_agent.energy, max_energy)

    def _draw_energy_bar(self, rect, energy, max_energy):
        """Draw energy bar above agent"""
        bar_width = CELL_SIZE
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

    def _draw_ai_indicator(self, rect, species_name):
        """Draw AI badge for learning agents"""
        from src.config.colors import AI_BADGE_COLORS

        symbol, color = AI_BADGE_COLORS.get(species_name, ('A', (255, 255, 255)))

        badge_center = (rect.left + int(CELL_SIZE * 0.2), rect.top + int(CELL_SIZE * 0.2))
        badge_radius = int(CELL_SIZE * 0.18)

        # Draw circle background
        circle_surface = pygame.Surface((badge_radius * 2 + 2, badge_radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (0, 0, 0, 180), (badge_radius + 1, badge_radius + 1), badge_radius)
        pygame.draw.circle(circle_surface, color, (badge_radius + 1, badge_radius + 1), badge_radius, 2)
        self.screen.blit(circle_surface, (badge_center[0] - badge_radius - 1, badge_center[1] - badge_radius - 1))

        # Draw symbol
        symbol_font = pygame.font.Font(None, int(CELL_SIZE * 0.8))
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
