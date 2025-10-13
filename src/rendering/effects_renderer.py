"""
Effects Rendering
Weather, day/night cycle, particles
"""

import random
import pygame
from terrain_generator import TerrainType
from src.config.settings import VIEWPORT_WIDTH, VIEWPORT_HEIGHT, MAX_VOLCANO_PARTICLES
from src.entities.particle import VolcanoParticle


class EffectsRenderer:
    """Renders visual effects (weather, lighting, particles)"""

    def __init__(self, screen, camera):
        """
        Initialize effects renderer

        Args:
            screen: Pygame screen surface
            camera: Camera object
        """
        self.screen = screen
        self.camera = camera
        self.time_of_day = 0.5  # 0=midnight, 0.5=noon, 1.0=midnight
        self.volcano_particles = []

    def update_time(self, time_speed):
        """Update time of day"""
        self.time_of_day = (self.time_of_day + time_speed) % 1.0

    def update_particles(self, terrain_map, model_width, model_height):
        """Update volcano particles"""
        if terrain_map is None:
            return

        # Spawn new particles from volcanoes
        if len(self.volcano_particles) < MAX_VOLCANO_PARTICLES:
            for world_y in range(model_height):
                for world_x in range(model_width):
                    terrain_type = TerrainType(terrain_map[world_y, world_x])
                    if terrain_type == TerrainType.VOLCANO and random.random() < 0.01:
                        if self.camera.is_visible(world_x, world_y):
                            self.volcano_particles.append(VolcanoParticle(world_x, world_y))

        # Update existing particles
        for particle in self.volcano_particles[:]:
            particle.update()
            if particle.is_dead():
                self.volcano_particles.remove(particle)

    def draw_rain(self, rainfall):
        """Draw rain effect"""
        if rainfall <= 120:
            return

        rain_surface = pygame.Surface((VIEWPORT_WIDTH, VIEWPORT_HEIGHT), pygame.SRCALPHA)

        for _ in range(int(rainfall / 3)):
            x = random.randint(0, VIEWPORT_WIDTH)
            y = random.randint(0, VIEWPORT_HEIGHT)
            pygame.draw.line(rain_surface, (100, 150, 200, 100), (x, y), (x + 2, y + 5), 1)

        self.screen.blit(rain_surface, (0, 0))

    def draw_heat_shimmer(self, temperature):
        """Draw heat shimmer effect"""
        if temperature <= 32:
            return

        heat_surface = pygame.Surface((VIEWPORT_WIDTH, VIEWPORT_HEIGHT), pygame.SRCALPHA)
        heat_alpha = min(80, int((temperature - 32) * 5))
        heat_surface.fill((255, 100, 50, heat_alpha))
        self.screen.blit(heat_surface, (0, 0))

    def draw_volcano_particles(self):
        """Draw volcano smoke particles"""
        for particle in self.volcano_particles:
            screen_x, screen_y = self.camera.world_to_screen(
                particle.x + particle.offset_x,
                particle.y + particle.offset_y
            )

            # Only draw if visible
            if 0 <= screen_x < VIEWPORT_WIDTH and 0 <= screen_y < VIEWPORT_HEIGHT:
                alpha = particle.get_alpha()
                smoke_surface = pygame.Surface((particle.size * 2, particle.size * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    smoke_surface,
                    (80, 80, 80, alpha),
                    (particle.size, particle.size),
                    particle.size
                )
                self.screen.blit(smoke_surface,
                               (int(screen_x) - particle.size, int(screen_y) - particle.size))

    def draw_day_night_cycle(self):
        """Draw lighting overlay for day/night cycle"""
        # Calculate darkness based on time of day
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
        if darkness > 0.05:
            night_surface = pygame.Surface((VIEWPORT_WIDTH, VIEWPORT_HEIGHT), pygame.SRCALPHA)
            night_alpha = int(darkness * 180)
            night_color = (10, 20, 40, night_alpha)
            night_surface.fill(night_color)
            self.screen.blit(night_surface, (0, 0))

    def draw_all_effects(self, model, terrain_map, model_width, model_height):
        """
        Draw all visual effects

        Args:
            model: Simulation model
            terrain_map: Terrain map array
            model_width: World width
            model_height: World height
        """
        # Weather effects
        self.draw_rain(model.rainfall)
        self.draw_heat_shimmer(model.temperature)

        # Particles
        self.draw_volcano_particles()

        # Lighting
        self.draw_day_night_cycle()

    def reset(self):
        """Reset effects state"""
        self.time_of_day = 0.5
        self.volcano_particles = []
