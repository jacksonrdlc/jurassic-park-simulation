"""
UI Rendering
Panels, legends, minimap, etc.
"""

import pygame
from my_first_island import (GrassAgent, Triceratops, Gallimimus, TRex, Velociraptor)
from terrain_generator import TERRAIN_COLORS, TerrainType
from src.config.colors import COLORS
from src.config.settings import (
    VIEWPORT_WIDTH, VIEWPORT_HEIGHT, PANEL_WIDTH, WINDOW_HEIGHT,
    MINIMAP_WIDTH, MINIMAP_HEIGHT,
    FONT_TITLE_SIZE, FONT_LARGE_SIZE, FONT_MEDIUM_SIZE, FONT_SMALL_SIZE
)


class UIRenderer:
    """Renders UI elements (panel, legend, minimap)"""

    def __init__(self, screen, camera):
        """
        Initialize UI renderer

        Args:
            screen: Pygame screen surface
            camera: Camera object
        """
        self.screen = screen
        self.camera = camera

        # Fonts
        self.font_title = pygame.font.Font(None, FONT_TITLE_SIZE)
        self.font_large = pygame.font.Font(None, FONT_LARGE_SIZE)
        self.font_medium = pygame.font.Font(None, FONT_MEDIUM_SIZE)
        self.font_small = pygame.font.Font(None, FONT_SMALL_SIZE)

        self.show_minimap = True

    def toggle_minimap(self):
        """Toggle minimap visibility"""
        self.show_minimap = not self.show_minimap

    def draw_panel(self, model, step_count, speed, paused, use_learning_agents):
        """
        Draw right-side info panel

        Args:
            model: Simulation model
            step_count: Current step count
            speed: Simulation speed
            paused: Whether simulation is paused
            use_learning_agents: Whether using AI agents
        """
        panel_x = VIEWPORT_WIDTH

        # Background
        pygame.draw.rect(self.screen, COLORS['panel'],
                        (panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT))

        # Yellow border
        pygame.draw.rect(self.screen, COLORS['panel_border'],
                        (panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT), 3)

        y_offset = 15

        # === BRANDING ===
        title = self.font_title.render("JURASSIC PARK", True, COLORS['text_highlight'])
        title_rect = title.get_rect(center=(panel_x + PANEL_WIDTH // 2, y_offset + 20))
        self.screen.blit(title, title_rect)
        y_offset += 50

        subtitle = self.font_small.render("ISLA NUBLAR - SECTOR 7G", True, COLORS['text'])
        subtitle_rect = subtitle.get_rect(center=(panel_x + PANEL_WIDTH // 2, y_offset))
        self.screen.blit(subtitle, subtitle_rect)
        y_offset += 25

        # Warning stripe separator
        for i in range(0, PANEL_WIDTH, 40):
            stripe_color = COLORS['jp_yellow'] if (i // 40) % 2 == 0 else COLORS['jp_black']
            pygame.draw.rect(self.screen, stripe_color, (panel_x + i, y_offset, 40, 8))
        y_offset += 18

        # === CONTAINMENT STATUS ===
        y_offset = self._draw_containment_status(panel_x, y_offset, model)

        # Separator
        pygame.draw.line(self.screen, COLORS['jp_yellow'],
                        (panel_x + 20, y_offset),
                        (panel_x + PANEL_WIDTH - 20, y_offset), 2)
        y_offset += 20

        # === SYSTEM INFO ===
        y_offset = self._draw_system_info(panel_x, y_offset, step_count, speed, paused, use_learning_agents)

        # === POPULATION ===
        y_offset = self._draw_population(panel_x, y_offset, model)

        # === EVENT LOG ===
        y_offset = self._draw_event_log(panel_x, y_offset, model)

        # === CONTROLS ===
        self._draw_controls(panel_x)

    def _draw_containment_status(self, panel_x, y_offset, model):
        """Draw containment status indicator"""
        tri_count = len([a for a in model.agents if isinstance(a, Triceratops)])
        gal_count = len([a for a in model.agents if isinstance(a, Gallimimus)])
        trex_count = len([a for a in model.agents if isinstance(a, TRex)])
        vel_count = len([a for a in model.agents if isinstance(a, Velociraptor)])
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
        status_rect = status.get_rect(center=(panel_x + PANEL_WIDTH // 2, y_offset))
        self.screen.blit(status, status_rect)

        return y_offset + 35

    def _draw_system_info(self, panel_x, y_offset, step_count, speed, paused, use_learning_agents):
        """Draw system status"""
        system_header = self.font_medium.render("SYSTEM STATUS", True, COLORS['text_highlight'])
        self.screen.blit(system_header, (panel_x + 20, y_offset))
        y_offset += 30

        agent_type = "AI (LEARNING)" if use_learning_agents else "TRADITIONAL"

        system_stats = [
            f"Step: {step_count:,}",
            f"Speed: {speed}x",
            f"Mode: {agent_type}",
            f"Status: {'PAUSED' if paused else 'RUNNING'}",
        ]

        for stat in system_stats:
            text = self.font_small.render(stat, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 25, y_offset))
            y_offset += 20

        return y_offset + 10

    def _draw_population(self, panel_x, y_offset, model):
        """Draw population counts"""
        pop_header = self.font_medium.render("POPULATION", True, COLORS['text_highlight'])
        self.screen.blit(pop_header, (panel_x + 20, y_offset))
        y_offset += 30

        # Herbivores
        herb_label = self.font_small.render("HERBIVORES:", True, COLORS['text'])
        self.screen.blit(herb_label, (panel_x + 25, y_offset))
        y_offset += 22

        tri_count = len([a for a in model.agents if isinstance(a, Triceratops)])
        gal_count = len([a for a in model.agents if isinstance(a, Gallimimus)])

        pygame.draw.circle(self.screen, COLORS['triceratops'], (panel_x + 35, y_offset + 6), 5)
        tri_text = self.font_small.render(f"Triceratops: {tri_count}", True, COLORS['text'])
        self.screen.blit(tri_text, (panel_x + 50, y_offset))
        y_offset += 20

        pygame.draw.circle(self.screen, COLORS['gallimimus'], (panel_x + 35, y_offset + 6), 5)
        gal_text = self.font_small.render(f"Gallimimus: {gal_count}", True, COLORS['text'])
        self.screen.blit(gal_text, (panel_x + 50, y_offset))
        y_offset += 25

        # Carnivores
        carn_label = self.font_small.render("CARNIVORES:", True, COLORS['text'])
        self.screen.blit(carn_label, (panel_x + 25, y_offset))
        y_offset += 22

        trex_count = len([a for a in model.agents if isinstance(a, TRex)])
        vel_count = len([a for a in model.agents if isinstance(a, Velociraptor)])

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

        g_count = len([a for a in model.agents if isinstance(a, GrassAgent) and a.energy > 0])
        pygame.draw.rect(self.screen, COLORS['grass_full'], (panel_x + 30, y_offset + 3, 10, 10))
        grass_text = self.font_small.render(f"Vegetation: {g_count}", True, COLORS['text'])
        self.screen.blit(grass_text, (panel_x + 50, y_offset))
        y_offset += 20

        temp_text = self.font_small.render(f"Temp: {model.temperature}°C", True, COLORS['text'])
        self.screen.blit(temp_text, (panel_x + 50, y_offset))
        y_offset += 18

        rain_text = self.font_small.render(f"Rainfall: {model.rainfall}mm", True, COLORS['text'])
        self.screen.blit(rain_text, (panel_x + 50, y_offset))
        y_offset += 20

        # Separator
        y_offset += 10
        pygame.draw.line(self.screen, COLORS['text'],
                        (panel_x + 20, y_offset),
                        (panel_x + PANEL_WIDTH - 20, y_offset), 1)
        y_offset += 20

        return y_offset

    def _draw_event_log(self, panel_x, y_offset, model):
        """Draw event log"""
        log_title = self.font_medium.render("EVENT LOG", True, COLORS['text'])
        self.screen.blit(log_title, (panel_x + 20, y_offset))
        y_offset += 30

        max_log_y = WINDOW_HEIGHT - 180

        if hasattr(model, 'event_log'):
            events = list(reversed(model.event_log[-25:]))

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

        return y_offset

    def _draw_controls(self, panel_x):
        """Draw controls section (fixed at bottom)"""
        y_offset = WINDOW_HEIGHT - 160
        pygame.draw.line(self.screen, COLORS['text'],
                        (panel_x + 20, y_offset),
                        (panel_x + PANEL_WIDTH - 20, y_offset), 1)
        y_offset += 20

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

    def draw_minimap(self, terrain_map, model_width, model_height):
        """Draw minimap overlay"""
        if not self.show_minimap or terrain_map is None:
            return

        minimap_x = 10
        minimap_y = VIEWPORT_HEIGHT - MINIMAP_HEIGHT - 10

        # Background
        minimap_bg = pygame.Surface((MINIMAP_WIDTH, MINIMAP_HEIGHT))
        minimap_bg.set_alpha(220)
        minimap_bg.fill(COLORS['panel'])
        self.screen.blit(minimap_bg, (minimap_x, minimap_y))

        # Border
        pygame.draw.rect(self.screen, COLORS['panel_border'],
                        (minimap_x, minimap_y, MINIMAP_WIDTH, MINIMAP_HEIGHT), 2)

        # Scale factors
        scale_x = (MINIMAP_WIDTH - 4) / model_width
        scale_y = (MINIMAP_HEIGHT - 4) / model_height

        # Draw simplified terrain
        step = max(1, int(1 / min(scale_x, scale_y)))
        for y in range(0, model_height, step):
            for x in range(0, model_width, step):
                terrain_type = TerrainType(terrain_map[y, x])
                terrain_color = TERRAIN_COLORS[terrain_type]

                mini_x = minimap_x + 2 + int(x * scale_x)
                mini_y = minimap_y + 2 + int(y * scale_y)
                mini_w = max(1, int(step * scale_x))
                mini_h = max(1, int(step * scale_y))

                pygame.draw.rect(self.screen, terrain_color,
                               (mini_x, mini_y, mini_w, mini_h))

        # Draw viewport indicator
        viewport_rect = self.camera.minimap_coords(MINIMAP_WIDTH - 4, MINIMAP_HEIGHT - 4)
        viewport_rect.x += minimap_x + 2
        viewport_rect.y += minimap_y + 2
        pygame.draw.rect(self.screen, COLORS['panel_border'], viewport_rect, 2)

        # Title
        title = self.font_small.render("MAP", True, COLORS['text_highlight'])
        self.screen.blit(title, (minimap_x + 5, minimap_y + 3))
